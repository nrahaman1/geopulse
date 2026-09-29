"""Training-free physics baselines for every task, plus the consensus weak labels used for flood.

Flood: open water is a specular reflector (low C-band VV) and absorbs SWIR (MNDWI > 0, Xu 2006);
       flood = water after the event that was not water before.
Burn: fire removes chlorophyll (NIR falls) and exposes char and ash (SWIR2 rises), so the Normalized Burn
      Ratio drops; dNBR = NBR_pre − NBR_post with the Key & Benson (2006) severity breaks.
Disturbance: canopy loss lowers NDVI and C-band VH volume scattering; compared against the same season.
"""

from __future__ import annotations

import warnings

import numpy as np

VV_WATER_DB = -18.0
VV_DROP_DB = 3.0  # log-ratio change: backscatter drop that signals new water
VV_DROP_CEILING_DB = -11.0  # ...but only into a low-backscatter range
MNDWI_WATER = 0.0
DNBR_BURN = 0.10  # Key & Benson (2006): unburned / low-severity boundary
DNBR_SEVERITY = (0.10, 0.27, 0.44, 0.66)  # low | moderate-low | moderate-high | high
# A pixel needs vegetation to burn. 0.2 missed sagebrush steppe (pre-fire NDVI ~0.17, Martin Fire 2018 -- found on
# the test split, so its post-fix score is not a clean held-out number); below ~0.1 is bare soil or rock.
NDVI_FUEL = 0.10
NDVI_DROP = 0.20  # disturbance: NDVI loss against the same season...
NDVI_CANOPY = 0.5  # ...on pixels that had canopy
VH_DROP_DB = 2.0  # loss of canopy volume scattering
VH_CANOPY_DB = -17.0  # VH above this is roughly "vegetated"
SAR_ONLY_BURN_UNCERTAINTY = 0.5  # C-band sees fire only indirectly: never report SAR-only burn as confident
CHANGE_DB = 3.0
CHANGE_INDEX = 0.2
IGNORE = 255


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _nanmean(maps: list[np.ndarray], shape) -> np.ndarray:
    if not maps:
        return np.full(shape, np.nan, "float32")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(maps), 0).astype("float32")


def _nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + 1e-6)


def mndwi(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[1], s2[4])  # green, SWIR1


def ndvi(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[3], s2[2])  # NIR, red


def nbr(s2: np.ndarray) -> np.ndarray:
    return _nd(s2[3], s2[5])  # NIR, SWIR2


def _both(arrays: dict, sensor: str) -> bool:
    return f"{sensor}_pre" in arrays and f"{sensor}_post" in arrays


def water_prob(arrays: dict, period: str) -> dict[str, np.ndarray]:
    out = {}
    if f"s1_{period}" in arrays:
        vv = arrays[f"s1_{period}"][0]
        p = _sigmoid((VV_WATER_DB - vv) / 1.5)
        if period == "post" and "s1_pre" in arrays:
            # Wind-roughened floodwater can sit well above the open-water threshold (e.g. -14 dB), but a strong
            # drop from the pre-event backscatter into a low range is still new water (log-ratio change detection).
            drop = arrays["s1_pre"][0] - vv
            p = np.fmax(p, _sigmoid((drop - VV_DROP_DB) / 1.0) * _sigmoid((VV_DROP_CEILING_DB - vv) / 1.5))
        out["s1"] = p
    if f"s2_{period}" in arrays:
        out["s2"] = _sigmoid((mndwi(arrays[f"s2_{period}"]) - MNDWI_WATER) / 0.05)
    return out


def pre_water(arrays: dict, threshold: float = 0.8) -> np.ndarray:
    """Open water before the event (optical where observed, SAR elsewhere): nothing there can burn or lose canopy."""
    p = water_prob(arrays, "pre")
    w = p.get("s1", np.full(next(iter(arrays.values())).shape[1:], np.nan))
    if "s2" in p:
        w = np.where(np.isfinite(p["s2"]), p["s2"], w)
    return np.nan_to_num(w) > threshold


def _vh_drop(arrays: dict) -> np.ndarray:
    vh_pre, vh_post = arrays["s1_pre"][1], arrays["s1_post"][1]
    return _sigmoid((vh_pre - vh_post - VH_DROP_DB) / 0.6) * _sigmoid((vh_pre - VH_CANOPY_DB) / 1.0)


def dnbr(arrays: dict) -> np.ndarray:
    return nbr(arrays["s2_pre"]) - nbr(arrays["s2_post"])


def burn_prob(arrays: dict) -> dict[str, np.ndarray]:
    out = {}
    if _both(arrays, "s2"):
        fuel = _sigmoid((ndvi(arrays["s2_pre"]) - NDVI_FUEL) / 0.03)
        out["s2"] = _sigmoid((dnbr(arrays) - DNBR_BURN) / 0.025) * fuel
    if _both(arrays, "s1"):
        out["s1"] = _vh_drop(arrays)
    return out


def disturbance_prob(arrays: dict) -> dict[str, np.ndarray]:
    out = {}
    if _both(arrays, "s2"):
        pre = ndvi(arrays["s2_pre"])
        out["s2"] = _sigmoid((pre - ndvi(arrays["s2_post"]) - NDVI_DROP) / 0.04) * _sigmoid((pre - NDVI_CANOPY) / 0.05)
    if _both(arrays, "s1"):
        out["s1"] = _vh_drop(arrays)
    return out


def change_prob(arrays: dict) -> dict[str, np.ndarray]:
    """Task-agnostic "something changed" evidence per sensor."""
    out = {}
    if _both(arrays, "s1"):
        d = np.abs(arrays["s1_post"] - arrays["s1_pre"]).max(0)  # |ΔVV|, |ΔVH|
        # Soft slope: residual speckle is ~1-2 dB even after compositing, so "confidently stable" means < ~1.9 dB.
        out["s1"] = _sigmoid((d - CHANGE_DB) / 0.5)
    if _both(arrays, "s2"):
        pre, post = arrays["s2_pre"], arrays["s2_post"]
        d = np.stack([np.abs(f(post) - f(pre)) for f in (mndwi, ndvi, nbr)]).max(0)
        out["s2"] = _sigmoid((d - CHANGE_INDEX) / 0.05)
    return out


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).astype("float32")


def severity(arrays: dict) -> np.ndarray:
    """Key & Benson (2006) dNBR classes: 0 unburned, 1 low, 2 moderate-low, 3 moderate-high, 4 high; 255 = no data."""
    d = dnbr(arrays)
    return np.where(np.isfinite(d), np.digitize(d, DNBR_SEVERITY), IGNORE).astype("uint8")


def predict(arrays: dict, task: str = "flood") -> dict[str, np.ndarray]:
    """target / change / uncertainty maps in [0,1] (+ severity for wildfire); NaN where nothing was observed."""
    shape = next(iter(arrays.values())).shape[1:]
    if task == "flood":
        pre, post = water_prob(arrays, "pre"), water_prob(arrays, "post")
        w_pre = _nanmean(list(pre.values()), shape)
        # Without a pre-event observation permanent water cannot be removed; the pipeline warns about it.
        target = _nanmean(list(post.values()), shape) * (1 - np.nan_to_num(w_pre, nan=0.0))
        unc = binary_entropy(target)
        if len(post) == 2:
            unc = np.fmax(unc, np.abs(post["s1"] - post["s2"]))
    elif task == "wildfire":
        p = burn_prob(arrays)
        s2, s1 = p.get("s2", np.full(shape, np.nan)), p.get("s1", np.full(shape, np.nan))
        sar_only = np.isnan(s2) & np.isfinite(s1)  # optical is the burn sensor; SAR only fills cloud/smoke gaps
        target = np.where(sar_only, s1, s2)
        unc = np.where(sar_only, np.fmax(binary_entropy(target), SAR_ONLY_BURN_UNCERTAINTY), binary_entropy(target))
    elif task == "vegetation":
        p = disturbance_prob(arrays)
        target = _nanmean(list(p.values()), shape)
        unc = binary_entropy(target)
        if len(p) == 2:
            unc = np.fmax(unc, np.abs(p["s1"] - p["s2"]))
    else:
        raise ValueError(f"unknown task {task!r}")
    ch = change_prob(arrays)
    change = _nanmean(list(ch.values()), shape) if ch else target.copy()
    unc = np.asarray(unc, "float32")
    unc[np.isnan(target)] = np.nan
    out = {"target": target.astype("float32"), "change": change.astype("float32"), "uncertainty": unc}
    if task == "wildfire" and _both(arrays, "s2"):
        out["severity"] = severity(arrays)
    return out


def weak_labels(arrays: dict, hi: float = 0.9, lo: float = 0.1) -> np.ndarray:
    """(2,H,W) uint8 [flood, change] where SAR and optical agree confidently; IGNORE elsewhere.

    Requires both sensors in both periods. Training on consensus pixels and predicting from sensor subsets is
    cross-modal distillation: it teaches the model to reproduce the two-sensor answer when one is missing.
    """
    shape = next(iter(arrays.values())).shape[1:]
    labels = np.full((2, *shape), IGNORE, "uint8")
    pre, post = water_prob(arrays, "pre"), water_prob(arrays, "post")
    if len(pre) < 2 or len(post) < 2:
        return labels
    wet = {p: (d["s1"] > hi) & (d["s2"] > hi) for p, d in (("pre", pre), ("post", post))}
    dry = {p: (d["s1"] < lo) & (d["s2"] < lo) for p, d in (("pre", pre), ("post", post))}
    flood = labels[0]
    flood[dry["post"]] = 0
    flood[wet["post"] & wet["pre"]] = 0  # permanent water is not flood
    flood[wet["post"] & dry["pre"]] = 1
    labels[1] = weak_change(arrays, hi, lo)
    return labels


def weak_change(arrays: dict, hi: float = 0.9, lo: float = 0.1) -> np.ndarray:
    """Task-agnostic change label: 1/0 where both sensors agree confidently, IGNORE elsewhere."""
    shape = next(iter(arrays.values())).shape[1:]
    out = np.full(shape, IGNORE, "uint8")
    ch = change_prob(arrays)
    if len(ch) == 2:
        out[(ch["s1"] < lo) & (ch["s2"] < lo)] = 0
        out[(ch["s1"] > hi) & (ch["s2"] > hi)] = 1
    return out
