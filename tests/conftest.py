import numpy as np
import pytest

from geopulse.data import Inputs
from geopulse.grid import bbox_geometry, make_grid

AOI = bbox_geometry(-79.10, 34.58, -79.08, 34.60)  # ~1.8 x 2.2 km
EVENT = (slice(50, 110), slice(60, 140))  # where the task's change happens
CONTROL = (slice(150, 170), slice(20, 60))  # a look-alike that must NOT be mapped

# Surface classes: Sentinel-2 (B02 B03 B04 B08 B11 B12 reflectance) and Sentinel-1 (VV, VH dB).
SURFACES = {
    "land": ([0.04, 0.07, 0.05, 0.30, 0.22, 0.12], (-9.0, -15.0)),
    "water": ([0.05, 0.07, 0.05, 0.03, 0.01, 0.01], (-23.0, -29.0)),
    "forest": ([0.03, 0.06, 0.04, 0.35, 0.18, 0.08], (-8.0, -14.0)),
    "burned": ([0.05, 0.06, 0.07, 0.12, 0.20, 0.20], (-9.0, -17.0)),
    "clearcut": ([0.07, 0.10, 0.12, 0.22, 0.30, 0.22], (-10.0, -18.5)),
    "bare": ([0.18, 0.22, 0.26, 0.28, 0.40, 0.36], (-12.0, -20.0)),  # NDVI ≈ 0.04
    "bare_dark": ([0.10, 0.14, 0.18, 0.20, 0.30, 0.34], (-12.0, -20.0)),  # dNBR ≈ 0.13 but nothing to burn
    "sagebrush": ([0.06, 0.09, 0.10, 0.15, 0.24, 0.18], (-11.0, -19.0)),  # NDVI ≈ 0.20: sparse, but it burns
    "sagebrush_burned": ([0.05, 0.06, 0.07, 0.10, 0.20, 0.20], (-11.5, -19.5)),
    "grass": ([0.05, 0.08, 0.08, 0.18, 0.20, 0.12], (-11.0, -19.0)),
    "grass_dry": ([0.08, 0.10, 0.12, 0.18, 0.28, 0.20], (-11.0, -19.5)),  # senescence, not disturbance
}
# task: (background, event before -> after, control before -> after)
SCENARIOS = {
    "flood": ("land", ("land", "water"), ("water", "water")),  # control = permanent water
    "wildfire": ("forest", ("forest", "burned"), ("bare", "bare_dark")),
    "vegetation": ("forest", ("forest", "clearcut"), ("grass", "grass_dry")),
}


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOPULSE_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("GEOPULSE_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("GEOPULSE_CACHE", str(tmp_path / "cache"))


def synthetic_inputs(task: str = "flood", sensors=("s1", "s2")) -> Inputs:
    grid = make_grid(AOI)
    h, w = grid.height, grid.width
    rng = np.random.default_rng(0)
    background, event, control = SCENARIOS[task]

    def scene(period: int) -> dict[str, np.ndarray]:
        s2 = np.empty((6, h, w), "float32")
        s1 = np.empty((2, h, w), "float32")
        for region, surface in (
            ((slice(None), slice(None)), background),
            (EVENT, event[period]),
            (CONTROL, control[period]),
        ):
            spectrum, backscatter = SURFACES[surface]
            s2[(slice(None), *region)] = np.array(spectrum)[:, None, None]
            s1[(slice(None), *region)] = np.array(backscatter)[:, None, None]
        return {"s2": s2 + rng.normal(0, 0.003, s2.shape), "s1": s1 + rng.normal(0, 0.5, s1.shape)}

    pre, post = scene(0), scene(1)
    arrays = {"dem": np.stack([np.full((h, w), 40.0), np.full((h, w), 1.0)]).astype("float32")}
    for s in sensors:
        arrays |= {f"{s}_pre": pre[s].astype("float32"), f"{s}_post": post[s].astype("float32")}
    scenes = {k: [{"id": f"synthetic-{k}"}] for k in arrays}
    return Inputs(grid, arrays, scenes, [])
