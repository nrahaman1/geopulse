"""From candidates to GeoPulse cases: merge duplicates, plan the study area and windows, check imagery, store.

Everything here is deterministic: the LLM only read the articles (reader.py). A case has the same shape as an
example event (title, task, aoi, before, after, sensors), so the web app and `geopulse infer` take it as is.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import pipeline, stac

MERGE_KM = 50  # same hazard this close, and starting within MERGE_DAYS, is one event
MERGE_DAYS = 10
KEEP_DAYS = 45  # cases not updated for this long are dropped
POINT_KM = 15.0  # a case known only by a point: 15 x 15 km (runs in the built-in browser engine too)
MAX_SIDE_KM = 30.0  # official extents are kept up to 30 x 30 km (the PC engine takes up to 1500 km²)
MIN_SIDE_KM = 5.0
PRIORITY = {"cems": 0, "gdacs": 1, "eonet": 2, "news": 3}  # whose geometry wins when sources disagree


def scout_dir() -> Path:
    return Path(os.environ.get("GEOPULSE_SCOUT", Path(os.environ.get("GEOPULSE_OUTPUTS", "outputs")) / "scout"))


def km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    (lon1, lat1), (lon2, lat2) = map(lambda p: (math.radians(p[0]), math.radians(p[1])), (a, b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def box(lon: float, lat: float, w_km: float, h_km: float) -> dict:
    dx, dy = w_km / 2 / (111.32 * math.cos(math.radians(lat))), h_km / 2 / 110.574
    ring = [
        [lon - dx, lat - dy],
        [lon + dx, lat - dy],
        [lon + dx, lat + dy],
        [lon - dx, lat + dy],
        [lon - dx, lat - dy],
    ]
    return {"type": "Polygon", "coordinates": [[[round(x, 5), round(y, 5)] for x, y in ring]]}


# --------------------------------------------------------------------------- merge


def merge(cases: list[dict], candidates: list[dict], today: dt.date) -> list[dict]:
    """Fold candidates into cases: an event already known gains the source; a new one becomes a case."""
    for c in sorted(candidates, key=lambda c: PRIORITY[c["source"]["kind"]]):
        start = c["start"].isoformat()
        match = next(
            (
                k
                for k in cases
                if k["hazard"] == c["hazard"]
                and km_between((k["lon"], k["lat"]), (c["lon"], c["lat"])) <= MERGE_KM
                and abs((dt.date.fromisoformat(k["event_start"]) - c["start"]).days) <= MERGE_DAYS
            ),
            None,
        )
        if match is None:
            match = {
                "id": f"{c['hazard'][:2]}-{c['start']:%Y%m%d}-{c['lat']:.2f}_{c['lon']:.2f}",
                "hazard": c["hazard"],
                "lon": c["lon"],
                "lat": c["lat"],
                "bbox": c["bbox"],
                "geometry_from": c["source"]["kind"],
                "event_start": start,
                "event_end": c["end"].isoformat(),
                "place": c["place"],
                "country": c["country"],
                "sources": [],
                "first_seen": today.isoformat(),
            }
            cases.append(match)
        key = (c["source"]["kind"], c["source"].get("id") or c["source"].get("url"))
        if key not in {(s["kind"], s.get("id") or s.get("url")) for s in match["sources"]}:
            match["sources"].append(c["source"])
            match["updated"] = today.isoformat()
        if PRIORITY[c["source"]["kind"]] < PRIORITY[match["geometry_from"]] or (c["bbox"] and not match["bbox"]):
            match.update(lon=c["lon"], lat=c["lat"], bbox=c["bbox"] or match["bbox"], geometry_from=c["source"]["kind"])
        match["event_end"] = max(match["event_end"], c["end"].isoformat())
        match["place"] = match["place"] or c["place"]
        match["country"] = match["country"] or c["country"]
    return cases


# --------------------------------------------------------------------------- plan


def windows(hazard: str, start: dt.date, end: dt.date, today: dt.date) -> tuple[tuple, tuple] | None:
    """(before, after) date windows following each task's rules, or None if the event is too recent to map."""
    d = dt.timedelta
    if hazard == "flood":
        # Dry weeks before it began; the days after the latest report (floods recede, and a long-running flood is
        # mapped as it is now, not as it was weeks ago).
        first = max(start, end - d(10))
        before, after = (start - d(35), start - d(5)), (first, min(first + d(14), today))
    elif hazard == "wildfire":  # weeks before ignition; after the fire, or the last month while it still burns
        ongoing = end >= today - d(2)
        first = max(start + d(2), today - d(30)) if ongoing else end + d(1)
        before, after = (start - d(40), start - d(2)), (first, min(first + d(40), today))
    else:  # vegetation: the same season one year apart, so phenology cancels
        after = (start - d(60), min(start, today))
        before = (after[0] - d(365), after[1] - d(365))
    return (before, after) if after[0] <= after[1] else None


def plan(case: dict, today: dt.date) -> dict:
    """Study area, windows and title. The request passes GeoPulse's own validation (make_request)."""
    if case.get("bbox"):
        w, s, e, n = case["bbox"]
        lat = (s + n) / 2
        w_km = (e - w) * 111.32 * math.cos(math.radians(lat))
        h_km = (n - s) * 110.574
        clip = lambda v: min(max(v, MIN_SIDE_KM), MAX_SIDE_KM)  # noqa: E731
        # An extent that fits is mapped whole; a larger one around the reported event point, kept inside it.
        cx = (w + e) / 2 if w_km <= MAX_SIDE_KM else min(max(case["lon"], w), e)
        cy = lat if h_km <= MAX_SIDE_KM else min(max(case["lat"], s), n)
        aoi = box(cx, cy, clip(w_km), clip(h_km))
    else:
        aoi = box(case["lon"], case["lat"], POINT_KM, POINT_KM)
    win = windows(case["hazard"], dt.date.fromisoformat(case["event_start"]),
                  dt.date.fromisoformat(case["event_end"]), today)  # fmt: skip
    label = {"flood": "Flood", "wildfire": "Wildfire", "vegetation": "Forest loss"}[case["hazard"]]
    where = ", ".join(dict.fromkeys(p for p in (case.get("place"), case.get("country")) if p)) or "unknown place"
    case["title"] = f"{label} · {where} ({case['event_start']})"
    case["task"] = case["hazard"]
    if win is None:
        case.update(status="too recent", aoi=aoi)
        return case
    req = pipeline.make_request(aoi, [str(x) for x in win[0]], [str(x) for x in win[1]], case["hazard"])
    case.update(aoi=req["aoi"], aoi_km2=req["aoi_km2"], before=req["before"], after=req["after"])
    return case


def check_imagery(case: dict) -> dict:
    """Scene counts per sensor and window; the case is ready when some sensor covers both windows."""
    if "before" not in case:
        return case
    counts = {}
    for sensor in ("s1", "s2"):
        for period in ("before", "after"):
            items = stac.search(sensor, case["aoi"], case[period], max_cloud=60 if sensor == "s2" else None)
            counts[f"{sensor}_{period}"] = len(items)
    usable = [s for s in ("s1", "s2") if counts[f"{s}_before"] and counts[f"{s}_after"]]
    case["scenes"] = counts
    case["sensors"] = usable or ["s1", "s2"]
    after = counts["s1_after"] or counts["s2_after"]
    case["status"] = "ready" if usable else "partial imagery" if after else "waiting for imagery"
    return case


def check_all(selected: list[dict], log=print) -> None:
    """check_imagery for many cases at once (each is four catalog searches; network-bound, so threads)."""

    def one(case):
        try:
            check_imagery(case)
        except (OSError, ValueError) as e:
            log(f"! imagery check for {case['id']}: {e}")

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, selected))


def score(case: dict) -> dict:
    kinds = {s["kind"] for s in case["sources"]}
    outlets = {s.get("outlet") for s in case["sources"] if s["kind"] == "news"}
    official = 0.5 * ("cems" in kinds) + 0.35 * bool(kinds & {"gdacs", "eonet"})
    value = official + min(0.45, 0.15 * len(outlets))
    case["score"] = round(value, 2)
    case["confidence"] = "high" if value >= 0.6 else "medium" if value >= 0.35 else "low"
    return case


# --------------------------------------------------------------------------- store


def load_cases() -> list[dict]:
    path = scout_dir() / "cases.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_cases(cases: list[dict]) -> Path:
    path = scout_dir() / "cases.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    order = {"ready": 0, "partial imagery": 1, "waiting for imagery": 2, "too recent": 3}
    cases.sort(key=lambda c: c["event_start"], reverse=True)  # newest first within equal status and score
    cases.sort(key=lambda c: (order.get(c.get("status"), 4), -c.get("score", 0)))
    path.write_text(json.dumps(cases, indent=1, default=str), encoding="utf-8")
    return path
