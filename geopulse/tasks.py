"""The change tasks GeoPulse maps. One place defines naming, temporal compositing and seasonal rules."""

from __future__ import annotations

TASKS = {
    "flood": {
        "title": "Flood inundation",
        "target": "flood",  # output names: flood_probability.tif, flood_extent.geojson
        "verb": "flooded",
        # pre = median (stable baseline); post = earliest valid observation (floods recede).
        "composite": ("median", "first"),
        "scenes": "closest",  # scenes nearest the event
        "seasonal": False,
    },
    "wildfire": {
        "title": "Wildfire / burn",
        "target": "burn",
        "verb": "burned",
        # Burn scars persist for months; a post-fire median suppresses residual smoke and cloud.
        "composite": ("median", "median"),
        "scenes": "closest",
        "seasonal": False,
    },
    "vegetation": {
        "title": "Vegetation disturbance",
        "target": "disturbance",
        "verb": "disturbed",
        "composite": ("median", "median"),
        # Compare like season with like season, or senescence looks like disturbance: sample both windows evenly.
        "scenes": "spread",
        "seasonal": True,
    },
}
