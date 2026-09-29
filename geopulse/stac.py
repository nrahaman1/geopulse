"""STAC discovery. Provider details live here; the rest of GeoPulse asks for sensors, not collections."""

from __future__ import annotations

import planetary_computer
from pystac import Item
from pystac_client import Client

PROVIDER_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# GeoPulse sensor key -> STAC collection on the provider.
COLLECTIONS = {
    "s1": "sentinel-1-rtc",  # radiometrically terrain-corrected gamma0, linear power, UTM
    "s2": "sentinel-2-l2a",  # surface reflectance + scene classification (SCL)
    "dem": "cop-dem-glo-30",  # Copernicus GLO-30 elevation
    "mtbs": "mtbs",  # label source: USGS/USFS burn severity mosaics, CONUS, 1984-2018
}


def search(
    sensor: str,
    geometry: dict,
    datetime: str | None = None,
    max_cloud: float | None = None,
    limit: int = 100,
) -> list[Item]:
    """Return signed STAC items for `sensor` intersecting `geometry` (GeoJSON, EPSG:4326), oldest first."""
    if sensor not in COLLECTIONS:
        raise ValueError(f"unknown sensor {sensor!r}; expected one of {sorted(COLLECTIONS)}")
    client = Client.open(PROVIDER_URL, modifier=planetary_computer.sign_inplace)
    client.add_conforms_to("FILTER")  # the provider supports CQL2 but does not advertise it
    cql = None
    if sensor == "s2" and max_cloud is not None:
        cql = {"op": "<", "args": [{"property": "eo:cloud_cover"}, max_cloud]}
    items = client.search(
        collections=[COLLECTIONS[sensor]],
        intersects=geometry,
        datetime=None if sensor == "dem" else datetime,
        filter=cql,
        filter_lang="cql2-json" if cql else None,
        max_items=limit,
    ).item_collection()
    return sorted(items, key=lambda i: i.datetime.isoformat() if i.datetime else "")


def describe(item: Item) -> dict:
    """Small JSON-safe summary of an item for logs, provenance and the API."""
    p = item.properties
    return {
        "id": item.id,
        "collection": item.collection_id,
        "datetime": item.datetime.isoformat() if item.datetime else None,
        "cloud_cover": p.get("eo:cloud_cover"),
        "orbit": p.get("sat:orbit_state"),
        "relative_orbit": p.get("sat:relative_orbit"),
        "platform": p.get("platform"),
    }
