"""Where the Scout hears about events. All open, no keys.

Official alerts (candidates with a location: {hazard, lon, lat, bbox, start, end, place, country, source}):
- GDACS (UN/EC Global Disaster Alert and Coordination System): floods (GloFAS) and wildfires (GWIS) with a point and
  an affected-area polygon.
- Copernicus EMS rapid mapping: activations with expert-drawn areas of interest.
- NASA EONET: natural events, today mostly US wildfires (IRWIN) with their size.

News (articles for reader.py to read: {hazard, url, title, date, outlet}), from two independent channels so that one
refusing (GDELT rate-limits busy addresses) never leaves the Scout without news:
- GDELT DOC 2.0: news worldwide, one combined query per run (GDELT allows one request every 5 seconds);
- topic RSS feeds (NEWS_FEEDS), which exist to be read by programs.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from .. import __version__

UA = f"GeoPulse-Scout/{__version__} (+https://github.com/nrahaman1/geopulse)"
GDACS = "https://www.gdacs.org/gdacsapi/api"
CEMS = "https://rapidmapping.emergency.copernicus.eu/backend/dashboard-api"
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_FIRE_HA = 10_000  # GDACS lists ~100 routine "green" savanna fires a week; smaller ones need an orange alert
HAZARDS = {"FL": "flood", "WF": "wildfire", "Flood": "flood", "Wildfire": "wildfire"}
EONET = "https://eonet.gsfc.nasa.gov/api/v3/events"
MIN_FIRE_ACRES = 1000  # EONET (US) fire sizes are in acres
# One GDELT query for all three hazards; each article is then sorted by the words in its title.
NEWS_QUERY = (
    '(flood OR flooding OR "flash flood" OR wildfire OR "forest fire" OR bushfire OR deforestation '
    'OR "illegal logging" OR "forest loss")'
)
HAZARD_WORDS = {
    "vegetation": r"deforest|logging|forest (loss|clearing)|clear-?cut|land clearing",
    "wildfire": r"wildfire|bush ?fire|forest fire|brush ?fire|grass ?fire|blaze|burn",
    "flood": r"flood|inundat|deluge|overflow|submerg",
}
NEWS_FEEDS = [  # (url, hazard, title words that make an item worth reading; None = every item)
    ("https://wildfiretoday.com/feed/", "wildfire", None),
    ("https://inciweb.wildfire.gov/incidents/rss.xml", "wildfire", None),
    ("https://news.mongabay.com/feed/", "vegetation", HAZARD_WORDS["vegetation"]),
]


def get(url: str, params: dict | None = None, tries: int = 3, timeout: int = 30) -> bytes:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                return r.read()
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(2**i)
    raise AssertionError("unreachable")


def get_json(url: str, params: dict | None = None):
    return json.loads(get(url, params))


def bbox_of(coords) -> list[float] | None:
    """[west, south, east, north] of any nested coordinate list, or of the "x y" pairs in a WKT string."""
    if isinstance(coords, str):
        pts = [(float(x), float(y)) for x, y in re.findall(r"(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?)", coords)]
    else:
        pts, stack = [], [coords]
        while stack:
            c = stack.pop()
            if len(c) >= 2 and all(isinstance(v, int | float) for v in c[:2]):
                pts.append((c[0], c[1]))
            else:
                stack.extend(c)
    if not pts:
        return None
    xs, ys = zip(*pts, strict=True)
    return [min(xs), min(ys), max(xs), max(ys)]


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s[:10])


def candidate(hazard, lon, lat, start, end, place, country, source, bbox=None) -> dict:
    return {
        "hazard": hazard,
        "lon": round(float(lon), 5),
        "lat": round(float(lat), 5),
        "bbox": bbox,
        "start": start,
        "end": end,
        "place": place,
        "country": country,
        "source": source,
    }


# --------------------------------------------------------------------------- official feeds


def gdacs(since: dt.date, until: dt.date, log=print) -> list[dict]:
    """Floods and wildfires (≥ MIN_FIRE_HA or orange/red) active in [since, until], with affected-area extents."""
    out = []
    for kind in ("FL", "WF"):
        for page in range(1, 6):
            params = {"eventlist": kind, "alertlevel": "Green;Orange;Red", "fromDate": since, "toDate": until}
            feats = get_json(f"{GDACS}/events/geteventlist/SEARCH", params | {"pagenumber": page}).get("features", [])
            for f in feats:
                p = f["properties"]
                severity = (p.get("severitydata") or {}).get("severity") or 0
                if kind == "WF" and p.get("alertlevel") == "Green" and severity < MIN_FIRE_HA:
                    continue
                lon, lat = f["geometry"]["coordinates"][:2]
                source = {
                    "kind": "gdacs",
                    "id": f"{kind}{p['eventid']}",
                    "url": p.get("url", {}).get("report", ""),
                    "title": f"{p.get('name') or p.get('description')} ({p.get('alertlevel')} alert)",
                    "date": p["fromdate"][:10],
                    "detail": (p.get("severitydata") or {}).get("severitytext", "").strip(),
                }
                out.append(
                    candidate(
                        HAZARDS[kind],
                        lon,
                        lat,
                        _date(p["fromdate"]),
                        _date(p["todate"]),
                        "",
                        p.get("country", ""),
                        source,
                        _gdacs_extent(p.get("url", {}).get("geometry")),
                    )
                )
            if len(feats) < 100:  # last page
                break
    log(f"  GDACS: {len(out)} floods/wildfires")
    return out


def _gdacs_extent(url: str | None) -> list[float] | None:
    if not url:
        return None
    try:
        feats = get_json(url).get("features", [])
    except (OSError, ValueError):
        return None
    affected = [
        f["geometry"]["coordinates"] for f in feats if str(f["properties"].get("Class", "")).startswith("Poly_A")
    ]
    return bbox_of(affected) if affected else None


def cems(since: dt.date, log=print) -> list[dict]:
    """Copernicus EMS rapid-mapping activations for floods and wildfires since `since`."""
    out = []
    for r in get_json(f"{CEMS}/public-activations-info/", {"limit": 40}).get("results", []):
        hazard = HAZARDS.get(r.get("category"))
        if not hazard or _date(r["eventTime"]) < since:
            continue
        detail = (get_json(f"{CEMS}/public-activations/", {"code": r["code"]}).get("results") or [{}])[0]
        extents = [a["extent"] for a in detail.get("aois", []) if a.get("extent")]
        lon, lat = bbox_of(r["centroid"])[:2]
        source = {
            "kind": "cems",
            "id": r["code"],
            "url": f"https://rapidmapping.emergency.copernicus.eu/{r['code']}",
            "title": r["name"],
            "date": r["eventTime"][:10],
            "detail": (detail.get("reason") or "")[:300],
        }
        place = ", ".join(a["name"] for a in detail.get("aois", [])[:2])
        bbox = bbox_of(" ".join(extents)) if extents else None
        when = _date(r["eventTime"])
        out.append(candidate(hazard, lon, lat, when, when, place, ", ".join(r.get("countries", [])), source, bbox))
    log(f"  Copernicus EMS: {len(out)} activations")
    return out


# --------------------------------------------------------------------------- news index

_last_gdelt = 0.0


def hazard_of(title: str) -> str | None:
    return next((h for h, rx in HAZARD_WORDS.items() if re.search(rx, title, re.I)), None)


def gdelt(hours: int = 24, max_records: int = 250) -> list[dict]:
    """Recent English news about any of the three hazards: {hazard, url, title, date, outlet}. Waits ≥ 5.5 s between
    requests and backs off when refused."""
    global _last_gdelt
    params = {
        "query": f"{NEWS_QUERY} sourcelang:english",
        "mode": "artlist",
        "format": "json",
        "maxrecords": max_records,
        "timespan": f"{hours}h",
        "sort": "hybridrel",
    }
    articles = None
    for attempt in range(4):
        time.sleep(max(0.0, _last_gdelt + 5.5 - time.time()))
        try:
            body = get(GDELT, params, tries=1)
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            body = b""
        _last_gdelt = time.time()
        try:
            articles = json.loads(body).get("articles", [])
            break
        except ValueError:  # 429, or "Please limit requests to one every 5 seconds…": back off and retry
            time.sleep(20 * (attempt + 1))
    if articles is None:
        raise OSError("GDELT kept refusing requests (rate limit)")
    seen, out = set(), []
    for a in articles:
        title = a.get("title", "")
        key = re.sub(r"\W+", " ", title.lower()).strip()
        hazard = hazard_of(title)
        if not key or key in seen or not hazard:
            continue
        seen.add(key)
        date = a.get("seendate", "")
        out.append(
            {
                "hazard": hazard,
                "url": a["url"],
                "title": title,
                "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}" if len(date) >= 8 else "",
                "outlet": a.get("domain", ""),
            }
        )
    return out


def rss(url: str, hazard: str, hours: int, words: str | None = None) -> list[dict]:
    """Items of an RSS 2.0 feed published in the last `hours` (and whose title or summary matches `words`)."""
    root = ET.fromstring(get(url)[:5_000_000])
    since = dt.datetime.now(dt.UTC) - dt.timedelta(hours=hours)
    out = []
    for item in root.iter("item"):
        title, link = (item.findtext("title") or "").strip(), (item.findtext("link") or "").strip()
        try:
            when = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=dt.UTC)
        text = f"{title} {item.findtext('description') or ''}"
        if not link.startswith("http") or when < since or (words and not re.search(words, text, re.I)):
            continue
        out.append({"hazard": hazard, "url": link, "title": title, "date": when.date().isoformat(),
                    "outlet": urllib.parse.urlsplit(link).netloc.removeprefix("www.")})  # fmt: skip
    return out


def eonet(since: dt.date, log=print) -> list[dict]:
    """NASA EONET open wildfires and floods since `since` (no prescribed burns; fires ≥ MIN_FIRE_ACRES)."""
    out = []
    params = {"status": "open", "category": "wildfires,floods", "days": (dt.date.today() - since).days + 1}
    for e in get_json(EONET, params).get("events", []):
        hazard = {"wildfires": "wildfire", "floods": "flood"}.get(e["categories"][0]["id"])
        last = e["geometry"][-1]
        if not hazard or last.get("type") != "Point" or e["title"].lower().startswith("prescribed"):
            continue
        if hazard == "wildfire" and (last.get("magnitudeValue") or 0) < MIN_FIRE_ACRES:
            continue
        lon, lat = last["coordinates"][:2]
        source = {
            "kind": "eonet",
            "id": e["id"],
            "url": (e.get("sources") or [{}])[0].get("url", f"https://eonet.gsfc.nasa.gov/api/v3/events/{e['id']}"),
            "title": e["title"],
            "date": e["geometry"][0]["date"][:10],
            "detail": f"{last.get('magnitudeValue') or ''} {last.get('magnitudeUnit') or ''}".strip(),
        }
        first, latest = _date(e["geometry"][0]["date"]), _date(last["date"])
        place = re.sub(r"^(Wildfire|Flood)s?\s+", "", e["title"])  # "Wildfire Rafter 4B, Schleicher, Texas"
        out.append(candidate(hazard, lon, lat, first, latest, place, "", source))
    log(f"  NASA EONET: {len(out)} events")
    return out
