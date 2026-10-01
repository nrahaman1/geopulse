"""GeoPulse Scout: finds floods, wildfires and forest loss in official alerts and the news, and turns each into a
ready-to-run GeoPulse case (study area + before/after windows).

    geopulse scout discover        # official feeds + news (needs a local LLM, see reader.py), then plan cases
    geopulse scout list
    geopulse scout run <case-id>   # the GeoPulse analysis of one case

Nothing runs an analysis unless asked: discovery only proposes cases. Each pass writes, next to cases.json:
status.json (what every source did: the health check of a scheduled run) and extractions.jsonl (every article the
model read, with its verdict and no article text: the evaluation and training log for the reader model).
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import time

from .. import pipeline
from .cases import check_all, load_cases, merge, plan, save_cases, score, scout_dir
from .feeds import NEWS_FEEDS, cems, eonet, gdacs, gdelt, rss
from .reader import LLM_MODEL, extract, fetch_text, geocode

__all__ = ["discover", "load_cases", "run_case", "scout_dir"]


def _append(name: str, record: dict) -> None:
    path = scout_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_news(hours: int, max_articles: int, deadline: float, status: dict, log=print) -> list[dict]:
    """News candidates: GDELT articles -> text -> LLM event -> verified quote -> geocoded place. Hazards are read in
    turn (one flood story, one wildfire story, …) so a slow model still covers all three within the time budget."""
    articles, seen = [], set()
    channels = [("gdelt", lambda: gdelt(hours))]
    channels += [
        (f"rss:{url.split('/')[2]}", lambda u=url, h=hz, w=words: rss(u, h, hours, w)) for url, hz, words in NEWS_FEEDS
    ]
    for name, fetch in channels:
        try:
            found = [a for a in fetch() if a["url"] not in seen]
            status["sources"][name] = {"ok": True, "articles": len(found)}
        except (OSError, ValueError, SyntaxError) as e:  # SyntaxError: ElementTree's ParseError
            status["sources"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
            log(f"! news source {name}: {e}")
            continue
        seen.update(a["url"] for a in found)
        articles += found
    queues = {h: [a for a in articles if a["hazard"] == h] for h in ("flood", "wildfire", "vegetation")}
    turns = (a for group in itertools.zip_longest(*queues.values()) for a in group if a)
    read = dict.fromkeys(queues, 0)
    out = []
    for art in turns:
        if time.monotonic() > deadline:
            status["budget_reached"] = True
            log("! time budget reached: the remaining articles wait for the next run")
            break
        if read[art["hazard"]] >= max_articles:
            continue
        try:
            text = fetch_text(art["url"])
        except Exception as e:  # any site can fail in any way; one article never stops the run
            log(f"  skip {art['outlet']}: {type(e).__name__}")
            continue
        if len(text) < 400:
            continue
        read[art["hazard"]] += 1
        record = {"date": dt.date.today(), "query": art["hazard"], "url": art["url"], "outlet": art["outlet"],
                  "title": art["title"], "model": LLM_MODEL, "chars": len(text)}  # fmt: skip
        why: list = []
        try:
            event = extract(art, text, why)
        except Exception as e:
            log(f"! LLM ({LLM_MODEL}) on {art['outlet']}: {type(e).__name__}: {e}")
            _append("extractions.jsonl", record | {"verdict": "error", "reason": f"{type(e).__name__}"})
            continue
        loc = event and next(
            (g for p in event["locations"] if (g := geocode(p["place"], p["admin_area"], p["country"]))), None
        )
        if not event or not loc:
            reason = why[0] if why else f"could not place {event['locations'][0]['place']!r} precisely"
            _append("extractions.jsonl", record | {"verdict": "rejected", "reason": reason})
            continue
        top = event["locations"][0]
        accepted = {"verdict": "accepted", "hazard": event["hazard"], "place": loc["label"], "lon": loc["lon"],
                    "lat": loc["lat"], "quote": event["quote"]}  # fmt: skip
        _append("extractions.jsonl", record | accepted)
        out.append(
            {
                "hazard": event["hazard"],
                "lon": loc["lon"],
                "lat": loc["lat"],
                "bbox": loc.get("bbox"),  # the place's own extent: a town, or a park the planner caps at 30 km
                "start": event["start"],
                "end": event["end"],
                "place": ", ".join(dict.fromkeys(p for p in (top["place"], top["admin_area"]) if p)),
                "country": top["country"],
                "source": {
                    "kind": "news",
                    "url": art["url"],
                    "title": art["title"],
                    "outlet": art["outlet"],
                    "date": art["date"],
                    "quote": event["quote"],
                    "detail": event["impact"],
                },
            }
        )
    status["news"] = {"read": sum(read.values()), "per_hazard": read, "events": len(out), "model": LLM_MODEL}
    log(f"  news: read {read}, {len(out)} events placed")
    return out


def discover(
    days: int = 3,
    news: bool = True,
    max_articles: int = 15,
    check: bool = True,
    budget_minutes: float | None = None,
    log=print,
) -> list[dict]:
    """One Scout pass: collect candidates, merge them into the stored cases, plan and check the changed ones.
    `budget_minutes` bounds the news reading (the slow part), so a scheduled run always finishes and publishes."""
    t0 = time.monotonic()
    deadline = t0 + 60 * budget_minutes if budget_minutes else float("inf")
    today = dt.date.today()
    since = today - dt.timedelta(days=days)
    status = {"started": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "sources": {}}
    candidates = []
    official = (("gdacs", lambda: gdacs(since, today, log)), ("cems", lambda: cems(since, log)),
                ("eonet", lambda: eonet(since, log)))  # fmt: skip
    for name, fn in official:
        try:
            found = fn()
            candidates += found
            status["sources"][name] = {"ok": True, "events": len(found)}
        except (OSError, ValueError, KeyError) as e:
            status["sources"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200]}
            log(f"! {name} unavailable: {type(e).__name__}: {e}")
    if news:
        candidates += read_news(24 * days, max_articles, deadline, status, log)
    keep = (today - dt.timedelta(days=45)).isoformat()
    cases = [c for c in load_cases() if c.get("updated", c["first_seen"]) >= keep]
    before = {c["id"]: len(c["sources"]) for c in cases}
    cases = merge(cases, candidates, today)
    changed = [c for c in cases if before.get(c["id"]) != len(c["sources"]) or c.get("status") != "ready"]
    for case in changed:
        plan(score(case), today)
    if check:
        log(f"  checking imagery for {len(changed)} cases…")
        check_all([c for c in changed if c.get("status") != "too recent"], log)
    path = save_cases(cases)
    new = [c["id"] for c in cases if c["id"] not in before]
    _append("history.jsonl", {"date": today, "candidates": len(candidates), "new": new})
    status |= {
        "finished": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - t0),
        "cases": len(cases),
        "new": len(new),
        "ok": any(s["ok"] for k, s in status["sources"].items() if k in ("gdacs", "cems", "eonet")),
    }
    (scout_dir() / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    log(f"✓ {len(cases)} cases ({len(new)} new) -> {path}")
    return cases


def run_case(case_id: str, log=print, progress=lambda f, s: None) -> dict:
    """Run the GeoPulse analysis of one stored case; the summary is kept on the case."""
    cases = load_cases()
    case = next((c for c in cases if c["id"] == case_id), None)
    if case is None:
        raise KeyError(f"no case {case_id!r}; see `geopulse scout list`")
    if "before" not in case:
        raise ValueError(f"case {case_id} is {case.get('status')}: no windows yet")
    request = pipeline.make_request(case["aoi"], case["before"], case["after"], case["task"], case["sensors"])
    out = scout_dir() / "runs" / case_id
    summary = pipeline.run(request, out, log=log, progress=progress)
    case["run"] = {"output": str(out), "affected_km2": summary.get("affected_km2"), "date": dt.date.today().isoformat()}
    save_cases(cases)
    return summary
