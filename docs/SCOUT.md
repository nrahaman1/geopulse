# GeoPulse Scout

The Scout finds floods, wildfires and forest loss in official alerts and in the news, and turns each event into a
ready-to-run GeoPulse **case**: a study area, before/after windows and the sensors with imagery. A case has the same
shape as an example event, so the web app, the desktop app and `geopulse infer` take it as is. Nothing is analysed
unless someone asks: the Scout only proposes.

```bash
geopulse scout discover            # alerts + news, merge, plan, check imagery
geopulse scout discover --no-news  # official alerts only (no LLM needed)
geopulse scout list
geopulse scout run <case-id>       # the GeoPulse analysis of one case
```

## How a case is made

| Step | What happens | Code |
|---|---|---|
| Official alerts | GDACS (floods from GloFAS, wildfires from GWIS ≥ 10,000 ha or orange/red), Copernicus EMS rapid-mapping activations (expert-drawn areas) and NASA EONET (US wildfires ≥ 1,000 acres, no prescribed burns) | `scout/feeds.py` |
| News | Two independent channels: GDELT (one combined query per run, each article sorted by the words in its title) and topic RSS feeds (Wildfire Today, InciWeb, Mongabay's deforestation stories). Scrapling fetches each story (`<article>`, else `<main>`) as Markdown with hidden content stripped, only where robots.txt allows | `scout/feeds.py`, `scout/reader.py` |
| Reading | An open-weights LLM behind any OpenAI-compatible endpoint fills a strict JSON schema: evidence first, then places, dates and finally *has it happened?* | `scout/reader.py` |
| Verification | Every quote must appear verbatim in the article; every place in a quote or the title; forecasts, watches and warnings are rejected, also when the model calls them events (a quote about a watch or a risk is no evidence); dates outside the publication window fall back to it | `scout/reader.py` |
| Placing | Places are geocoded (OpenStreetMap Nominatim, ≤ 1 request/s, cached), never taken from the model; only towns, districts, parks and natural features no larger than 150 km, never buildings, states or countries, and only a place whose name is the one the story gives (no "State" for "the state") | `scout/reader.py` |
| Merging | Same hazard within 50 km and 10 days is one event; official geometry wins over news; confidence grows with independent sources | `scout/cases.py` |
| Planning | Deterministic windows per task (flood: dry weeks before / the days after the latest report; wildfire: before ignition / after the fire; forest loss: the same season one year apart), validated by `pipeline.make_request` | `scout/cases.py` |
| Imagery | Sentinel-1/2 scenes counted in both windows: *ready*, *waiting for imagery* or *partial imagery* | `scout/cases.py` |

The model reads untrusted text, so it gets no tools, its output is schema-constrained, and code, not the model,
decides places, windows and whether a case exists.

## Live operation

`.github/workflows/scout.yml` runs every 6 hours on a free GitHub runner (4 CPU cores, no GPU):

1. checks out the state branch `scout-data` (never `main`);
2. starts Ollama with Qwen3 8B (runtime and model cached between runs);
3. `geopulse scout discover --days 2 --max-articles 10 --budget-minutes 90`;
4. commits `cases.json`, `status.json`, `history.jsonl`, `extractions.jsonl` and the geocoder cache to `scout-data`;
5. rebuilds GitHub Pages, whose build includes the cases: the apps' **Recent events** button lists them with their
   sources, and **Add to examples** adds one to the user's example list (kept in that browser or app); the desktop
   app and `geopulse serve` fetch them from Pages;
6. re-enables its own schedule (GitHub disables schedules in repositories idle for 60 days).

What keeps it working: every source fails on its own (a dead feed or a rate-limited news query is recorded in
`status.json` and the run goes on); retries with backoff (GDELT's rate limit included); the news time budget, so a
run always finishes and publishes; a run never overlaps the previous one; GitHub e-mails the owner when a scheduled
run fails.

### Reader benchmark

Eight articles (five real, three constructed: a forecast, a policy story and one with an injected instruction), the
full reader including verification:

| Model | 4 CPU cores | RTX 4060 Laptop | Correct |
|---|---:|---:|---:|
| Qwen3 1.7B | 15.6 s / article | | 4/8 |
| Qwen3 4B | 34.8 s | | 7/8 |
| Qwen3 8B | 64.9 s | 5.5 s | 8/8 |

The live run uses 8B: about 30 articles in 35–50 minutes per run. Eight articles is a smoke test, not an evaluation;
the improvement loop below starts with a real one.

## Improving the reader (with Soup)

[Soup](https://github.com/MakazhanAlpamys/Soup) fine-tunes open models from one YAML file on a consumer GPU and exports
GGUF for Ollama, so a better reader drops in without code changes. The plan, in order:

1. **Evaluation set first.** `extractions.jsonl` lists every article read and the verdict. A few hundred are checked
   by hand (and through "wrong case?" reports) into a locked gold set. Article texts stay on the maintainer's machine:
   they are copyrighted and are never published.
2. **Gate every model change** on that set (Soup's eval-gated training idea): a new reader replaces the live one only
   if it is at least as accurate on the gold set.
3. **Distil the 8B reader into a 4B student** (sequence-level distillation: SFT on the teacher's verified outputs, QLoRA
   on the RTX 4060), export it as GGUF and publish it as a release asset. The goal: 8B accuracy at about twice the
   speed on the CPU runner, so each run reads twice the news.
4. **Repeat** as reports accumulate (Soup's data-flywheel idea), always through the gate.

## Limits

- GDELT rate-limits busy IP addresses and can refuse a run's query; the RSS channel and the official alerts carry on.
  Flood news then comes mostly from official alerts: no reliable open flood-news feed exists (FloodList stopped in 2024).
- News is English-only for now (GDELT `sourcelang:english`).
- A case's study area is a 15 × 15 km box around the reported place unless an official feed gives an extent; the
  place a story names is not always where the water or fire is.
- GDACS wildfire alerts favour large savanna fires; small but newsworthy fires come from the news.
- Scheduled runs are best-effort on GitHub's side and can start late.
