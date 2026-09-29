# Contributing to GeoPulse

Thanks for helping. GeoPulse aims to be small, reproducible and honest about what it measures; contributions that
keep it that way are very welcome: new benchmark events, label sources, sensors, tasks, bug reports and docs.

## Set up

```bash
git clone https://github.com/nrahaman1/geopulse && cd geopulse
uv sync                       # installs the package, dev tools and PyTorch (CUDA on Windows/Linux)
uv run pre-commit install     # ruff + basic hygiene on every commit
uv run geopulse doctor
uv run pytest                 # offline: no STAC or Hub access needed (browser-engine parity tests need Node.js)
```

The web platform is `geopulse/web/`. `geopulse serve` serves it with the API; for the static build (no server, the
browser computes everything) run `python scripts/build_web.py site && python -m http.server -d site 8080`. A change to
the Python pipeline that alters results needs the matching change in `geopulse/web/engine.js` or `worker.js`;
`tests/test_js_parity.py` compares the two.

CPU-only machine: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu` after `uv sync`.
Pull trained weights with `uv run geopulse models pull`; rebuild or download benchmarks with
`geopulse dataset build configs/data/<task>_bench.yaml` or `geopulse dataset pull`.

## Ground rules

- **Tests with logic.** Anything with a branch, threshold or geometry gets a small test. Geospatial changes must keep
  `tests/test_grid.py` passing (no half-pixel shifts). Physics thresholds get a synthetic scene in `tests/conftest.py`
  showing the event is mapped and a look-alike is not.
- **Open data only**, referenced by STAC item ID, AOI and time window. Never commit imagery, tiles or checkpoints
  (pre-commit rejects files over 1 MB). Add every new data or label source to `DATA_LICENSES.md` with its license.
- **Split by event, never by tile.** New benchmark events go in a manifest under `configs/data/`; `bench.audit` must pass.
- **Report what you measured.** Metrics in docs must come from `geopulse evaluate` on a committed config. Say which
  labels they are against, and flag anything tuned on a test split (see the burn fuel gate in `MODEL_CARD.md`).
- **Keep it lean.** Prefer the standard library and existing dependencies; explain any new dependency in the PR.
- Style: `ruff check` and `ruff format` (line length 120), type hints on public functions.

## Adding a task

1. Add it to `geopulse/tasks.py` (target name, compositing, scene selection, seasonal rule).
2. Add a training-free baseline branch in `geopulse/baseline.py::predict` with a synthetic-scene test.
3. Write a benchmark manifest with an independent label source (`geopulse/bench.py::task_labels`).
4. Train (`configs/experiments/`), evaluate against the baseline on the test split, and update `MODEL_CARD.md`.

## Releasing (maintainers)

1. Update `geopulse/__init__.py::__version__`, `CITATION.cff` and `CHANGELOG.md`.
2. `git tag vX.Y.Z && git push --tags` — `.github/workflows/release.yml` builds, publishes to PyPI (trusted
   publishing) and creates a GitHub release.
3. New weights: `geopulse models push`; new benchmark tiles: `geopulse dataset push` (needs `hf auth login`).
4. In-browser platform: `geopulse models export-onnx --model all`, then `python scripts/build_web.py site && hf upload nafizrahaman/geopulse site --repo-type space` (GitHub Pages deploys automatically).

By contributing you agree that your contributions are licensed under Apache-2.0, and you confirm that you have the
right to submit them (no employer-, sponsor- or third-party-restricted code or data).

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
