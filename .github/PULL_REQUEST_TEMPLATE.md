## What and why

<!-- One or two sentences. Link the issue if there is one. -->

## Checks

- [ ] `uv run pytest` passes and new logic has a test
- [ ] `uv run ruff check geopulse tests` and `uv run ruff format --check geopulse tests` pass
- [ ] Geospatial changes keep grids aligned (`tests/test_grid.py`)
- [ ] New data or label sources are listed in `DATA_LICENSES.md` with their license
- [ ] Changed metrics come from a committed config (`geopulse evaluate ...`), and `MODEL_CARD.md` says so
- [ ] `CHANGELOG.md` updated for user-visible changes
