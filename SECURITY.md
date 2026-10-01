# Security policy

## Reporting a vulnerability

Please report privately through GitHub:
<https://github.com/nrahaman1/geopulse/security/advisories/new>. Do not open a public issue. You should get a
response within a week; fixes are released as a patch version with credit unless you prefer otherwise.

## Supported versions

Only the latest release receives security fixes while GeoPulse is pre-1.0.

## Design notes for deployers

- `geopulse serve` binds to `127.0.0.1` by default. Binding to `0.0.0.0` (as the Docker image does) exposes an
  unauthenticated API: put it behind a reverse proxy with authentication and rate limiting on shared networks.
- For shared demos set `GEOPULSE_PUBLIC=1` (no listing of other visitors' jobs) and a small
  `GEOPULSE_MAX_JOB_KM2`. Jobs are validated (geometry, area, windows) and capped at 20 pending.
- The API never fetches user-supplied URLs; imagery comes only from the configured STAC provider.
- Place search runs in the visitor's browser against OpenStreetMap's Nominatim; the typed query and the browser's IP
  go to that service, not to GeoPulse. Typing coordinates (`lat, lon`) needs no lookup.
- Result files are served only from inside the job's directory (path traversal is rejected; see `tests/test_api.py`).
- Checkpoints are loaded with `torch.load(weights_only=True)`, and `geopulse models pull` verifies each file's
  SHA-256 against its model card before use.
- The web app (GitHub Pages) has no backend. The visitor's browser talks directly to the Planetary Computer (STAC
  search, anonymous SAS tokens, imagery), Esri (basemap tiles) and, for place search, Nominatim; the app, its
  libraries and the ONNX models come from GitHub Pages (models are verified by SHA-256 before use). The AOI and
  dates are sent to the Planetary Computer as a STAC search; results stay in the browser's IndexedDB and are never
  uploaded. Delete them from the jobs list or by clearing the site's data.
- Desktop app: the Python engine binds to 127.0.0.1 on a random port and rejects any request without the per-launch
  token (a random 192-bit value held by the app); CORS allows only the app's own origins. The engine environment is
  installed by the bundled uv from the bundled lockfile (hashes pinned); checkpoints come from this repository's
  `models-v1` release and are SHA-256 verified. The app's content security policy allows scripts only from the app.
- Desktop updates: the app installs an update only if its minisign signature matches the public key built into the
  app (`plugins.updater` in `tauri.conf.json`); the private key is a repository secret used only by the release
  workflow. Updates come from this repository's latest GitHub Release over HTTPS.
- Scout: news pages are untrusted input. The model reading them has no tools and must fill a strict JSON schema;
  code checks every quote against the article, geocodes places itself and plans the case. The scheduled run writes
  only to the `scout-data` branch, and the apps display case text as text, never as HTML.
- No credentials are stored in the repository. Publishing releases uses the `gh` CLI's own login.
