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
- No credentials are stored in the repository. Hugging Face uploads use the token from `hf auth login`.
