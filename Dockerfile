# GeoPulse API + web map, CPU build. rasterio wheels bundle GDAL and PROJ, so no system GIS stack is needed.
#   docker build -t geopulse . && docker run -p 8000:8000 geopulse      (or: docker compose up)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU PyTorch first, so installing GeoPulse does not pull the multi-GB CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY geopulse ./geopulse
COPY examples ./examples
RUN pip install . && rm -rf /app/build

# Hugging Face Spaces (and good practice) run as uid 1000.
RUN useradd --create-home --uid 1000 geopulse
USER geopulse
ENV GEOPULSE_MODELS=/home/geopulse/data/models \
    GEOPULSE_CACHE=/home/geopulse/data/cache \
    GEOPULSE_OUTPUTS=/home/geopulse/data/outputs \
    PORT=8000
WORKDIR /home/geopulse
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

# Fetch the published checkpoints if reachable; without them GeoPulse falls back to the physics baseline.
CMD ["sh", "-c", "geopulse models pull || echo 'models not pulled: serving the physics baseline'; exec geopulse serve --host 0.0.0.0 --port ${PORT}"]
