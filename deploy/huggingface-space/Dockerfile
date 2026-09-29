# Hugging Face Space: installs GeoPulse from GitHub and serves the public demo on port 7860.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# rasterio's manylinux wheels link against the system libexpat, which slim images no longer include.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install "geopulse-eo @ https://github.com/nrahaman1/geopulse/archive/refs/heads/main.zip"

RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user
ENV GEOPULSE_MODELS=/home/user/data/models \
    GEOPULSE_CACHE=/home/user/data/cache \
    GEOPULSE_OUTPUTS=/home/user/data/outputs \
    GEOPULSE_PUBLIC=1 \
    GEOPULSE_MAX_JOB_KM2=150 \
    GEOPULSE_MC_PASSES=4

CMD ["sh", "-c", "geopulse models pull || echo 'models not pulled: serving the physics baseline'; exec geopulse serve --host 0.0.0.0 --port 7860"]
