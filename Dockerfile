# LTS - Linux image for a VPS (see README "Running on the VPS" and docker-compose.yaml).
FROM python:3.12-slim

# the git commit this image was built from (set by the GitHub workflow); logged at start
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC \
    LTS_DB_PATH=/app/data/lts.sqlite \
    LTS_MODEL_DIR=/app/data/models \
    LTS_LOG_DIR=/app/data/logs

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 1000 lts && mkdir -p /app/data && chown -R lts /app
USER lts

# "start" = on an empty volume: backfill 30 days, rebuild candidates, train; then run the service
CMD ["python", "-m", "lts", "start"]
