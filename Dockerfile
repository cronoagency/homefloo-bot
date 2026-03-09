FROM python:3.13-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# procps for healthcheck (pgrep)
RUN apt-get update && apt-get install -y --no-install-recommends procps && rm -rf /var/lib/apt/lists/*

# App
COPY homefloo-telegram-bot.py .

# Data volume for sessions and conversation logs
RUN mkdir -p /app/data

# Non-root user
RUN useradd -r -s /bin/false botuser && chown -R botuser:botuser /app
USER botuser

# No port exposed — bot does polling only

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD pgrep -f homefloo-telegram-bot || exit 1

CMD ["python", "homefloo-telegram-bot.py"]
