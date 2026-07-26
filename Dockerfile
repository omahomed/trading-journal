# Playwright base image includes Python + Chromium + all the system libs
# Chromium needs (libgbm, libnss3, fonts, etc.) that nixpacks doesn't
# install by default. Locked to a specific Playwright version so upgrades
# are explicit — bumping the tag pulls in a matching Chromium.
#
# ~2GB image; Railway handles it fine but cold-starts are slower than
# the prior nixpacks build. Acceptable for a screenshot endpoint that
# runs once a day per user.
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

# Run everything as root so file writes (build-info.json, migration
# tracking table state, etc.) work without permission gymnastics. The
# Playwright base image defaults to `pwuser` which can bite copied
# files that root-owned COPY left behind.
USER root

WORKDIR /app

# Install Python deps. Copy just requirements.txt first for layer
# caching so a code-only change doesn't invalidate the pip install
# layer. The `psycopg2-binary` wheel is manylinux-compatible so it
# installs cleanly on the Playwright image's Python 3.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright's `pip install playwright` doesn't include browser
# binaries; the base image already ships Chromium so we skip
# `playwright install`.

COPY . .

# Railway respects the Procfile's `web:` command as the entrypoint
# when both a Dockerfile CMD and Procfile exist. Keep the CMD here
# as a fallback for local `docker run`.
CMD ["sh", "-c", "python migrations/run.py && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
