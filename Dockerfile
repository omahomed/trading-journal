# Playwright base image includes Python + Chromium + all the system libs
# Chromium needs (libgbm, libnss3, fonts, etc.) that nixpacks doesn't
# install by default. Locked to a specific Playwright version so upgrades
# are explicit — bumping the tag pulls in a matching Chromium.
#
# ~2GB image; Railway handles it fine but cold-starts are slower than
# the prior nixpacks build. Acceptable for a screenshot endpoint that
# runs once a day per user.
FROM mcr.microsoft.com/playwright/python:v1.49.0-jammy

WORKDIR /app

# Install Python deps. Copy just requirements.txt first for layer caching
# so a code-only change doesn't invalidate the pip install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Playwright's `pip install playwright` doesn't include browser binaries;
# the base image already ships Chromium so we skip `playwright install`.

COPY . .

# Match Procfile's release + web commands. Railway auto-detects the
# Dockerfile and uses this CMD instead of the Procfile's web command.
CMD python migrations/run.py && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
