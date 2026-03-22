# --- Build stage ---
# Install Python deps with gcc (needed to compile C extensions like trafilatura),
# then copy only the installed packages to the runtime stage — discards gcc/headers.
FROM python:3.12-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --prefix=/install puts packages in a separate tree so we can COPY them cleanly
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Runtime stage ---
# Clean image: no gcc, no pip cache, no build artifacts (~530 MB vs ~1.2 GB single-stage)
FROM python:3.12-slim

WORKDIR /app

# Bring in pre-built Python packages from the builder
COPY --from=builder /install /usr/local

# COPY . . brings in app code + VERSION file (read by config.py at runtime)
COPY . .

RUN mkdir -p output logs

EXPOSE 8000

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/info')" || exit 1

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
