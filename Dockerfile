# syntax=docker/dockerfile:1.6
# ── Lumos: FastAPI + uv ──────────────────────────────────────────
# Slim Python base + only the system packages unstructured[pdf] needs.
# Keeps the image ~400MB vs the 7GB unstructured/unstructured base.

FROM python:3.12-slim-bookworm AS builder

# System deps for unstructured[pdf], python-magic, pymupdf.
RUN apt-get update && apt-get install -y --no-install-recommends \
      poppler-utils \
      tesseract-ocr \
      libmagic1 \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

# uv — fast Python package manager; brings its own resolver.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy manifest + lockfile first for layer caching: dependency changes
# retrigger this layer, but code changes below do not.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra book --no-install-project

# Now copy the source and install the project itself (editable).
COPY . .
RUN uv sync --frozen --no-dev --extra book

# ── Runtime image ────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# Same runtime system deps (no build-essential needed).
RUN apt-get update && apt-get install -y --no-install-recommends \
      poppler-utils \
      tesseract-ocr \
      libmagic1 \
      libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /app /app

EXPOSE 10000
CMD ["uvicorn", "lumos.server.app:app", "--host", "0.0.0.0", "--port", "10000"]
