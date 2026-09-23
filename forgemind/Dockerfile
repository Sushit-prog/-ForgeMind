# ForgeMind worker image (also used for the API service if you add one).
FROM python:3.12-slim

WORKDIR /app

# git is required at runtime by the Research/Developer agents (worktree
# clone, ls-tree, rev-parse); python:3.12-slim does not ship it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install the package + pinned deps first so rebuilds reuse the layer.
COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY tests ./tests

# .[dev] so the TESTING stage can run the repo's pytest command inside the
# worktree (pytest/pytest-asyncio are not in the base install).
RUN pip install --no-cache-dir ".[dev]"

EXPOSE 8000

# Migrate on boot, then run the arq worker.
CMD ["sh", "-c", "alembic upgrade head && python -m app.worker.worker"]
