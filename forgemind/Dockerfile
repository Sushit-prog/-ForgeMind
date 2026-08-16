# ForgeMind worker image (also used for the API service if you add one).
FROM python:3.12-slim

WORKDIR /app

# Install the package + pinned deps first so rebuilds reuse the layer.
COPY pyproject.toml README.md alembic.ini ./
COPY app ./app
COPY tests ./tests

RUN pip install --no-cache-dir .

EXPOSE 8000

# Migrate on boot, then run the arq worker.
CMD ["sh", "-c", "alembic upgrade head && python -m app.worker.worker"]
