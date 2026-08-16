"""ForgeMind API entrypoint (Phase 1 foundation).

Run locally:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.tasks import router as tasks_router
from app.config import get_settings
from app.database.session import check_database_connection
from app.logging import setup_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Application factory — also used by tests with an overridden DB."""
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail fast: refuse to start if the DB is unreachable. Never hang.
        check_database_connection()
        logger.info("Started %s (%s)", settings.app_name, settings.environment)
        yield

    app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)
    app.include_router(tasks_router)

    @app.get("/health", tags=["health"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


setup_logging()
app = create_app()
