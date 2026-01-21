"""FastAPI application entry point"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logging import setup_logging, shutdown_logging
from app.routers import health, jobs
from app.db.database import init_db
from app.services.cleanup_task import start_cleanup_task, stop_cleanup_task

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    # Startup: Initialize logging first
    queue_listener = setup_logging()
    queue_listener.start()

    logger.info(
        "Application started - log_level=%s, log_dir=%s",
        settings.log_level,
        settings.log_dir,
    )

    # Initialize database connection
    await init_db()

    start_cleanup_task()

    yield

    # Shutdown
    logger.info("Application shutting down")
    stop_cleanup_task()
    shutdown_logging()


app = FastAPI(
    title="Document Summarizer Pipeline API",
    description="REST API for the privacy-first document intelligence system",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(jobs.router, prefix="/api", tags=["jobs"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Document Summarizer Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }
