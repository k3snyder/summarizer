"""Async SQLite database setup with SQLAlchemy"""

from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import String, Integer, DateTime, Text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Database URL from config (will be configurable via settings)
DATABASE_URL = "sqlite+aiosqlite:///./jobs.db"


# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging during development
    future=True,
)


# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


# Base class for all models
class Base(DeclarativeBase):
    pass


# Job table model
class Job(Base):
    """
    Job table for tracking document processing jobs.

    Stores job metadata, configuration, status, and results.
    """
    __tablename__ = "jobs"

    # Primary key
    id: Mapped[str] = mapped_column(String, primary_key=True)

    # Status fields
    status: Mapped[str] = mapped_column(String, nullable=False, index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    current_stage: Mapped[str] = mapped_column(String, nullable=True)
    message: Mapped[str] = mapped_column(String, nullable=True)

    # File information
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    file_path: Mapped[str] = mapped_column(String, nullable=False)

    # Configuration and output
    config_json: Mapped[str] = mapped_column(Text, nullable=False)
    output_path: Mapped[str] = mapped_column(String, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    # Error tracking
    error: Mapped[str] = mapped_column(Text, nullable=True)


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.

    Yields an async database session and ensures proper cleanup.

    Usage:
        @app.get("/example")
        async def example(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(Job))
            return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
