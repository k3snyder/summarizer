"""Health check endpoint"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint

    Returns system status and version information
    """
    return {
        "status": "healthy",
        "version": "1.0.0"
    }
