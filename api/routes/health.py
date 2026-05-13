from fastapi import APIRouter
from schemas.schemas import HealthResponse
from services.llm_service import get_model_status

router = APIRouter(tags=["System", "Root"])

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Tutor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "auth": "/auth/login",
            "generate": "/generate",
            "retrieve": "/retrieve",
            "student": "/student/save-question",
            "teacher": "/teacher/questions",
            "health": "/health"
        }
    }

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "loaded" if get_model_status() else "not loaded",
        "database": "connected"
    }
