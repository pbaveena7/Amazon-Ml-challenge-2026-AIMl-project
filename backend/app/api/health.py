from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "pipeline_loaded": True, # Assume true for now, this would check the ML service
        "environment": "development"
    }
