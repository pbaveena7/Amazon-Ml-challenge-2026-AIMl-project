from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import match, health, stats

app = FastAPI(
    title="EntityMatch AI Backend",
    description="Backend for Amazon ML Challenge 2026 Entity Resolution",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="", tags=["health"])
app.include_router(match.router, prefix="/api", tags=["matching"])
app.include_router(stats.router, prefix="/api", tags=["statistics"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
