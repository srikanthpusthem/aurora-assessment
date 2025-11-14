import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router
from app.services.indexing import get_index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Aurora QA System",
    description="Question-answering API for Aurora's take-home assignment",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ask_router, prefix="/api", tags=["QA"])


@app.on_event("startup")
async def startup_event():
    """Build FAISS index on application startup."""
    logger.info("Starting up Aurora QA System...")
    try:
        index = get_index()
        await index.build_index()
        logger.info("Startup complete. Index is ready.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    index = get_index()
    return {
        "status": "healthy",
        "index_ready": index.is_ready()
    }

