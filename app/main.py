import logging
import time
import os
from logging.handlers import RotatingFileHandler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ask import router as ask_router
from app.services.indexing import get_index
from app.core.config import settings

# Configure logging with enhanced format
log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

# Create formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler (always enabled)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(log_level)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(log_level)
root_logger.addHandler(console_handler)

# File handler (if log_file is configured)
if settings.log_file:
    try:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(settings.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            settings.log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
        # Log that file logging is enabled
        root_logger.info(f"File logging enabled: {settings.log_file}")
    except Exception as e:
        root_logger.warning(f"Failed to setup file logging: {e}")

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
    startup_start = time.time()
    logger.info("[STARTUP] Starting Aurora QA System...")
    logger.info(f"[STARTUP] Log level: {settings.log_level.upper()}")
    try:
        index = get_index()
        await index.build_index()
        startup_time = time.time() - startup_start
        logger.info(f"[STARTUP] Complete in {startup_time:.2f}s. Index ready.")
    except Exception as e:
        startup_time = time.time() - startup_start
        logger.error(f"[STARTUP] Error during startup after {startup_time:.2f}s: {e}", exc_info=True)
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    index = get_index()
    return {
        "status": "healthy",
        "index_ready": index.is_ready()
    }

