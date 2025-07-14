"""Health check endpoints."""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import time
import psutil
import os

from ...utils.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": settings.app_name,
        "version": settings.app_version,
    }


@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system metrics."""
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Check if data directories exist
    faiss_index_exists = os.path.exists(settings.faiss_index_path)
    chromadb_exists = os.path.exists(settings.chromadb_path)
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": settings.app_name,
        "version": settings.app_version,
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
        },
        "components": {
            "faiss_index": "available" if faiss_index_exists else "not_found",
            "chromadb": "available" if chromadb_exists else "not_found",
            "database": "unknown",  # TODO: Add database health check
            "redis": "unknown",     # TODO: Add Redis health check
        },
        "configuration": {
            "debug": settings.debug,
            "log_level": settings.log_level,
            "api_prefix": settings.api_prefix,
        }
    }


@router.get("/ready")
async def readiness_check() -> Dict[str, Any]:
    """Kubernetes readiness probe endpoint."""
    # Check if essential components are ready
    ready = True
    components = {}
    
    # Check data directories
    if not os.path.exists(settings.faiss_index_path):
        ready = False
        components["faiss_index"] = "not_ready"
    else:
        components["faiss_index"] = "ready"
    
    if not os.path.exists(settings.chromadb_path):
        ready = False
        components["chromadb"] = "not_ready"
    else:
        components["chromadb"] = "ready"
    
    # TODO: Add database connectivity check
    # TODO: Add Redis connectivity check
    
    status = "ready" if ready else "not_ready"
    
    return {
        "status": status,
        "timestamp": time.time(),
        "components": components,
    }


@router.get("/live")
async def liveness_check() -> Dict[str, Any]:
    """Kubernetes liveness probe endpoint."""
    return {
        "status": "alive",
        "timestamp": time.time(),
    }
