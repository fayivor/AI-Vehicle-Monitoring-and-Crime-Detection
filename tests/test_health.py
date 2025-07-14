"""Test health check endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_basic_health_check(client: TestClient):
    """Test basic health check endpoint."""
    response = client.get("/health/")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "service" in data
    assert "version" in data


def test_detailed_health_check(client: TestClient):
    """Test detailed health check endpoint."""
    response = client.get("/health/detailed")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert "system" in data
    assert "components" in data
    assert "configuration" in data
    
    # Check system metrics
    assert "cpu_percent" in data["system"]
    assert "memory" in data["system"]
    assert "disk" in data["system"]
    
    # Check components
    assert "faiss_index" in data["components"]
    assert "chromadb" in data["components"]


def test_readiness_check(client: TestClient):
    """Test readiness probe endpoint."""
    response = client.get("/health/ready")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "components" in data
    assert data["status"] in ["ready", "not_ready"]


def test_liveness_check(client: TestClient):
    """Test liveness probe endpoint."""
    response = client.get("/health/live")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "alive"
    assert "timestamp" in data
