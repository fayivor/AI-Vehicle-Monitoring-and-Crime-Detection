"""Pytest configuration and fixtures."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock

from src.api.main import app
from src.models.user import User, UserRole, SecurityClearance


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Mock user fixture."""
    return User(
        user_id="test-user-001",
        username="testuser",
        email="test@ghana.gov.gh",
        full_name="Test User",
        role=UserRole.OFFICER,
        security_clearance=SecurityClearance.RESTRICTED,
        is_active=True,
        permissions=["view_vehicles", "view_incidents", "create_incidents"]
    )


@pytest.fixture
def mock_admin_user():
    """Mock admin user fixture."""
    return User(
        user_id="admin-001",
        username="admin",
        email="admin@ghana.gov.gh",
        full_name="System Administrator",
        role=UserRole.ADMINISTRATOR,
        security_clearance=SecurityClearance.SECRET,
        is_active=True,
        permissions=["*"]
    )


@pytest.fixture
def auth_headers(mock_user):
    """Authentication headers fixture."""
    # TODO: Generate actual JWT token
    return {"Authorization": "Bearer mock-token"}


@pytest.fixture
def admin_auth_headers(mock_admin_user):
    """Admin authentication headers fixture."""
    # TODO: Generate actual JWT token
    return {"Authorization": "Bearer mock-admin-token"}
