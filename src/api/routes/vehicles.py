"""Vehicle management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid

from ...models.vehicle import Vehicle, VehicleCreate, VehicleUpdate, VehicleSearchResult
from ...models.user import User
from ...utils.logging import get_logger, audit_logger
from .auth import get_current_user

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", response_model=List[Vehicle])
async def list_vehicles(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    status: Optional[str] = Query(None, description="Filter by vehicle status"),
    current_user: User = Depends(get_current_user)
) -> List[Vehicle]:
    """
    List vehicles with pagination and filtering.
    
    Args:
        skip: Number of records to skip
        limit: Number of records to return
        status: Filter by vehicle status
        current_user: Current authenticated user
        
    Returns:
        List of vehicles
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "vehicle",
        "list",
        "read"
    )
    
    # TODO: Implement actual database query
    # For now, return mock data
    mock_vehicles = [
        Vehicle(
            vehicle_id="VH-001-2024",
            registration_number="GH-1234-24",
            owner_details={
                "name": "John Doe",
                "id_number": "GHA123456789",
                "phone": "+233123456789",
                "address": "123 Main St, Accra",
                "email": "john.doe@email.com"
            },
            incident_history=["INC-001"],
            risk_score=0.3,
            make="Toyota",
            model="Camry",
            year=2020,
            color="Blue",
            status="active"
        ),
        Vehicle(
            vehicle_id="VH-002-2024",
            registration_number="GH-5678-24",
            owner_details={
                "name": "Jane Smith",
                "id_number": "GHA987654321",
                "phone": "+233987654321",
                "address": "456 Oak Ave, Kumasi",
                "email": "jane.smith@email.com"
            },
            incident_history=[],
            risk_score=0.1,
            make="Honda",
            model="Civic",
            year=2019,
            color="Red",
            status="active"
        )
    ]
    
    # Apply status filter if provided
    if status:
        mock_vehicles = [v for v in mock_vehicles if v.status == status]
    
    # Apply pagination
    return mock_vehicles[skip:skip + limit]


@router.get("/{vehicle_id}", response_model=Vehicle)
async def get_vehicle(
    vehicle_id: str,
    current_user: User = Depends(get_current_user)
) -> Vehicle:
    """
    Get a specific vehicle by ID.
    
    Args:
        vehicle_id: Vehicle identifier
        current_user: Current authenticated user
        
    Returns:
        Vehicle information
        
    Raises:
        HTTPException: If vehicle not found
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "vehicle",
        vehicle_id,
        "read"
    )
    
    # TODO: Implement actual database lookup
    # For now, return mock data
    if vehicle_id == "VH-001-2024":
        return Vehicle(
            vehicle_id="VH-001-2024",
            registration_number="GH-1234-24",
            owner_details={
                "name": "John Doe",
                "id_number": "GHA123456789",
                "phone": "+233123456789",
                "address": "123 Main St, Accra",
                "email": "john.doe@email.com"
            },
            incident_history=["INC-001"],
            risk_score=0.3,
            make="Toyota",
            model="Camry",
            year=2020,
            color="Blue",
            status="active"
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Vehicle not found"
    )


@router.post("/", response_model=Vehicle)
async def create_vehicle(
    vehicle_create: VehicleCreate,
    current_user: User = Depends(get_current_user)
) -> Vehicle:
    """
    Create a new vehicle record.
    
    Args:
        vehicle_create: Vehicle creation data
        current_user: Current authenticated user
        
    Returns:
        Created vehicle
        
    Raises:
        HTTPException: If user lacks permission
    """
    # Check permissions
    if "create_vehicles" not in current_user.permissions and current_user.role != "administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to create vehicles"
        )
    
    # TODO: Implement actual vehicle creation
    # For now, return mock created vehicle
    vehicle_id = f"VH-{str(uuid.uuid4())[:8]}-2024"
    
    new_vehicle = Vehicle(
        vehicle_id=vehicle_id,
        registration_number=vehicle_create.registration_number,
        owner_details=vehicle_create.owner_details,
        make=vehicle_create.make,
        model=vehicle_create.model,
        year=vehicle_create.year,
        color=vehicle_create.color,
        last_seen_location=vehicle_create.initial_location,
        status="active"
    )
    
    audit_logger.log_data_access(
        current_user.user_id,
        "vehicle",
        vehicle_id,
        "create"
    )
    
    logger.info(
        "Vehicle created",
        vehicle_id=vehicle_id,
        registration=vehicle_create.registration_number,
        created_by=current_user.user_id
    )
    
    return new_vehicle


@router.put("/{vehicle_id}", response_model=Vehicle)
async def update_vehicle(
    vehicle_id: str,
    vehicle_update: VehicleUpdate,
    current_user: User = Depends(get_current_user)
) -> Vehicle:
    """
    Update a vehicle record.
    
    Args:
        vehicle_id: Vehicle identifier
        vehicle_update: Vehicle update data
        current_user: Current authenticated user
        
    Returns:
        Updated vehicle
        
    Raises:
        HTTPException: If vehicle not found or user lacks permission
    """
    # Check permissions
    if "update_vehicles" not in current_user.permissions and current_user.role != "administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to update vehicles"
        )
    
    # TODO: Implement actual vehicle update
    # For now, return mock updated vehicle
    if vehicle_id != "VH-001-2024":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found"
        )
    
    audit_logger.log_data_access(
        current_user.user_id,
        "vehicle",
        vehicle_id,
        "update"
    )
    
    logger.info(
        "Vehicle updated",
        vehicle_id=vehicle_id,
        updated_by=current_user.user_id
    )
    
    # Return mock updated vehicle
    return Vehicle(
        vehicle_id=vehicle_id,
        registration_number="GH-1234-24",
        owner_details=vehicle_update.owner_details or {
            "name": "John Doe",
            "id_number": "GHA123456789",
            "phone": "+233123456789",
            "address": "123 Main St, Accra",
            "email": "john.doe@email.com"
        },
        incident_history=["INC-001"],
        risk_score=vehicle_update.risk_score or 0.3,
        last_seen_location=vehicle_update.last_seen_location,
        alert_flags=vehicle_update.alert_flags or [],
        make=vehicle_update.make or "Toyota",
        model=vehicle_update.model or "Camry",
        year=vehicle_update.year or 2020,
        color=vehicle_update.color or "Blue",
        status=vehicle_update.status or "active"
    )


@router.get("/search/", response_model=List[VehicleSearchResult])
async def search_vehicles(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results"),
    current_user: User = Depends(get_current_user)
) -> List[VehicleSearchResult]:
    """
    Search vehicles using the RAG system.
    
    Args:
        q: Search query
        max_results: Maximum number of results
        current_user: Current authenticated user
        
    Returns:
        List of vehicle search results with similarity scores
    """
    audit_logger.log_query_execution(
        current_user.user_id,
        "vehicle_search",
        f"search-{uuid.uuid4()}",
        0  # TODO: Add actual processing time
    )
    
    # TODO: Implement actual vector search using FAISS
    # For now, return mock search results
    mock_results = [
        VehicleSearchResult(
            vehicle=Vehicle(
                vehicle_id="VH-001-2024",
                registration_number="GH-1234-24",
                owner_details={
                    "name": "John Doe",
                    "id_number": "GHA123456789",
                    "phone": "+233123456789",
                    "address": "123 Main St, Accra",
                    "email": "john.doe@email.com"
                },
                incident_history=["INC-001"],
                risk_score=0.3,
                make="Toyota",
                model="Camry",
                year=2020,
                color="Blue",
                status="active"
            ),
            similarity_score=0.85,
            rank=1,
            match_reasons=["registration_number_match", "location_proximity"]
        )
    ]
    
    logger.info(
        "Vehicle search performed",
        query=q,
        user_id=current_user.user_id,
        results_count=len(mock_results)
    )
    
    return mock_results[:max_results]
