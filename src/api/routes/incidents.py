"""Incident management endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid
from datetime import datetime

from ...models.incident import Incident, IncidentCreate, IncidentUpdate, IncidentSearchResult, IncidentSummary
from ...models.user import User
from ...utils.logging import get_logger, audit_logger
from .auth import get_current_user

router = APIRouter()
logger = get_logger(__name__)


@router.get("/", response_model=List[Incident])
async def list_incidents(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of records to return"),
    status: Optional[str] = Query(None, description="Filter by incident status"),
    incident_type: Optional[str] = Query(None, description="Filter by incident type"),
    severity_level: Optional[int] = Query(None, ge=1, le=5, description="Filter by severity level"),
    current_user: User = Depends(get_current_user)
) -> List[Incident]:
    """
    List incidents with pagination and filtering.
    
    Args:
        skip: Number of records to skip
        limit: Number of records to return
        status: Filter by incident status
        incident_type: Filter by incident type
        severity_level: Filter by severity level
        current_user: Current authenticated user
        
    Returns:
        List of incidents
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "incident",
        "list",
        "read"
    )
    
    # TODO: Implement actual database query
    # For now, return mock data
    mock_incidents = [
        Incident(
            incident_id="INC-001-2024",
            vehicle_involved="VH-001-2024",
            incident_type="smuggling",
            location={
                "latitude": 5.6037,
                "longitude": -0.1870,
                "address": "Tema Port, Ghana"
            },
            timestamp=datetime.utcnow(),
            severity_level=4,
            description="Suspicious vehicle detected at border crossing with undeclared goods",
            status="investigating",
            reported_by="Officer Smith",
            evidence_files=["evidence_001.jpg", "evidence_002.pdf"],
            witnesses=["+233123456789"]
        ),
        Incident(
            incident_id="INC-002-2024",
            vehicle_involved="VH-002-2024",
            incident_type="speeding",
            location={
                "latitude": 5.5560,
                "longitude": -0.1969,
                "address": "Accra-Tema Highway"
            },
            timestamp=datetime.utcnow(),
            severity_level=2,
            description="Vehicle exceeding speed limit by 30 km/h",
            status="resolved",
            reported_by="Traffic Camera System"
        )
    ]
    
    # Apply filters
    if status:
        mock_incidents = [i for i in mock_incidents if i.status == status]
    if incident_type:
        mock_incidents = [i for i in mock_incidents if i.incident_type == incident_type]
    if severity_level:
        mock_incidents = [i for i in mock_incidents if i.severity_level == severity_level]
    
    # Apply pagination
    return mock_incidents[skip:skip + limit]


@router.get("/{incident_id}", response_model=Incident)
async def get_incident(
    incident_id: str,
    current_user: User = Depends(get_current_user)
) -> Incident:
    """
    Get a specific incident by ID.
    
    Args:
        incident_id: Incident identifier
        current_user: Current authenticated user
        
    Returns:
        Incident information
        
    Raises:
        HTTPException: If incident not found
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "incident",
        incident_id,
        "read"
    )
    
    # TODO: Implement actual database lookup
    # For now, return mock data
    if incident_id == "INC-001-2024":
        return Incident(
            incident_id="INC-001-2024",
            vehicle_involved="VH-001-2024",
            incident_type="smuggling",
            location={
                "latitude": 5.6037,
                "longitude": -0.1870,
                "address": "Tema Port, Ghana"
            },
            timestamp=datetime.utcnow(),
            severity_level=4,
            description="Suspicious vehicle detected at border crossing with undeclared goods",
            status="investigating",
            reported_by="Officer Smith",
            evidence_files=["evidence_001.jpg", "evidence_002.pdf"],
            witnesses=["+233123456789"]
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Incident not found"
    )


@router.post("/", response_model=Incident)
async def create_incident(
    incident_create: IncidentCreate,
    current_user: User = Depends(get_current_user)
) -> Incident:
    """
    Create a new incident record.
    
    Args:
        incident_create: Incident creation data
        current_user: Current authenticated user
        
    Returns:
        Created incident
        
    Raises:
        HTTPException: If user lacks permission
    """
    # Check permissions
    if "create_incidents" not in current_user.permissions and current_user.role != "administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to create incidents"
        )
    
    # TODO: Implement actual incident creation
    # For now, return mock created incident
    incident_id = f"INC-{str(uuid.uuid4())[:8]}-2024"
    
    new_incident = Incident(
        incident_id=incident_id,
        vehicle_involved=incident_create.vehicle_involved,
        incident_type=incident_create.incident_type,
        location=incident_create.location,
        severity_level=incident_create.severity_level,
        description=incident_create.description,
        status="open",
        reported_by=incident_create.reported_by or current_user.full_name,
        evidence_files=incident_create.evidence_files,
        witnesses=incident_create.witnesses
    )
    
    audit_logger.log_data_access(
        current_user.user_id,
        "incident",
        incident_id,
        "create"
    )
    
    logger.info(
        "Incident created",
        incident_id=incident_id,
        vehicle_involved=incident_create.vehicle_involved,
        incident_type=incident_create.incident_type,
        severity_level=incident_create.severity_level,
        created_by=current_user.user_id
    )
    
    return new_incident


@router.put("/{incident_id}", response_model=Incident)
async def update_incident(
    incident_id: str,
    incident_update: IncidentUpdate,
    current_user: User = Depends(get_current_user)
) -> Incident:
    """
    Update an incident record.
    
    Args:
        incident_id: Incident identifier
        incident_update: Incident update data
        current_user: Current authenticated user
        
    Returns:
        Updated incident
        
    Raises:
        HTTPException: If incident not found or user lacks permission
    """
    # Check permissions
    if "update_incidents" not in current_user.permissions and current_user.role != "administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to update incidents"
        )
    
    # TODO: Implement actual incident update
    # For now, return mock updated incident
    if incident_id != "INC-001-2024":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incident not found"
        )
    
    audit_logger.log_data_access(
        current_user.user_id,
        "incident",
        incident_id,
        "update"
    )
    
    logger.info(
        "Incident updated",
        incident_id=incident_id,
        updated_by=current_user.user_id
    )
    
    # Return mock updated incident
    return Incident(
        incident_id=incident_id,
        vehicle_involved="VH-001-2024",
        incident_type=incident_update.incident_type or "smuggling",
        location=incident_update.location or {
            "latitude": 5.6037,
            "longitude": -0.1870,
            "address": "Tema Port, Ghana"
        },
        timestamp=datetime.utcnow(),
        severity_level=incident_update.severity_level or 4,
        description=incident_update.description or "Updated description",
        status=incident_update.status or "investigating",
        reported_by="Officer Smith",
        evidence_files=incident_update.evidence_files or ["evidence_001.jpg"],
        witnesses=incident_update.witnesses or ["+233123456789"],
        investigation_notes=incident_update.investigation_notes or [],
        resolution_details=incident_update.resolution_details
    )


@router.get("/search/", response_model=List[IncidentSearchResult])
async def search_incidents(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(10, ge=1, le=100, description="Maximum results"),
    current_user: User = Depends(get_current_user)
) -> List[IncidentSearchResult]:
    """
    Search incidents using the RAG system.
    
    Args:
        q: Search query
        max_results: Maximum number of results
        current_user: Current authenticated user
        
    Returns:
        List of incident search results with relevance scores
    """
    audit_logger.log_query_execution(
        current_user.user_id,
        "incident_search",
        f"search-{uuid.uuid4()}",
        0  # TODO: Add actual processing time
    )
    
    # TODO: Implement actual vector search using FAISS
    # For now, return mock search results
    mock_results = [
        IncidentSearchResult(
            incident=Incident(
                incident_id="INC-001-2024",
                vehicle_involved="VH-001-2024",
                incident_type="smuggling",
                location={
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "address": "Tema Port, Ghana"
                },
                timestamp=datetime.utcnow(),
                severity_level=4,
                description="Suspicious vehicle detected at border crossing with undeclared goods",
                status="investigating",
                reported_by="Officer Smith"
            ),
            relevance_score=0.92,
            rank=1,
            match_reasons=["description_match", "location_match", "incident_type_match"]
        )
    ]
    
    logger.info(
        "Incident search performed",
        query=q,
        user_id=current_user.user_id,
        results_count=len(mock_results)
    )
    
    return mock_results[:max_results]


@router.get("/summary/dashboard", response_model=IncidentSummary)
async def get_incident_summary(
    current_user: User = Depends(get_current_user)
) -> IncidentSummary:
    """
    Get incident summary for dashboard.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Incident summary statistics
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "incident",
        "summary",
        "read"
    )
    
    # TODO: Implement actual summary calculation
    # For now, return mock summary
    return IncidentSummary(
        total_incidents=150,
        open_incidents=25,
        high_severity_incidents=8,
        incidents_by_type={
            "smuggling": 45,
            "speeding": 60,
            "suspicious_activity": 20,
            "traffic_violation": 15,
            "other": 10
        },
        recent_incidents=[
            Incident(
                incident_id="INC-001-2024",
                vehicle_involved="VH-001-2024",
                incident_type="smuggling",
                location={
                    "latitude": 5.6037,
                    "longitude": -0.1870,
                    "address": "Tema Port, Ghana"
                },
                timestamp=datetime.utcnow(),
                severity_level=4,
                description="Suspicious vehicle detected at border crossing",
                status="investigating",
                reported_by="Officer Smith"
            )
        ],
        trend_data={
            "weekly_trend": [10, 12, 8, 15, 11, 9, 13],
            "monthly_growth": 5.2
        }
    )
