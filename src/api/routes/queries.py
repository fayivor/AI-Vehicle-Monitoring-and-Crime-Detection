"""Query and RAG system endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional
import uuid
import time
from datetime import datetime

from ...models.query import QueryRequest, QueryResponse, QueryHistory, QueryAnalytics, RealTimeAlert
from ...models.user import User
from ...core.rag_pipeline import RAGPipeline
from ...utils.logging import get_logger, audit_logger
from .auth import get_current_user

router = APIRouter()
logger = get_logger(__name__)

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()


@router.post("/", response_model=QueryResponse)
async def execute_query(
    query_request: QueryRequest,
    current_user: User = Depends(get_current_user)
) -> QueryResponse:
    """
    Execute a query using the RAG system.
    
    Args:
        query_request: Query request with context and parameters
        current_user: Current authenticated user
        
    Returns:
        Query response with results and metadata
    """
    start_time = time.time()
    query_id = f"query-{uuid.uuid4()}"
    
    # Log query execution
    audit_logger.log_query_execution(
        current_user.user_id,
        query_request.query_type,
        query_id,
        0  # Will be updated with actual processing time
    )
    
    logger.info(
        "Query execution started",
        query_id=query_id,
        user_id=current_user.user_id,
        query_type=query_request.query_type,
        query_text=query_request.query_text[:100]  # Truncate for logging
    )
    
    # Set user role in query request
    query_request.user_role = current_user.role

    # Process query through RAG pipeline with MCP integration
    response = await rag_pipeline.process_query(query_request, current_user)
    
    # Update audit log with processing time
    audit_logger.log_query_execution(
        current_user.user_id,
        query_request.query_type,
        response.query_id,
        response.processing_time_ms
    )

    logger.info(
        "Query execution completed",
        query_id=response.query_id,
        user_id=current_user.user_id,
        processing_time_ms=response.processing_time_ms,
        confidence_score=response.confidence_score
    )

    return response


@router.get("/history", response_model=List[QueryHistory])
async def get_query_history(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of records to return"),
    current_user: User = Depends(get_current_user)
) -> List[QueryHistory]:
    """
    Get query history for the current user.
    
    Args:
        skip: Number of records to skip
        limit: Number of records to return
        current_user: Current authenticated user
        
    Returns:
        List of query history records
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "query_history",
        "list",
        "read"
    )
    
    # TODO: Implement actual database query
    # For now, return mock data
    mock_history = [
        QueryHistory(
            query_id="query-001",
            user_id=current_user.user_id,
            query_request=QueryRequest(
                query_text="Find high-risk vehicles near Tema Port",
                query_type="vehicle_search",
                user_role=current_user.role,
                max_results=10
            ),
            query_response=QueryResponse(
                query_id="query-001",
                response_text="Found 3 high-risk vehicles in the Tema Port area...",
                confidence_score=0.9,
                processing_time_ms=1500,
                user_role=current_user.role
            ),
            feedback_score=4,
            feedback_comments="Very helpful results"
        )
    ]
    
    return mock_history[skip:skip + limit]


@router.post("/{query_id}/feedback")
async def submit_query_feedback(
    query_id: str,
    feedback_score: int = Query(..., ge=1, le=5, description="Feedback score (1-5)"),
    feedback_comments: Optional[str] = Query(None, description="Optional feedback comments"),
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Submit feedback for a query.
    
    Args:
        query_id: Query identifier
        feedback_score: Feedback score (1-5)
        feedback_comments: Optional feedback comments
        current_user: Current authenticated user
        
    Returns:
        Feedback submission confirmation
    """
    # TODO: Implement actual feedback storage
    
    audit_logger.log_data_access(
        current_user.user_id,
        "query_feedback",
        query_id,
        "create"
    )
    
    logger.info(
        "Query feedback submitted",
        query_id=query_id,
        user_id=current_user.user_id,
        feedback_score=feedback_score
    )
    
    return {
        "message": "Feedback submitted successfully",
        "query_id": query_id,
        "feedback_score": feedback_score
    }


@router.get("/analytics", response_model=QueryAnalytics)
async def get_query_analytics(
    current_user: User = Depends(get_current_user)
) -> QueryAnalytics:
    """
    Get query analytics and metrics (admin only).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Query analytics data
        
    Raises:
        HTTPException: If user is not authorized
    """
    # Check if user is administrator or analyst
    if current_user.role not in ["administrator", "analyst"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to view analytics"
        )
    
    audit_logger.log_data_access(
        current_user.user_id,
        "query_analytics",
        "summary",
        "read"
    )
    
    # TODO: Implement actual analytics calculation
    # For now, return mock analytics
    return QueryAnalytics(
        total_queries=1250,
        queries_by_type={
            "vehicle_search": 450,
            "incident_search": 380,
            "risk_assessment": 220,
            "pattern_analysis": 120,
            "general_inquiry": 80
        },
        queries_by_role={
            "officer": 600,
            "analyst": 350,
            "administrator": 200,
            "supervisor": 100
        },
        average_response_time=1850.5,
        average_confidence_score=0.82,
        user_satisfaction=4.2,
        top_queries=[
            "Find vehicles with high risk scores",
            "Recent incidents near border crossings",
            "Suspicious activity patterns",
            "Vehicle registration verification",
            "Traffic violation trends"
        ],
        performance_trends={
            "daily_queries": [45, 52, 38, 61, 49, 55, 43],
            "response_time_trend": [1800, 1750, 1900, 1650, 1850, 1700, 1850],
            "satisfaction_trend": [4.1, 4.3, 4.0, 4.4, 4.2, 4.5, 4.2]
        }
    )


@router.get("/alerts", response_model=List[RealTimeAlert])
async def get_real_time_alerts(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of records to return"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    current_user: User = Depends(get_current_user)
) -> List[RealTimeAlert]:
    """
    Get real-time alerts.
    
    Args:
        skip: Number of records to skip
        limit: Number of records to return
        acknowledged: Filter by acknowledgment status
        current_user: Current authenticated user
        
    Returns:
        List of real-time alerts
    """
    audit_logger.log_data_access(
        current_user.user_id,
        "real_time_alerts",
        "list",
        "read"
    )
    
    # TODO: Implement actual alert retrieval
    # For now, return mock alerts
    mock_alerts = [
        RealTimeAlert(
            alert_id="alert-001",
            configuration_id="config-001",
            triggered_by="high_risk_vehicle_detected",
            alert_level=4,
            message="High-risk vehicle detected near sensitive location",
            data={
                "vehicle_id": "VH-001-2024",
                "location": "Tema Port",
                "risk_score": 0.9
            },
            acknowledged=False,
            triggered_at=datetime.utcnow()
        ),
        RealTimeAlert(
            alert_id="alert-002",
            configuration_id="config-002",
            triggered_by="suspicious_pattern_detected",
            alert_level=3,
            message="Suspicious movement pattern detected",
            data={
                "pattern_type": "unusual_route",
                "vehicles_involved": ["VH-002-2024", "VH-003-2024"]
            },
            acknowledged=True,
            acknowledged_by=current_user.user_id,
            triggered_at=datetime.utcnow(),
            acknowledged_at=datetime.utcnow()
        )
    ]
    
    # Apply acknowledgment filter
    if acknowledged is not None:
        mock_alerts = [a for a in mock_alerts if a.acknowledged == acknowledged]
    
    return mock_alerts[skip:skip + limit]


@router.post("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_user: User = Depends(get_current_user)
) -> dict:
    """
    Acknowledge a real-time alert.
    
    Args:
        alert_id: Alert identifier
        current_user: Current authenticated user
        
    Returns:
        Acknowledgment confirmation
    """
    # TODO: Implement actual alert acknowledgment
    
    audit_logger.log_data_access(
        current_user.user_id,
        "real_time_alerts",
        alert_id,
        "acknowledge"
    )
    
    logger.info(
        "Alert acknowledged",
        alert_id=alert_id,
        user_id=current_user.user_id
    )
    
    return {
        "message": "Alert acknowledged successfully",
        "alert_id": alert_id,
        "acknowledged_by": current_user.user_id,
        "acknowledged_at": datetime.utcnow().isoformat()
    }
