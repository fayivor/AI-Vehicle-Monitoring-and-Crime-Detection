"""Model Context Protocol (MCP) integration layer."""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from .context_manager import ContextManager, ContextItem, ContextPriority
from .threat_assessment import ThreatAssessor, ThreatLevel
from ..models.user import UserSession, User
from ..models.query import QueryRequest, QueryResponse
from ..models.vehicle import Vehicle
from ..models.incident import Incident
from ..utils.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class MCPProcessor:
    """
    Model Context Protocol processor that orchestrates context management,
    threat assessment, and real-time updates for the RAG pipeline.
    """
    
    def __init__(self):
        self.context_manager = ContextManager()
        self.threat_assessor = ThreatAssessor()
        self.active_sessions: Dict[str, UserSession] = {}
    
    async def initialize_user_session(
        self,
        user: User,
        location: Optional[str] = None,
        intent: Optional[str] = None
    ) -> UserSession:
        """
        Initialize a new user session with MCP context.
        
        Args:
            user: User object
            location: User's current location
            intent: User's current intent/task
            
        Returns:
            Initialized user session
        """
        # Create user session
        session_id = f"session-{user.user_id}-{int(datetime.utcnow().timestamp())}"
        
        user_session = UserSession(
            user=user,
            session_id=session_id,
            location=location,
            intent=intent,
            threat_level=1,  # Default to normal
            active_queries=[],
            session_start=datetime.utcnow(),
            last_activity=datetime.utcnow()
        )
        
        # Store session
        self.active_sessions[session_id] = user_session
        
        # Create context window
        self.context_manager.create_session_context(user_session)
        
        # Perform initial threat assessment
        await self._update_threat_level(user_session)
        
        logger.info(
            "Initialized MCP session",
            session_id=session_id,
            user_id=user.user_id,
            user_role=user.role,
            location=location,
            threat_level=user_session.threat_level
        )
        
        return user_session
    
    async def process_query_with_mcp(
        self,
        query_request: QueryRequest,
        user_session: UserSession,
        retrieval_results: List[Any] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the MCP layer for context shaping.
        
        Args:
            query_request: Query request
            user_session: Current user session
            retrieval_results: Raw retrieval results
            
        Returns:
            MCP-processed context and metadata
        """
        logger.info(
            "Processing query with MCP",
            session_id=user_session.session_id,
            query_type=query_request.query_type,
            user_role=user_session.user.role
        )
        
        # Update session activity
        user_session.last_activity = datetime.utcnow()
        user_session.active_queries.append(query_request.query_text)
        
        # Keep only recent queries
        if len(user_session.active_queries) > 5:
            user_session.active_queries = user_session.active_queries[-5:]
        
        # Update threat level based on query context
        await self._update_threat_level(user_session, query_request)
        
        # Convert retrieval results to context items
        context_items = self._convert_to_context_items(retrieval_results or [])
        
        # Update session context with new items
        context_window = self.context_manager.update_session_context(
            user_session.session_id,
            context_items,
            query_request
        )
        
        # Generate MCP metadata
        mcp_metadata = {
            'session_id': user_session.session_id,
            'user_role': user_session.user.role,
            'threat_level': user_session.threat_level,
            'context_items_count': len(context_window.items),
            'context_shaped': True,
            'security_filtered': True,
            'location_context': user_session.location,
            'intent_context': user_session.intent,
            'session_duration': (datetime.utcnow() - user_session.session_start).total_seconds(),
            'context_summary': self.context_manager.get_context_summary(user_session.session_id)
        }
        
        # Generate role-specific context summary
        role_context = self._generate_role_specific_context(
            user_session, context_window.items, query_request
        )
        
        logger.info(
            "MCP processing completed",
            session_id=user_session.session_id,
            threat_level=user_session.threat_level,
            context_items=len(context_window.items),
            role_context_length=len(role_context)
        )
        
        return {
            'mcp_metadata': mcp_metadata,
            'shaped_context': context_window.items,
            'role_context': role_context,
            'threat_assessment': await self._get_current_threat_assessment(user_session)
        }
    
    async def _update_threat_level(
        self,
        user_session: UserSession,
        query_request: Optional[QueryRequest] = None
    ) -> None:
        """Update threat level for the user session."""
        try:
            # Check for cached assessment first
            cached_assessment = self.threat_assessor.get_cached_assessment(
                user_session.session_id,
                user_session.location
            )
            
            if cached_assessment:
                user_session.threat_level = cached_assessment.overall_threat_level.value
                return
            
            # TODO: Fetch recent vehicles and incidents from database
            # For now, use empty lists - in production this would query the database
            recent_vehicles: List[Vehicle] = []
            recent_incidents: List[Incident] = []
            
            # Perform threat assessment
            assessment = self.threat_assessor.assess_threat_level(
                user_session,
                recent_vehicles,
                recent_incidents,
                user_session.location
            )
            
            # Update session threat level
            old_threat_level = user_session.threat_level
            user_session.threat_level = assessment.overall_threat_level.value
            
            if old_threat_level != user_session.threat_level:
                logger.info(
                    "Threat level updated",
                    session_id=user_session.session_id,
                    old_level=old_threat_level,
                    new_level=user_session.threat_level,
                    threat_score=assessment.threat_score
                )
        
        except Exception as e:
            logger.error(
                "Failed to update threat level",
                session_id=user_session.session_id,
                error=str(e)
            )
            # Keep current threat level on error
    
    def _convert_to_context_items(self, retrieval_results: List[Any]) -> List[ContextItem]:
        """Convert retrieval results to context items."""
        context_items = []
        
        for result in retrieval_results:
            # Determine priority based on result score/relevance
            if hasattr(result, 'score') and result.score > 0.8:
                priority = ContextPriority.HIGH
            elif hasattr(result, 'score') and result.score > 0.6:
                priority = ContextPriority.MEDIUM
            else:
                priority = ContextPriority.LOW
            
            # Extract content
            if hasattr(result, 'content'):
                content = result.content
            elif hasattr(result, 'vehicle'):
                content = f"Vehicle: {result.vehicle.registration_number} (Risk: {result.vehicle.risk_score})"
            elif hasattr(result, 'incident'):
                content = f"Incident: {result.incident.incident_type} (Severity: {result.incident.severity_level})"
            else:
                content = str(result)
            
            # Extract source
            source = getattr(result, 'source', 'unknown')
            
            # Create context item
            context_item = ContextItem(
                content=content,
                source=source,
                priority=priority,
                timestamp=datetime.utcnow(),
                relevance_score=getattr(result, 'score', 0.5),
                metadata={
                    'result_type': type(result).__name__,
                    'original_result': result
                }
            )
            
            context_items.append(context_item)
        
        return context_items
    
    def _generate_role_specific_context(
        self,
        user_session: UserSession,
        context_items: List[ContextItem],
        query_request: QueryRequest
    ) -> str:
        """Generate role-specific context summary."""
        user_role = user_session.user.role
        
        if not context_items:
            return f"No specific context available for {user_role} role."
        
        # Sort by relevance
        sorted_items = sorted(context_items, key=lambda x: x.relevance_score, reverse=True)
        
        if user_role == "officer":
            # Officers need immediate actionable intelligence
            actionable_items = [
                item for item in sorted_items[:5]
                if item.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]
            ]
            
            if actionable_items:
                context_summary = "IMMEDIATE ACTION REQUIRED:\n"
                for i, item in enumerate(actionable_items, 1):
                    context_summary += f"{i}. {item.content[:100]}...\n"
            else:
                context_summary = "Current situation: Normal operations. Monitor for developments."
        
        elif user_role == "analyst":
            # Analysts need detailed historical patterns
            context_summary = "ANALYTICAL CONTEXT:\n"
            context_summary += f"Total data points: {len(context_items)}\n"
            
            # Group by source
            sources = {}
            for item in sorted_items:
                source = item.source
                if source not in sources:
                    sources[source] = []
                sources[source].append(item)
            
            context_summary += f"Data sources: {', '.join(sources.keys())}\n"
            context_summary += "Key patterns:\n"
            
            for i, item in enumerate(sorted_items[:3], 1):
                context_summary += f"{i}. {item.content[:150]}...\n"
        
        elif user_role == "administrator":
            # Administrators need system metrics and summaries
            high_priority_count = sum(
                1 for item in context_items 
                if item.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]
            )
            
            context_summary = "SYSTEM OVERVIEW:\n"
            context_summary += f"Total context items: {len(context_items)}\n"
            context_summary += f"High priority items: {high_priority_count}\n"
            context_summary += f"Current threat level: {user_session.threat_level}\n"
            context_summary += f"Session duration: {(datetime.utcnow() - user_session.session_start).total_seconds():.0f}s\n"
            
            if high_priority_count > 0:
                context_summary += "\nHigh priority items:\n"
                high_priority_items = [
                    item for item in sorted_items 
                    if item.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]
                ]
                for i, item in enumerate(high_priority_items[:3], 1):
                    context_summary += f"{i}. {item.content[:100]}...\n"
        
        elif user_role == "supervisor":
            # Supervisors need oversight-level information
            critical_count = sum(1 for item in context_items if item.priority == ContextPriority.CRITICAL)
            
            context_summary = "SUPERVISORY BRIEFING:\n"
            context_summary += f"Threat level: {user_session.threat_level}\n"
            context_summary += f"Critical items: {critical_count}\n"
            
            if critical_count > 0:
                context_summary += "CRITICAL ATTENTION REQUIRED:\n"
                critical_items = [
                    item for item in sorted_items 
                    if item.priority == ContextPriority.CRITICAL
                ]
                for i, item in enumerate(critical_items, 1):
                    context_summary += f"{i}. {item.content[:120]}...\n"
            else:
                context_summary += "No critical items requiring immediate attention.\n"
            
            # Add summary of top items
            context_summary += "\nKey developments:\n"
            for i, item in enumerate(sorted_items[:3], 1):
                context_summary += f"{i}. {item.content[:100]}...\n"
        
        else:
            # Default context
            context_summary = "CONTEXT SUMMARY:\n"
            for i, item in enumerate(sorted_items[:5], 1):
                context_summary += f"{i}. {item.content[:100]}...\n"
        
        return context_summary
    
    async def _get_current_threat_assessment(self, user_session: UserSession) -> Optional[Dict[str, Any]]:
        """Get current threat assessment for the session."""
        assessment = self.threat_assessor.get_cached_assessment(
            user_session.session_id,
            user_session.location
        )
        
        if assessment:
            return {
                'threat_level': assessment.overall_threat_level.name,
                'threat_score': assessment.threat_score,
                'confidence': assessment.confidence,
                'indicators_count': len(assessment.indicators),
                'recommendations': assessment.recommendations,
                'assessment_time': assessment.assessment_time.isoformat()
            }
        
        return None
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get user session by ID."""
        return self.active_sessions.get(session_id)
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions and contexts."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session in self.active_sessions.items():
            # Sessions expire after 4 hours of inactivity
            if (current_time - session.last_activity).total_seconds() > 14400:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info("Cleaned up expired MCP session", session_id=session_id)
        
        # Clean up context manager and threat assessor
        self.context_manager.cleanup_expired_sessions()
        self.threat_assessor.cleanup_old_assessments()
    
    def get_mcp_stats(self) -> Dict[str, Any]:
        """Get MCP system statistics."""
        active_session_count = len(self.active_sessions)
        
        # Calculate session statistics
        session_durations = []
        threat_levels = []
        
        for session in self.active_sessions.values():
            duration = (datetime.utcnow() - session.session_start).total_seconds()
            session_durations.append(duration)
            threat_levels.append(session.threat_level)
        
        avg_session_duration = sum(session_durations) / len(session_durations) if session_durations else 0
        avg_threat_level = sum(threat_levels) / len(threat_levels) if threat_levels else 1
        
        return {
            'active_sessions': active_session_count,
            'average_session_duration': avg_session_duration,
            'average_threat_level': avg_threat_level,
            'context_cache_size': len(self.context_manager.context_cache),
            'threat_assessments_cached': len(self.threat_assessor.recent_assessments),
            'last_updated': datetime.utcnow().isoformat()
        }
