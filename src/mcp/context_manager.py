"""Model Context Protocol (MCP) implementation for dynamic context management."""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ..models.user import User, UserRole, SecurityClearance, UserSession
from ..models.query import QueryRequest, QueryType
from ..utils.config import get_settings
from ..utils.logging import get_logger
from ..utils.security import filter_sensitive_data, check_security_clearance

logger = get_logger(__name__)
settings = get_settings()


class ContextPriority(Enum):
    """Context priority levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ContextItem:
    """Individual context item with metadata."""
    content: str
    source: str
    priority: ContextPriority
    timestamp: datetime = field(default_factory=datetime.utcnow)
    relevance_score: float = 0.0
    security_level: SecurityClearance = SecurityClearance.PUBLIC
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextWindow:
    """Context window for a user session."""
    user_session: UserSession
    items: List[ContextItem] = field(default_factory=list)
    max_size: int = field(default=settings.mcp_context_window)
    last_updated: datetime = field(default_factory=datetime.utcnow)
    update_count: int = 0


class ContextShaper:
    """
    Context shaping logic implementing MCP requirements:
    - Officers: Focus on immediate actionable intel
    - Analysts: Provide detailed historical patterns  
    - Administrators: Include system metrics and summaries
    - High threat: Prioritize recent similar incidents
    - Location-based: Include regional crime patterns
    """
    
    def __init__(self):
        self.role_weights = self._initialize_role_weights()
        self.threat_multipliers = self._initialize_threat_multipliers()
    
    def _initialize_role_weights(self) -> Dict[UserRole, Dict[str, float]]:
        """Initialize role-specific weighting factors."""
        return {
            UserRole.OFFICER: {
                'recency': 0.4,
                'actionability': 0.5,
                'location_relevance': 0.3,
                'severity': 0.4,
                'historical_depth': 0.1
            },
            UserRole.ANALYST: {
                'recency': 0.2,
                'actionability': 0.2,
                'location_relevance': 0.2,
                'severity': 0.3,
                'historical_depth': 0.5
            },
            UserRole.ADMINISTRATOR: {
                'recency': 0.3,
                'actionability': 0.3,
                'location_relevance': 0.2,
                'severity': 0.3,
                'historical_depth': 0.3
            },
            UserRole.SUPERVISOR: {
                'recency': 0.3,
                'actionability': 0.4,
                'location_relevance': 0.3,
                'severity': 0.4,
                'historical_depth': 0.2
            }
        }
    
    def _initialize_threat_multipliers(self) -> Dict[int, float]:
        """Initialize threat level multipliers."""
        return {
            1: 1.0,    # Normal operations
            2: 1.2,    # Elevated awareness
            3: 1.5,    # Heightened alert
            4: 2.0,    # High threat
            5: 3.0     # Critical threat
        }
    
    def shape_context(
        self,
        user_session: UserSession,
        query_request: QueryRequest,
        raw_context: List[ContextItem]
    ) -> List[ContextItem]:
        """
        Shape context based on user role, location, intent, and threat level.
        
        Args:
            user_session: Current user session
            query_request: Query request with context
            raw_context: Raw context items
            
        Returns:
            Shaped and prioritized context items
        """
        logger.info(
            "Shaping context",
            user_role=user_session.user.role,
            threat_level=user_session.threat_level,
            context_items=len(raw_context)
        )
        
        # Filter by security clearance
        filtered_context = self._filter_by_security_clearance(
            raw_context, user_session.user
        )
        
        # Calculate relevance scores
        scored_context = self._calculate_relevance_scores(
            filtered_context, user_session, query_request
        )
        
        # Apply role-specific prioritization
        prioritized_context = self._apply_role_prioritization(
            scored_context, user_session.user.role
        )
        
        # Apply threat level adjustments
        threat_adjusted_context = self._apply_threat_level_adjustments(
            prioritized_context, user_session.threat_level
        )
        
        # Apply location-based filtering
        location_filtered_context = self._apply_location_filtering(
            threat_adjusted_context, user_session, query_request
        )
        
        # Sort by final relevance score and limit size
        final_context = sorted(
            location_filtered_context,
            key=lambda x: x.relevance_score,
            reverse=True
        )[:settings.mcp_context_window]
        
        logger.info(
            "Context shaping completed",
            original_items=len(raw_context),
            final_items=len(final_context),
            avg_relevance=sum(item.relevance_score for item in final_context) / len(final_context) if final_context else 0
        )
        
        return final_context
    
    def _filter_by_security_clearance(
        self,
        context_items: List[ContextItem],
        user: User
    ) -> List[ContextItem]:
        """Filter context items by user's security clearance."""
        filtered_items = []
        
        for item in context_items:
            if check_security_clearance(user, item.security_level):
                # Filter sensitive data from content
                filtered_content = filter_sensitive_data(
                    {'content': item.content}, user
                ).get('content', item.content)
                
                filtered_item = ContextItem(
                    content=filtered_content,
                    source=item.source,
                    priority=item.priority,
                    timestamp=item.timestamp,
                    relevance_score=item.relevance_score,
                    security_level=item.security_level,
                    metadata=item.metadata
                )
                filtered_items.append(filtered_item)
        
        return filtered_items
    
    def _calculate_relevance_scores(
        self,
        context_items: List[ContextItem],
        user_session: UserSession,
        query_request: QueryRequest
    ) -> List[ContextItem]:
        """Calculate relevance scores for context items."""
        scored_items = []
        
        for item in context_items:
            # Base score from item priority
            base_score = item.priority.value / 4.0
            
            # Recency factor
            hours_ago = (datetime.utcnow() - item.timestamp).total_seconds() / 3600
            recency_score = max(0.1, 1.0 - (hours_ago / 168))  # Decay over 1 week
            
            # Query relevance (simplified - could use semantic similarity)
            query_relevance = self._calculate_query_relevance(
                item.content, query_request.query_text
            )
            
            # Location relevance
            location_relevance = self._calculate_location_relevance(
                item, user_session, query_request
            )
            
            # Combine scores
            final_score = (
                base_score * 0.3 +
                recency_score * 0.3 +
                query_relevance * 0.3 +
                location_relevance * 0.1
            )
            
            item.relevance_score = final_score
            scored_items.append(item)
        
        return scored_items
    
    def _calculate_query_relevance(self, content: str, query: str) -> float:
        """Calculate relevance between content and query (simplified)."""
        # Simple keyword matching - in production, use semantic similarity
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(content_words))
        return min(overlap / len(query_words), 1.0)
    
    def _calculate_location_relevance(
        self,
        item: ContextItem,
        user_session: UserSession,
        query_request: QueryRequest
    ) -> float:
        """Calculate location-based relevance."""
        # TODO: Implement actual location-based relevance calculation
        # This would consider:
        # - User's current location
        # - Query location filters
        # - Item location metadata
        # - Regional crime patterns
        
        return 0.5  # Default neutral relevance
    
    def _apply_role_prioritization(
        self,
        context_items: List[ContextItem],
        user_role: UserRole
    ) -> List[ContextItem]:
        """Apply role-specific prioritization."""
        weights = self.role_weights.get(user_role, self.role_weights[UserRole.OFFICER])
        
        for item in context_items:
            role_multiplier = 1.0
            
            # Officers prefer actionable, recent information
            if user_role == UserRole.OFFICER:
                if 'actionable' in item.metadata:
                    role_multiplier *= 1.5
                if item.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]:
                    role_multiplier *= 1.3
            
            # Analysts prefer detailed historical data
            elif user_role == UserRole.ANALYST:
                if 'historical' in item.metadata:
                    role_multiplier *= 1.4
                if 'pattern' in item.metadata:
                    role_multiplier *= 1.3
            
            # Administrators prefer system metrics and summaries
            elif user_role == UserRole.ADMINISTRATOR:
                if 'system' in item.metadata or 'summary' in item.metadata:
                    role_multiplier *= 1.4
            
            # Supervisors prefer oversight-level information
            elif user_role == UserRole.SUPERVISOR:
                if item.priority == ContextPriority.CRITICAL:
                    role_multiplier *= 1.6
                if 'oversight' in item.metadata:
                    role_multiplier *= 1.3
            
            item.relevance_score *= role_multiplier
        
        return context_items
    
    def _apply_threat_level_adjustments(
        self,
        context_items: List[ContextItem],
        threat_level: int
    ) -> List[ContextItem]:
        """Apply threat level adjustments to context prioritization."""
        threat_multiplier = self.threat_multipliers.get(threat_level, 1.0)
        
        for item in context_items:
            # High threat levels prioritize recent similar incidents
            if threat_level >= 4:
                if item.priority in [ContextPriority.HIGH, ContextPriority.CRITICAL]:
                    item.relevance_score *= threat_multiplier
                
                # Boost recent high-severity items
                hours_ago = (datetime.utcnow() - item.timestamp).total_seconds() / 3600
                if hours_ago <= 24 and 'severity' in item.metadata:
                    severity = item.metadata.get('severity', 1)
                    if severity >= 4:
                        item.relevance_score *= 1.5
            
            # Medium threat levels boost relevant patterns
            elif threat_level == 3:
                if 'pattern' in item.metadata or 'trend' in item.metadata:
                    item.relevance_score *= threat_multiplier
        
        return context_items
    
    def _apply_location_filtering(
        self,
        context_items: List[ContextItem],
        user_session: UserSession,
        query_request: QueryRequest
    ) -> List[ContextItem]:
        """Apply location-based filtering and regional crime patterns."""
        # TODO: Implement location-based filtering
        # This would:
        # - Filter by geographic proximity
        # - Include regional crime patterns
        # - Consider jurisdiction boundaries
        # - Apply location-specific threat assessments
        
        return context_items


class ContextManager:
    """
    Context manager implementing real-time context updates and information flow control.
    """
    
    def __init__(self):
        self.context_shaper = ContextShaper()
        self.active_sessions: Dict[str, ContextWindow] = {}
        self.context_cache: Dict[str, List[ContextItem]] = {}
        self.cache_ttl = settings.mcp_cache_ttl
    
    def create_session_context(self, user_session: UserSession) -> ContextWindow:
        """Create a new context window for a user session."""
        context_window = ContextWindow(
            user_session=user_session,
            max_size=settings.mcp_context_window
        )
        
        self.active_sessions[user_session.session_id] = context_window
        
        logger.info(
            "Created session context",
            session_id=user_session.session_id,
            user_id=user_session.user.user_id,
            user_role=user_session.user.role
        )
        
        return context_window
    
    def update_session_context(
        self,
        session_id: str,
        new_context_items: List[ContextItem],
        query_request: Optional[QueryRequest] = None
    ) -> ContextWindow:
        """Update context for an active session."""
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        context_window = self.active_sessions[session_id]
        
        # Check update limits
        if context_window.update_count >= settings.mcp_max_context_updates:
            logger.warning(
                "Context update limit reached",
                session_id=session_id,
                update_count=context_window.update_count
            )
            return context_window
        
        # Shape context based on user profile and query
        if query_request:
            shaped_context = self.context_shaper.shape_context(
                context_window.user_session,
                query_request,
                new_context_items
            )
        else:
            shaped_context = new_context_items
        
        # Add to context window
        context_window.items.extend(shaped_context)
        
        # Maintain window size
        if len(context_window.items) > context_window.max_size:
            # Keep highest relevance items
            context_window.items = sorted(
                context_window.items,
                key=lambda x: x.relevance_score,
                reverse=True
            )[:context_window.max_size]
        
        # Update metadata
        context_window.last_updated = datetime.utcnow()
        context_window.update_count += 1
        
        logger.info(
            "Updated session context",
            session_id=session_id,
            new_items=len(shaped_context),
            total_items=len(context_window.items),
            update_count=context_window.update_count
        )
        
        return context_window
    
    def get_session_context(self, session_id: str) -> Optional[ContextWindow]:
        """Get context window for a session."""
        return self.active_sessions.get(session_id)
    
    def cleanup_expired_sessions(self) -> None:
        """Clean up expired sessions and context windows."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, context_window in self.active_sessions.items():
            # Check if session is expired (no activity for 1 hour)
            if (current_time - context_window.last_updated).total_seconds() > 3600:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
            logger.info("Cleaned up expired session", session_id=session_id)
    
    def get_context_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of context for a session."""
        context_window = self.active_sessions.get(session_id)
        if not context_window:
            return {}
        
        priority_counts = {}
        for priority in ContextPriority:
            priority_counts[priority.name] = sum(
                1 for item in context_window.items if item.priority == priority
            )
        
        return {
            'session_id': session_id,
            'user_role': context_window.user_session.user.role,
            'total_items': len(context_window.items),
            'priority_distribution': priority_counts,
            'last_updated': context_window.last_updated.isoformat(),
            'update_count': context_window.update_count,
            'avg_relevance': sum(item.relevance_score for item in context_window.items) / len(context_window.items) if context_window.items else 0
        }
