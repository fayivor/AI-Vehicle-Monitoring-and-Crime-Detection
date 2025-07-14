"""Hybrid retrieval system combining vector similarity and rank-based filtering."""

import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .vector_store import VectorStore
from ..models.query import QueryRequest, SearchMode, RetrievalResult
from ..models.vehicle import Vehicle, VehicleSearchResult
from ..models.incident import Incident, IncidentSearchResult
from ..models.user import UserRole
from ..utils.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RankingFactors:
    """Factors used for rank-based filtering."""
    recency_weight: float = 0.3
    risk_score_weight: float = 0.4
    severity_weight: float = 0.3
    location_proximity_weight: float = 0.2


class HybridRetriever:
    """
    Hybrid retrieval system implementing the RAG pipeline:
    Input → Vector Similarity (FAISS) → Rank-Based Filtering → Context Assembly
    
    Combines semantic search with exact matching and applies rank-based filtering
    to prioritize recent incidents and high-risk vehicles as per guidelines.
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store or VectorStore()
        self.ranking_factors = RankingFactors()
        
    def retrieve(
        self,
        query_request: QueryRequest
    ) -> Tuple[List[VehicleSearchResult], List[IncidentSearchResult], List[RetrievalResult]]:
        """
        Execute hybrid retrieval combining vector similarity and rank-based filtering.
        
        Args:
            query_request: Query request with parameters
            
        Returns:
            Tuple of (vehicle_results, incident_results, raw_retrieval_results)
        """
        start_time = time.time()
        
        logger.info(
            "Starting hybrid retrieval",
            query_text=query_request.query_text[:100],
            search_mode=query_request.search_mode,
            user_role=query_request.user_role
        )
        
        # Step 1: Vector Similarity Search
        vector_results = self._vector_similarity_search(query_request)
        
        # Step 2: Exact Match Search (if hybrid mode)
        exact_results = []
        if query_request.search_mode in [SearchMode.EXACT, SearchMode.HYBRID]:
            exact_results = self._exact_match_search(query_request)
        
        # Step 3: Combine and deduplicate results
        combined_results = self._combine_results(vector_results, exact_results)
        
        # Step 4: Rank-based filtering
        ranked_results = self._apply_rank_based_filtering(combined_results, query_request)
        
        # Step 5: Convert to typed results
        vehicle_results, incident_results, retrieval_results = self._convert_to_typed_results(
            ranked_results, query_request
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(
            "Hybrid retrieval completed",
            processing_time_ms=processing_time,
            vehicle_results=len(vehicle_results),
            incident_results=len(incident_results),
            total_results=len(retrieval_results)
        )
        
        return vehicle_results, incident_results, retrieval_results
    
    def _vector_similarity_search(self, query_request: QueryRequest) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform vector similarity search using FAISS.
        
        Args:
            query_request: Query request
            
        Returns:
            List of (doc_id, similarity_score, metadata) tuples
        """
        try:
            # Determine search parameters
            search_k = min(query_request.max_results * 3, 100)  # Get more for filtering
            
            # Search both vehicle and incident types
            all_results = []
            
            # Search vehicles if requested
            if query_request.query_type in ["vehicle_search", "general_inquiry", "risk_assessment"]:
                vehicle_results = self.vector_store.search(
                    query_request.query_text,
                    k=search_k,
                    filter_type="vehicle"
                )
                all_results.extend(vehicle_results)
            
            # Search incidents if requested
            if query_request.query_type in ["incident_search", "general_inquiry", "pattern_analysis"]:
                incident_results = self.vector_store.search(
                    query_request.query_text,
                    k=search_k,
                    filter_type="incident"
                )
                all_results.extend(incident_results)
            
            # Filter by similarity threshold
            filtered_results = [
                (doc_id, score, metadata)
                for doc_id, score, metadata in all_results
                if score >= query_request.similarity_threshold
            ]
            
            return filtered_results
            
        except Exception as e:
            logger.error("Vector similarity search failed", error=str(e))
            return []
    
    def _exact_match_search(self, query_request: QueryRequest) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Perform exact match search for specific identifiers.
        
        Args:
            query_request: Query request
            
        Returns:
            List of (doc_id, similarity_score, metadata) tuples
        """
        # TODO: Implement exact match search against database
        # This would search for exact matches on:
        # - Vehicle registration numbers
        # - Vehicle IDs
        # - Incident IDs
        # - Owner names
        # - Phone numbers
        
        # For now, return empty list
        # In production, this would query the database directly
        return []
    
    def _combine_results(
        self,
        vector_results: List[Tuple[str, float, Dict[str, Any]]],
        exact_results: List[Tuple[str, float, Dict[str, Any]]]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Combine and deduplicate vector and exact match results.
        
        Args:
            vector_results: Results from vector search
            exact_results: Results from exact match search
            
        Returns:
            Combined and deduplicated results
        """
        # Use dict to deduplicate by doc_id, keeping highest score
        combined = {}
        
        for doc_id, score, metadata in vector_results:
            combined[doc_id] = (doc_id, score, metadata)
        
        for doc_id, score, metadata in exact_results:
            if doc_id in combined:
                # Keep higher score
                existing_score = combined[doc_id][1]
                if score > existing_score:
                    combined[doc_id] = (doc_id, score, metadata)
            else:
                combined[doc_id] = (doc_id, score, metadata)
        
        return list(combined.values())
    
    def _apply_rank_based_filtering(
        self,
        results: List[Tuple[str, float, Dict[str, Any]]],
        query_request: QueryRequest
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Apply rank-based filtering to prioritize recent incidents and high-risk vehicles.
        
        Args:
            results: Combined search results
            query_request: Query request with context
            
        Returns:
            Ranked and filtered results
        """
        scored_results = []
        
        for doc_id, similarity_score, metadata in results:
            # Calculate ranking score based on multiple factors
            ranking_score = self._calculate_ranking_score(
                metadata, similarity_score, query_request
            )
            
            scored_results.append((doc_id, similarity_score, metadata, ranking_score))
        
        # Sort by ranking score (descending)
        scored_results.sort(key=lambda x: x[3], reverse=True)
        
        # Apply time range filter if specified
        if query_request.time_range_hours:
            cutoff_time = datetime.utcnow() - timedelta(hours=query_request.time_range_hours)
            scored_results = self._filter_by_time_range(scored_results, cutoff_time)
        
        # Apply location filters if specified
        if query_request.location_filters:
            scored_results = self._filter_by_location(scored_results, query_request.location_filters)
        
        # Apply vehicle/incident specific filters
        if query_request.vehicle_filters:
            scored_results = self._filter_vehicles(scored_results, query_request.vehicle_filters)
        
        if query_request.incident_filters:
            scored_results = self._filter_incidents(scored_results, query_request.incident_filters)
        
        # Return top results without ranking score
        return [(doc_id, sim_score, metadata) for doc_id, sim_score, metadata, _ in scored_results]
    
    def _calculate_ranking_score(
        self,
        metadata: Dict[str, Any],
        similarity_score: float,
        query_request: QueryRequest
    ) -> float:
        """
        Calculate ranking score based on multiple factors.
        
        Args:
            metadata: Document metadata
            similarity_score: Vector similarity score
            query_request: Query request with context
            
        Returns:
            Ranking score
        """
        base_score = similarity_score
        
        data = metadata.get('data', {})
        doc_type = metadata.get('type')
        
        # Recency factor
        recency_score = self._calculate_recency_score(data, doc_type)
        
        # Risk/severity factor
        risk_score = self._calculate_risk_severity_score(data, doc_type)
        
        # Location proximity factor (if user location provided)
        location_score = self._calculate_location_score(data, query_request)
        
        # User role relevance factor
        role_score = self._calculate_role_relevance_score(data, doc_type, query_request.user_role)
        
        # Threat level factor
        threat_score = self._calculate_threat_level_score(data, query_request.threat_level)
        
        # Combine scores with weights
        final_score = (
            base_score * 0.4 +
            recency_score * self.ranking_factors.recency_weight +
            risk_score * self.ranking_factors.risk_score_weight +
            location_score * self.ranking_factors.location_proximity_weight +
            role_score * 0.1 +
            threat_score * 0.1
        )
        
        return final_score
    
    def _calculate_recency_score(self, data: Dict[str, Any], doc_type: str) -> float:
        """Calculate recency score (0-1, higher for more recent)."""
        try:
            if doc_type == 'vehicle':
                timestamp_str = data.get('updated_at') or data.get('created_at')
            elif doc_type == 'incident':
                timestamp_str = data.get('timestamp')
            else:
                return 0.5
            
            if not timestamp_str:
                return 0.5
            
            # Parse timestamp
            if isinstance(timestamp_str, str):
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                timestamp = timestamp_str
            
            # Calculate hours since timestamp
            hours_ago = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds() / 3600
            
            # Score decreases with age (24 hours = 0.5, 168 hours = 0.1)
            if hours_ago <= 1:
                return 1.0
            elif hours_ago <= 24:
                return 0.8
            elif hours_ago <= 168:  # 1 week
                return 0.5
            else:
                return 0.1
                
        except Exception:
            return 0.5
    
    def _calculate_risk_severity_score(self, data: Dict[str, Any], doc_type: str) -> float:
        """Calculate risk/severity score (0-1, higher for higher risk/severity)."""
        if doc_type == 'vehicle':
            risk_score = data.get('risk_score', 0.0)
            return min(risk_score, 1.0)
        elif doc_type == 'incident':
            severity_level = data.get('severity_level', 1)
            return min(severity_level / 5.0, 1.0)
        return 0.5
    
    def _calculate_location_score(self, data: Dict[str, Any], query_request: QueryRequest) -> float:
        """Calculate location proximity score."""
        # TODO: Implement location proximity calculation
        # This would calculate distance between document location and user/query location
        return 0.5
    
    def _calculate_role_relevance_score(self, data: Dict[str, Any], doc_type: str, user_role: UserRole) -> float:
        """Calculate relevance score based on user role."""
        # Officers prefer recent, actionable incidents
        if user_role == UserRole.OFFICER:
            if doc_type == 'incident' and data.get('status') in ['open', 'investigating']:
                return 1.0
            return 0.7
        
        # Analysts prefer detailed historical data
        elif user_role == UserRole.ANALYST:
            if doc_type == 'incident' and data.get('investigation_notes'):
                return 1.0
            return 0.8
        
        # Administrators get balanced view
        elif user_role == UserRole.ADMINISTRATOR:
            return 0.9
        
        return 0.5
    
    def _calculate_threat_level_score(self, data: Dict[str, Any], threat_level: int) -> float:
        """Calculate score based on current threat level."""
        if threat_level >= 4:  # High threat
            # Prioritize high-risk vehicles and severe incidents
            if data.get('risk_score', 0) > 0.7 or data.get('severity_level', 1) >= 4:
                return 1.0
            return 0.6
        elif threat_level >= 3:  # Medium threat
            return 0.8
        else:  # Low threat
            return 0.5
    
    def _filter_by_time_range(
        self,
        results: List[Tuple[str, float, Dict[str, Any], float]],
        cutoff_time: datetime
    ) -> List[Tuple[str, float, Dict[str, Any], float]]:
        """Filter results by time range."""
        filtered = []
        for doc_id, sim_score, metadata, rank_score in results:
            data = metadata.get('data', {})
            doc_type = metadata.get('type')
            
            # Get timestamp based on document type
            if doc_type == 'vehicle':
                timestamp_str = data.get('updated_at') or data.get('created_at')
            elif doc_type == 'incident':
                timestamp_str = data.get('timestamp')
            else:
                continue
            
            if timestamp_str:
                try:
                    if isinstance(timestamp_str, str):
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    else:
                        timestamp = timestamp_str
                    
                    if timestamp.replace(tzinfo=None) >= cutoff_time:
                        filtered.append((doc_id, sim_score, metadata, rank_score))
                except Exception:
                    # Include if timestamp parsing fails
                    filtered.append((doc_id, sim_score, metadata, rank_score))
        
        return filtered
    
    def _filter_by_location(
        self,
        results: List[Tuple[str, float, Dict[str, Any], float]],
        location_filters: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any], float]]:
        """Filter results by location criteria."""
        # TODO: Implement location-based filtering
        # This would filter by radius, specific locations, etc.
        return results
    
    def _filter_vehicles(
        self,
        results: List[Tuple[str, float, Dict[str, Any], float]],
        vehicle_filters: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any], float]]:
        """Filter vehicle results by specific criteria."""
        filtered = []
        for doc_id, sim_score, metadata, rank_score in results:
            if metadata.get('type') != 'vehicle':
                filtered.append((doc_id, sim_score, metadata, rank_score))
                continue
            
            data = metadata.get('data', {})
            
            # Apply filters
            if 'risk_score_min' in vehicle_filters:
                if data.get('risk_score', 0) < vehicle_filters['risk_score_min']:
                    continue
            
            if 'status' in vehicle_filters:
                if data.get('status') != vehicle_filters['status']:
                    continue
            
            filtered.append((doc_id, sim_score, metadata, rank_score))
        
        return filtered
    
    def _filter_incidents(
        self,
        results: List[Tuple[str, float, Dict[str, Any], float]],
        incident_filters: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any], float]]:
        """Filter incident results by specific criteria."""
        filtered = []
        for doc_id, sim_score, metadata, rank_score in results:
            if metadata.get('type') != 'incident':
                filtered.append((doc_id, sim_score, metadata, rank_score))
                continue
            
            data = metadata.get('data', {})
            
            # Apply filters
            if 'severity_min' in incident_filters:
                if data.get('severity_level', 1) < incident_filters['severity_min']:
                    continue
            
            if 'incident_type' in incident_filters:
                if data.get('incident_type') != incident_filters['incident_type']:
                    continue
            
            if 'status' in incident_filters:
                if data.get('status') != incident_filters['status']:
                    continue
            
            filtered.append((doc_id, sim_score, metadata, rank_score))
        
        return filtered
    
    def _convert_to_typed_results(
        self,
        results: List[Tuple[str, float, Dict[str, Any]]],
        query_request: QueryRequest
    ) -> Tuple[List[VehicleSearchResult], List[IncidentSearchResult], List[RetrievalResult]]:
        """
        Convert raw results to typed result objects.
        
        Args:
            results: Raw search results
            query_request: Original query request
            
        Returns:
            Tuple of typed results
        """
        vehicle_results = []
        incident_results = []
        retrieval_results = []
        
        for rank, (doc_id, similarity_score, metadata) in enumerate(results[:query_request.max_results], 1):
            doc_type = metadata.get('type')
            data = metadata.get('data', {})
            
            # Create retrieval result
            retrieval_result = RetrievalResult(
                content=metadata.get('searchable_text', ''),
                source=f"{doc_type}_database",
                score=similarity_score,
                metadata={
                    'doc_id': doc_id,
                    'type': doc_type,
                    'rank': rank
                }
            )
            retrieval_results.append(retrieval_result)
            
            # Create typed results
            if doc_type == 'vehicle':
                try:
                    vehicle = Vehicle(**data)
                    vehicle_result = VehicleSearchResult(
                        vehicle=vehicle,
                        similarity_score=similarity_score,
                        rank=rank,
                        match_reasons=self._determine_match_reasons(metadata, query_request)
                    )
                    vehicle_results.append(vehicle_result)
                except Exception as e:
                    logger.error("Failed to create vehicle result", doc_id=doc_id, error=str(e))
            
            elif doc_type == 'incident':
                try:
                    incident = Incident(**data)
                    incident_result = IncidentSearchResult(
                        incident=incident,
                        relevance_score=similarity_score,
                        rank=rank,
                        match_reasons=self._determine_match_reasons(metadata, query_request)
                    )
                    incident_results.append(incident_result)
                except Exception as e:
                    logger.error("Failed to create incident result", doc_id=doc_id, error=str(e))
        
        return vehicle_results, incident_results, retrieval_results
    
    def _determine_match_reasons(self, metadata: Dict[str, Any], query_request: QueryRequest) -> List[str]:
        """Determine why a document matched the query."""
        reasons = []
        
        # TODO: Implement more sophisticated match reason detection
        # This could analyze which parts of the document matched the query
        
        if metadata.get('type') == 'vehicle':
            reasons.append("vehicle_content_match")
        elif metadata.get('type') == 'incident':
            reasons.append("incident_content_match")
        
        if query_request.search_mode == SearchMode.SEMANTIC:
            reasons.append("semantic_similarity")
        elif query_request.search_mode == SearchMode.EXACT:
            reasons.append("exact_match")
        else:
            reasons.append("hybrid_match")
        
        return reasons
