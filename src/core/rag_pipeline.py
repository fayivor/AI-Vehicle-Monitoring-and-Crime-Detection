"""RAG pipeline orchestrator combining retrieval and LLM processing."""

import time
import uuid
from typing import Dict, Any, List
from datetime import datetime

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from .retrieval import HybridRetriever
from .vector_store import VectorStore
from ..models.query import QueryRequest, QueryResponse, RetrievalResult
from ..models.user import UserRole, User
from ..mcp.mcp_integration import MCPProcessor
from ..utils.config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGPipeline:
    """
    RAG pipeline orchestrator implementing the complete flow:
    Input → Vector Similarity (FAISS) → Rank-Based Filtering → Context Assembly → LLM Processing → Output
    """
    
    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize the RAG pipeline.

        Args:
            vector_store: Vector store instance
        """
        self.vector_store = vector_store or VectorStore()
        self.retriever = HybridRetriever(self.vector_store)
        self.llm = self._initialize_llm()
        self.prompt_templates = self._load_prompt_templates()
        self.mcp_processor = MCPProcessor()
        
    def _initialize_llm(self) -> OpenAI:
        """Initialize the OpenAI LLM."""
        return OpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            max_tokens=settings.openai_max_tokens,
            temperature=settings.openai_temperature,
        )
    
    def _load_prompt_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates for different query types and user roles."""
        templates = {}
        
        # Base template for general queries
        base_template = """You are an AI assistant for Ghana's Vehicle Monitoring & Crime Detection System.

User Role: {user_role}
Query Type: {query_type}
Threat Level: {threat_level}

Context Information:
{context}

User Query: {query}

Based on the provided context and your role as an assistant for Ghana law enforcement, provide a comprehensive and accurate response. Focus on actionable information relevant to the user's role:

- Officers: Provide immediate actionable intelligence and clear next steps
- Analysts: Include detailed analysis, patterns, and historical context  
- Administrators: Include system metrics, summaries, and operational insights
- Supervisors: Provide oversight-level information with key highlights

Response:"""

        templates['general'] = PromptTemplate(
            input_variables=['user_role', 'query_type', 'threat_level', 'context', 'query'],
            template=base_template
        )
        
        # Vehicle search specific template
        vehicle_template = """You are an AI assistant for Ghana's Vehicle Monitoring & Crime Detection System.

User Role: {user_role}
Query: {query}

Vehicle Search Results:
{vehicle_results}

Incident Context:
{incident_context}

Additional Context:
{additional_context}

Provide a detailed analysis of the vehicle search results. Include:
1. Summary of vehicles found
2. Risk assessment and priority ranking
3. Any concerning patterns or flags
4. Recommended actions based on user role
5. Related incidents or connections

Focus on actionable intelligence for Ghana law enforcement operations.

Response:"""

        templates['vehicle_search'] = PromptTemplate(
            input_variables=['user_role', 'query', 'vehicle_results', 'incident_context', 'additional_context'],
            template=vehicle_template
        )
        
        # Incident search specific template
        incident_template = """You are an AI assistant for Ghana's Vehicle Monitoring & Crime Detection System.

User Role: {user_role}
Query: {query}

Incident Search Results:
{incident_results}

Vehicle Context:
{vehicle_context}

Additional Context:
{additional_context}

Provide a comprehensive analysis of the incident search results. Include:
1. Summary of incidents found
2. Severity assessment and urgency ranking
3. Pattern analysis and trends
4. Vehicle connections and risk factors
5. Recommended investigative actions

Tailor the response to the user's role and operational needs.

Response:"""

        templates['incident_search'] = PromptTemplate(
            input_variables=['user_role', 'query', 'incident_results', 'vehicle_context', 'additional_context'],
            template=incident_template
        )
        
        # Risk assessment template
        risk_template = """You are an AI assistant for Ghana's Vehicle Monitoring & Crime Detection System.

User Role: {user_role}
Query: {query}

Risk Assessment Data:
{risk_data}

Historical Context:
{historical_context}

Current Threat Level: {threat_level}

Provide a comprehensive risk assessment including:
1. Overall risk evaluation and scoring
2. Key risk factors and indicators
3. Trend analysis and predictions
4. Mitigation recommendations
5. Monitoring priorities

Focus on actionable risk management for Ghana law enforcement.

Response:"""

        templates['risk_assessment'] = PromptTemplate(
            input_variables=['user_role', 'query', 'risk_data', 'historical_context', 'threat_level'],
            template=risk_template
        )
        
        return templates
    
    async def process_query(self, query_request: QueryRequest, user: User = None) -> QueryResponse:
        """
        Process a query through the complete RAG pipeline with MCP integration.

        Args:
            query_request: Query request with context and parameters
            user: User making the request (for MCP session management)

        Returns:
            Query response with results and metadata
        """
        start_time = time.time()
        query_id = f"query-{uuid.uuid4()}"
        
        logger.info(
            "Starting RAG pipeline processing",
            query_id=query_id,
            query_type=query_request.query_type,
            user_role=query_request.user_role
        )
        
        try:
            # Initialize MCP session if user provided
            user_session = None
            if user:
                user_session = await self.mcp_processor.initialize_user_session(
                    user,
                    location=query_request.location,
                    intent=query_request.intent
                )

            # Step 1: Hybrid Retrieval (Vector Similarity + Rank-Based Filtering)
            vehicle_results, incident_results, retrieval_results = self.retriever.retrieve(query_request)

            # Step 2: MCP Context Processing
            mcp_context = {}
            if user_session:
                mcp_context = await self.mcp_processor.process_query_with_mcp(
                    query_request, user_session, retrieval_results
                )

            # Step 3: Context Assembly (enhanced with MCP)
            context = self._assemble_context(
                vehicle_results, incident_results, retrieval_results, query_request, mcp_context
            )

            # Step 4: LLM Processing
            response_text, confidence_score = await self._generate_response(
                query_request, context, vehicle_results, incident_results, mcp_context
            )
            
            # Step 5: Create response (enhanced with MCP metadata)
            processing_time = int((time.time() - start_time) * 1000)

            response = QueryResponse(
                query_id=query_id,
                response_text=response_text,
                confidence_score=confidence_score,
                vehicle_results=vehicle_results,
                incident_results=incident_results,
                retrieval_results=retrieval_results,
                context_used=context.get('context_summary', []),
                sources_cited=context.get('sources', []),
                processing_time_ms=processing_time,
                user_role=query_request.user_role,
                context_shaped=bool(mcp_context),
                security_filtered=True,
                retrieval_precision_at_k=self._calculate_precision_at_k(retrieval_results),
                response_quality_score=confidence_score
            )
            
            logger.info(
                "RAG pipeline processing completed",
                query_id=query_id,
                processing_time_ms=processing_time,
                confidence_score=confidence_score,
                results_count=len(retrieval_results)
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "RAG pipeline processing failed",
                query_id=query_id,
                error=str(e)
            )
            
            # Return error response
            return QueryResponse(
                query_id=query_id,
                response_text="I apologize, but I encountered an error processing your query. Please try again or contact system support.",
                confidence_score=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                user_role=query_request.user_role,
                context_shaped=False,
                security_filtered=True
            )
    
    def _assemble_context(
        self,
        vehicle_results: List,
        incident_results: List,
        retrieval_results: List[RetrievalResult],
        query_request: QueryRequest,
        mcp_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Assemble context for LLM processing with MCP integration.

        Args:
            vehicle_results: Vehicle search results
            incident_results: Incident search results
            retrieval_results: Raw retrieval results
            query_request: Original query request
            mcp_context: MCP-processed context and metadata

        Returns:
            Assembled context dictionary
        """
        context = {
            'context_summary': [],
            'sources': [],
            'vehicle_summary': '',
            'incident_summary': '',
            'additional_context': ''
        }
        
        # Summarize vehicle results
        if vehicle_results:
            vehicle_summaries = []
            for result in vehicle_results[:5]:  # Top 5 vehicles
                vehicle = result.vehicle
                summary = f"Vehicle {vehicle.registration_number} (Risk: {vehicle.risk_score:.2f}, Status: {vehicle.status})"
                vehicle_summaries.append(summary)
                context['sources'].append(f"vehicle_{vehicle.vehicle_id}")
            
            context['vehicle_summary'] = "\n".join(vehicle_summaries)
            context['context_summary'].append(f"Found {len(vehicle_results)} relevant vehicles")
        
        # Summarize incident results
        if incident_results:
            incident_summaries = []
            for result in incident_results[:5]:  # Top 5 incidents
                incident = result.incident
                summary = f"Incident {incident.incident_id} ({incident.incident_type}, Severity: {incident.severity_level})"
                incident_summaries.append(summary)
                context['sources'].append(f"incident_{incident.incident_id}")
            
            context['incident_summary'] = "\n".join(incident_summaries)
            context['context_summary'].append(f"Found {len(incident_results)} relevant incidents")
        
        # Add retrieval context
        if retrieval_results:
            context_pieces = []
            for result in retrieval_results[:10]:  # Top 10 pieces
                context_pieces.append(f"- {result.content[:200]}...")
            
            context['additional_context'] = "\n".join(context_pieces)
            context['context_summary'].append(f"Retrieved {len(retrieval_results)} context pieces")

        # Add MCP context if available
        if mcp_context:
            context['mcp_metadata'] = mcp_context.get('mcp_metadata', {})
            context['role_context'] = mcp_context.get('role_context', '')
            context['threat_assessment'] = mcp_context.get('threat_assessment', {})
            context['context_summary'].append("Enhanced with MCP context shaping")

        return context
    
    async def _generate_response(
        self,
        query_request: QueryRequest,
        context: Dict[str, Any],
        vehicle_results: List,
        incident_results: List,
        mcp_context: Dict[str, Any] = None
    ) -> tuple[str, float]:
        """
        Generate response using LLM.
        
        Args:
            query_request: Original query request
            context: Assembled context
            vehicle_results: Vehicle search results
            incident_results: Incident search results
            
        Returns:
            Tuple of (response_text, confidence_score)
        """
        try:
            # Select appropriate prompt template
            template_key = query_request.query_type
            if template_key not in self.prompt_templates:
                template_key = 'general'
            
            template = self.prompt_templates[template_key]
            
            # Prepare template variables
            if template_key == 'vehicle_search':
                template_vars = {
                    'user_role': query_request.user_role,
                    'query': query_request.query_text,
                    'vehicle_results': context['vehicle_summary'],
                    'incident_context': context['incident_summary'],
                    'additional_context': context['additional_context']
                }
            elif template_key == 'incident_search':
                template_vars = {
                    'user_role': query_request.user_role,
                    'query': query_request.query_text,
                    'incident_results': context['incident_summary'],
                    'vehicle_context': context['vehicle_summary'],
                    'additional_context': context['additional_context']
                }
            elif template_key == 'risk_assessment':
                template_vars = {
                    'user_role': query_request.user_role,
                    'query': query_request.query_text,
                    'risk_data': context['vehicle_summary'] + "\n" + context['incident_summary'],
                    'historical_context': context['additional_context'],
                    'threat_level': query_request.threat_level
                }
            else:  # general
                combined_context = "\n".join([
                    context['vehicle_summary'],
                    context['incident_summary'],
                    context['additional_context']
                ])
                template_vars = {
                    'user_role': query_request.user_role,
                    'query_type': query_request.query_type,
                    'threat_level': query_request.threat_level,
                    'context': combined_context,
                    'query': query_request.query_text
                }
            
            # Create and run chain
            chain = LLMChain(llm=self.llm, prompt=template)
            response_text = await chain.arun(**template_vars)
            
            # Calculate confidence score based on context quality
            confidence_score = self._calculate_confidence_score(
                context, len(vehicle_results), len(incident_results)
            )
            
            return response_text.strip(), confidence_score
            
        except Exception as e:
            logger.error("LLM response generation failed", error=str(e))
            return "I apologize, but I'm unable to generate a response at this time. Please try again.", 0.0
    
    def _calculate_confidence_score(
        self,
        context: Dict[str, Any],
        vehicle_count: int,
        incident_count: int
    ) -> float:
        """Calculate confidence score based on context quality."""
        base_score = 0.5
        
        # Boost confidence based on results found
        if vehicle_count > 0:
            base_score += min(vehicle_count * 0.1, 0.3)
        
        if incident_count > 0:
            base_score += min(incident_count * 0.1, 0.3)
        
        # Boost based on context richness
        context_pieces = len(context.get('context_summary', []))
        if context_pieces > 0:
            base_score += min(context_pieces * 0.05, 0.2)
        
        return min(base_score, 1.0)
    
    def _calculate_precision_at_k(self, retrieval_results: List[RetrievalResult]) -> float:
        """Calculate precision@k metric for retrieval quality."""
        if not retrieval_results:
            return 0.0
        
        # Simple precision calculation based on score threshold
        relevant_count = sum(1 for result in retrieval_results if result.score >= 0.7)
        return relevant_count / len(retrieval_results)
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        vector_stats = self.vector_store.get_stats()
        
        return {
            'vector_store': vector_stats,
            'llm_model': settings.openai_model,
            'max_tokens': settings.openai_max_tokens,
            'temperature': settings.openai_temperature,
            'last_updated': datetime.utcnow().isoformat()
        }
