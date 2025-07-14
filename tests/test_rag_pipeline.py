"""Test RAG pipeline functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch

from src.core.rag_pipeline import RAGPipeline
from src.core.vector_store import VectorStore
from src.models.query import QueryRequest, QueryType, SearchMode
from src.models.user import UserRole
from src.models.vehicle import Vehicle, OwnerDetails
from src.models.incident import Incident, Location


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    vector_store = Mock(spec=VectorStore)
    vector_store.get_stats.return_value = {
        'total_documents': 10,
        'vehicle_count': 6,
        'incident_count': 4,
        'dimension': 384,
        'index_trained': True
    }
    return vector_store


@pytest.fixture
def sample_vehicle():
    """Sample vehicle for testing."""
    return Vehicle(
        vehicle_id="VH-TEST-001",
        registration_number="GH-1234-24",
        owner_details=OwnerDetails(
            name="Test Owner",
            id_number="GHA123456789",
            phone="+233123456789",
            address="Test Address",
            email="test@example.com"
        ),
        risk_score=0.3,
        make="Toyota",
        model="Camry",
        year=2020,
        color="Blue",
        status="active"
    )


@pytest.fixture
def sample_incident():
    """Sample incident for testing."""
    return Incident(
        incident_id="INC-TEST-001",
        vehicle_involved="VH-TEST-001",
        incident_type="speeding",
        location=Location(
            latitude=5.6037,
            longitude=-0.1870,
            address="Test Location"
        ),
        severity_level=2,
        description="Test incident description",
        status="open",
        reported_by="Test Officer"
    )


@pytest.fixture
def sample_query_request():
    """Sample query request for testing."""
    return QueryRequest(
        query_text="Find high-risk vehicles",
        query_type=QueryType.VEHICLE_SEARCH,
        search_mode=SearchMode.HYBRID,
        user_role=UserRole.OFFICER,
        max_results=10
    )


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_pipeline_initialization(self, mock_openai, mock_vector_store):
        """Test RAG pipeline initialization."""
        pipeline = RAGPipeline(mock_vector_store)
        
        assert pipeline.vector_store == mock_vector_store
        assert pipeline.retriever is not None
        assert pipeline.prompt_templates is not None
        assert 'general' in pipeline.prompt_templates
        assert 'vehicle_search' in pipeline.prompt_templates
        assert 'incident_search' in pipeline.prompt_templates
    
    @patch('src.core.rag_pipeline.OpenAI')
    @patch('src.core.retrieval.HybridRetriever')
    @pytest.mark.asyncio
    async def test_query_processing(self, mock_retriever_class, mock_openai, mock_vector_store, sample_query_request):
        """Test query processing through RAG pipeline."""
        # Setup mocks
        mock_retriever = Mock()
        mock_retriever_class.return_value = mock_retriever
        
        # Mock retrieval results
        mock_retriever.retrieve.return_value = ([], [], [])
        
        # Mock LLM response
        mock_llm = Mock()
        mock_llm.arun = asyncio.coroutine(lambda **kwargs: "Test response")
        mock_openai.return_value = mock_llm
        
        # Create pipeline and process query
        pipeline = RAGPipeline(mock_vector_store)
        pipeline.llm = mock_llm
        
        response = await pipeline.process_query(sample_query_request)
        
        # Verify response
        assert response is not None
        assert response.query_id is not None
        assert response.user_role == UserRole.OFFICER
        assert response.processing_time_ms > 0
        assert response.context_shaped is True
        assert response.security_filtered is True
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_context_assembly(self, mock_openai, mock_vector_store, sample_vehicle, sample_incident, sample_query_request):
        """Test context assembly for LLM processing."""
        from src.models.vehicle import VehicleSearchResult
        from src.models.incident import IncidentSearchResult
        from src.models.query import RetrievalResult
        
        pipeline = RAGPipeline(mock_vector_store)
        
        # Create mock results
        vehicle_results = [VehicleSearchResult(
            vehicle=sample_vehicle,
            similarity_score=0.9,
            rank=1,
            match_reasons=["test_match"]
        )]
        
        incident_results = [IncidentSearchResult(
            incident=sample_incident,
            relevance_score=0.8,
            rank=1,
            match_reasons=["test_match"]
        )]
        
        retrieval_results = [RetrievalResult(
            content="Test content",
            source="test_source",
            score=0.9,
            metadata={"test": "data"}
        )]
        
        # Test context assembly
        context = pipeline._assemble_context(
            vehicle_results, incident_results, retrieval_results, sample_query_request
        )
        
        assert 'context_summary' in context
        assert 'sources' in context
        assert 'vehicle_summary' in context
        assert 'incident_summary' in context
        assert len(context['context_summary']) > 0
        assert len(context['sources']) > 0
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_confidence_score_calculation(self, mock_openai, mock_vector_store):
        """Test confidence score calculation."""
        pipeline = RAGPipeline(mock_vector_store)
        
        # Test with good context
        context = {
            'context_summary': ['Found vehicles', 'Found incidents'],
            'sources': ['source1', 'source2']
        }
        
        score = pipeline._calculate_confidence_score(context, 5, 3)
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high with good context
        
        # Test with poor context
        empty_context = {
            'context_summary': [],
            'sources': []
        }
        
        score = pipeline._calculate_confidence_score(empty_context, 0, 0)
        assert score == 0.5  # Base score
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_precision_calculation(self, mock_openai, mock_vector_store):
        """Test precision@k calculation."""
        from src.models.query import RetrievalResult
        
        pipeline = RAGPipeline(mock_vector_store)
        
        # Test with high-quality results
        good_results = [
            RetrievalResult(content="test", source="test", score=0.9, metadata={}),
            RetrievalResult(content="test", source="test", score=0.8, metadata={}),
            RetrievalResult(content="test", source="test", score=0.6, metadata={})
        ]
        
        precision = pipeline._calculate_precision_at_k(good_results)
        assert precision == 2/3  # 2 out of 3 above threshold (0.7)
        
        # Test with empty results
        precision = pipeline._calculate_precision_at_k([])
        assert precision == 0.0
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_pipeline_stats(self, mock_openai, mock_vector_store):
        """Test pipeline statistics."""
        pipeline = RAGPipeline(mock_vector_store)
        
        stats = pipeline.get_pipeline_stats()
        
        assert 'vector_store' in stats
        assert 'llm_model' in stats
        assert 'max_tokens' in stats
        assert 'temperature' in stats
        assert 'last_updated' in stats


class TestPromptTemplates:
    """Test prompt template functionality."""
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_template_loading(self, mock_openai):
        """Test that all required templates are loaded."""
        pipeline = RAGPipeline()
        
        required_templates = ['general', 'vehicle_search', 'incident_search', 'risk_assessment']
        
        for template_name in required_templates:
            assert template_name in pipeline.prompt_templates
            template = pipeline.prompt_templates[template_name]
            assert template is not None
            assert hasattr(template, 'input_variables')
            assert hasattr(template, 'template')
    
    @patch('src.core.rag_pipeline.OpenAI')
    def test_template_variables(self, mock_openai):
        """Test that templates have correct input variables."""
        pipeline = RAGPipeline()
        
        # Test general template
        general_template = pipeline.prompt_templates['general']
        expected_vars = ['user_role', 'query_type', 'threat_level', 'context', 'query']
        for var in expected_vars:
            assert var in general_template.input_variables
        
        # Test vehicle search template
        vehicle_template = pipeline.prompt_templates['vehicle_search']
        expected_vars = ['user_role', 'query', 'vehicle_results', 'incident_context', 'additional_context']
        for var in expected_vars:
            assert var in vehicle_template.input_variables
