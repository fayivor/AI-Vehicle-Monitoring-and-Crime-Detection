"""Evaluation framework for RAG pipeline performance assessment."""

import time
import json
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

from ..models.query import QueryRequest, QueryResponse
from ..models.vehicle import Vehicle, VehicleSearchResult
from ..models.incident import Incident, IncidentSearchResult
from ..utils.logging import get_logger
from ..utils.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class GroundTruthItem:
    """Ground truth item for evaluation."""
    query: str
    expected_vehicle_ids: List[str]
    expected_incident_ids: List[str]
    expected_response_keywords: List[str]
    relevance_threshold: float = 0.7
    user_role: str = "officer"


@dataclass
class EvaluationResult:
    """Evaluation result for a single query."""
    query: str
    precision_at_5: float
    precision_at_10: float
    recall_at_5: float
    recall_at_10: float
    bleu_score: float
    response_time_ms: int
    confidence_score: float
    human_rating: Optional[float] = None
    notes: str = ""


@dataclass
class BenchmarkResults:
    """Complete benchmark evaluation results."""
    overall_precision_at_5: float
    overall_precision_at_10: float
    overall_recall_at_5: float
    overall_recall_at_10: float
    average_bleu_score: float
    average_response_time: float
    average_confidence: float
    human_accuracy_rating: float
    false_positive_rate: float
    individual_results: List[EvaluationResult]
    evaluation_timestamp: datetime
    total_queries: int


class BLEUScorer:
    """BLEU score calculator for response quality assessment."""
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str, n_gram: int = 4) -> float:
        """
        Calculate BLEU score between reference and candidate text.
        
        Args:
            reference: Reference (ground truth) text
            candidate: Candidate (generated) text
            n_gram: Maximum n-gram order
            
        Returns:
            BLEU score (0-1)
        """
        try:
            # Tokenize texts
            ref_tokens = reference.lower().split()
            cand_tokens = candidate.lower().split()
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            # Calculate precision for each n-gram order
            precisions = []
            
            for n in range(1, n_gram + 1):
                ref_ngrams = BLEUScorer._get_ngrams(ref_tokens, n)
                cand_ngrams = BLEUScorer._get_ngrams(cand_tokens, n)
                
                if not cand_ngrams:
                    precisions.append(0.0)
                    continue
                
                # Count matches
                matches = 0
                for ngram in cand_ngrams:
                    if ngram in ref_ngrams:
                        matches += min(cand_ngrams[ngram], ref_ngrams[ngram])
                
                precision = matches / sum(cand_ngrams.values())
                precisions.append(precision)
            
            # Calculate brevity penalty
            ref_length = len(ref_tokens)
            cand_length = len(cand_tokens)
            
            if cand_length > ref_length:
                bp = 1.0
            else:
                bp = np.exp(1 - ref_length / cand_length) if cand_length > 0 else 0.0
            
            # Calculate geometric mean of precisions
            if all(p > 0 for p in precisions):
                geo_mean = np.exp(np.mean(np.log(precisions)))
            else:
                geo_mean = 0.0
            
            bleu = bp * geo_mean
            return min(bleu, 1.0)
            
        except Exception as e:
            logger.error("BLEU calculation failed", error=str(e))
            return 0.0
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        """Extract n-grams from tokens."""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams


class RAGEvaluator:
    """
    RAG pipeline evaluator implementing custom benchmarks as specified in guidelines.
    
    Evaluation metrics:
    - Precision@k for retrieval relevance
    - BLEU scores for LLM response quality
    - Human-in-the-loop accuracy validation
    - Performance benchmarks for real-time processing
    """
    
    def __init__(self, rag_pipeline=None):
        """
        Initialize the evaluator.
        
        Args:
            rag_pipeline: RAG pipeline instance to evaluate
        """
        self.rag_pipeline = rag_pipeline
        self.bleu_scorer = BLEUScorer()
        self.ground_truth_data: List[GroundTruthItem] = []
        self.evaluation_history: List[BenchmarkResults] = []
    
    def load_ground_truth(self, file_path: str) -> None:
        """
        Load ground truth data for evaluation.
        
        Args:
            file_path: Path to ground truth JSON file
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            self.ground_truth_data = []
            for item in data:
                ground_truth = GroundTruthItem(
                    query=item['query'],
                    expected_vehicle_ids=item.get('expected_vehicle_ids', []),
                    expected_incident_ids=item.get('expected_incident_ids', []),
                    expected_response_keywords=item.get('expected_response_keywords', []),
                    relevance_threshold=item.get('relevance_threshold', 0.7),
                    user_role=item.get('user_role', 'officer')
                )
                self.ground_truth_data.append(ground_truth)
            
            logger.info(
                "Loaded ground truth data",
                file_path=file_path,
                items_count=len(self.ground_truth_data)
            )
            
        except Exception as e:
            logger.error("Failed to load ground truth data", file_path=file_path, error=str(e))
            raise
    
    async def evaluate_single_query(
        self,
        ground_truth: GroundTruthItem,
        user=None
    ) -> EvaluationResult:
        """
        Evaluate a single query against ground truth.
        
        Args:
            ground_truth: Ground truth item
            user: User object for the query
            
        Returns:
            Evaluation result
        """
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline not provided")
        
        # Create query request
        from ..models.query import QueryRequest, QueryType, SearchMode
        from ..models.user import UserRole
        
        query_request = QueryRequest(
            query_text=ground_truth.query,
            query_type=QueryType.GENERAL_INQUIRY,
            search_mode=SearchMode.HYBRID,
            user_role=UserRole(ground_truth.user_role),
            max_results=10
        )
        
        # Execute query and measure time
        start_time = time.time()
        response = await self.rag_pipeline.process_query(query_request, user)
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate precision and recall
        precision_5, recall_5 = self._calculate_precision_recall(
            response, ground_truth, k=5
        )
        precision_10, recall_10 = self._calculate_precision_recall(
            response, ground_truth, k=10
        )
        
        # Calculate BLEU score
        reference_text = " ".join(ground_truth.expected_response_keywords)
        bleu_score = self.bleu_scorer.calculate_bleu(reference_text, response.response_text)
        
        return EvaluationResult(
            query=ground_truth.query,
            precision_at_5=precision_5,
            precision_at_10=precision_10,
            recall_at_5=recall_5,
            recall_at_10=recall_10,
            bleu_score=bleu_score,
            response_time_ms=response_time_ms,
            confidence_score=response.confidence_score
        )
    
    def _calculate_precision_recall(
        self,
        response: QueryResponse,
        ground_truth: GroundTruthItem,
        k: int
    ) -> Tuple[float, float]:
        """Calculate precision and recall at k."""
        # Extract retrieved IDs
        retrieved_vehicle_ids = [
            result.vehicle.vehicle_id 
            for result in response.vehicle_results[:k]
        ]
        retrieved_incident_ids = [
            result.incident.incident_id 
            for result in response.incident_results[:k]
        ]
        
        all_retrieved = set(retrieved_vehicle_ids + retrieved_incident_ids)
        all_expected = set(ground_truth.expected_vehicle_ids + ground_truth.expected_incident_ids)
        
        if not all_retrieved:
            return 0.0, 0.0
        
        # Calculate precision and recall
        relevant_retrieved = all_retrieved.intersection(all_expected)
        
        precision = len(relevant_retrieved) / len(all_retrieved) if all_retrieved else 0.0
        recall = len(relevant_retrieved) / len(all_expected) if all_expected else 0.0
        
        return precision, recall
    
    async def run_benchmark(self, user=None) -> BenchmarkResults:
        """
        Run complete benchmark evaluation.
        
        Args:
            user: User object for queries
            
        Returns:
            Benchmark results
        """
        if not self.ground_truth_data:
            raise ValueError("No ground truth data loaded")
        
        logger.info("Starting benchmark evaluation", queries_count=len(self.ground_truth_data))
        
        individual_results = []
        
        for i, ground_truth in enumerate(self.ground_truth_data):
            try:
                result = await self.evaluate_single_query(ground_truth, user)
                individual_results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Evaluated {i + 1}/{len(self.ground_truth_data)} queries")
                    
            except Exception as e:
                logger.error(
                    "Query evaluation failed",
                    query=ground_truth.query,
                    error=str(e)
                )
                # Add failed result
                failed_result = EvaluationResult(
                    query=ground_truth.query,
                    precision_at_5=0.0,
                    precision_at_10=0.0,
                    recall_at_5=0.0,
                    recall_at_10=0.0,
                    bleu_score=0.0,
                    response_time_ms=0,
                    confidence_score=0.0,
                    notes=f"Evaluation failed: {str(e)}"
                )
                individual_results.append(failed_result)
        
        # Calculate overall metrics
        benchmark_results = self._calculate_overall_metrics(individual_results)
        
        # Store results
        self.evaluation_history.append(benchmark_results)
        
        logger.info(
            "Benchmark evaluation completed",
            precision_at_5=benchmark_results.overall_precision_at_5,
            precision_at_10=benchmark_results.overall_precision_at_10,
            bleu_score=benchmark_results.average_bleu_score,
            response_time=benchmark_results.average_response_time
        )
        
        return benchmark_results
    
    def _calculate_overall_metrics(self, results: List[EvaluationResult]) -> BenchmarkResults:
        """Calculate overall benchmark metrics."""
        if not results:
            return BenchmarkResults(
                overall_precision_at_5=0.0,
                overall_precision_at_10=0.0,
                overall_recall_at_5=0.0,
                overall_recall_at_10=0.0,
                average_bleu_score=0.0,
                average_response_time=0.0,
                average_confidence=0.0,
                human_accuracy_rating=0.0,
                false_positive_rate=0.0,
                individual_results=[],
                evaluation_timestamp=datetime.utcnow(),
                total_queries=0
            )
        
        # Calculate averages
        precision_5 = np.mean([r.precision_at_5 for r in results])
        precision_10 = np.mean([r.precision_at_10 for r in results])
        recall_5 = np.mean([r.recall_at_5 for r in results])
        recall_10 = np.mean([r.recall_at_10 for r in results])
        bleu_score = np.mean([r.bleu_score for r in results])
        response_time = np.mean([r.response_time_ms for r in results])
        confidence = np.mean([r.confidence_score for r in results])
        
        # Calculate human accuracy rating (if available)
        human_ratings = [r.human_rating for r in results if r.human_rating is not None]
        human_accuracy = np.mean(human_ratings) if human_ratings else 0.0
        
        # Calculate false positive rate (simplified)
        false_positives = sum(1 for r in results if r.precision_at_5 < 0.5)
        false_positive_rate = false_positives / len(results)
        
        return BenchmarkResults(
            overall_precision_at_5=precision_5,
            overall_precision_at_10=precision_10,
            overall_recall_at_5=recall_5,
            overall_recall_at_10=recall_10,
            average_bleu_score=bleu_score,
            average_response_time=response_time,
            average_confidence=confidence,
            human_accuracy_rating=human_accuracy,
            false_positive_rate=false_positive_rate,
            individual_results=results,
            evaluation_timestamp=datetime.utcnow(),
            total_queries=len(results)
        )
    
    def save_results(self, results: BenchmarkResults, file_path: str) -> None:
        """Save benchmark results to file."""
        try:
            # Convert to serializable format
            results_dict = {
                'overall_precision_at_5': results.overall_precision_at_5,
                'overall_precision_at_10': results.overall_precision_at_10,
                'overall_recall_at_5': results.overall_recall_at_5,
                'overall_recall_at_10': results.overall_recall_at_10,
                'average_bleu_score': results.average_bleu_score,
                'average_response_time': results.average_response_time,
                'average_confidence': results.average_confidence,
                'human_accuracy_rating': results.human_accuracy_rating,
                'false_positive_rate': results.false_positive_rate,
                'evaluation_timestamp': results.evaluation_timestamp.isoformat(),
                'total_queries': results.total_queries,
                'individual_results': [
                    {
                        'query': r.query,
                        'precision_at_5': r.precision_at_5,
                        'precision_at_10': r.precision_at_10,
                        'recall_at_5': r.recall_at_5,
                        'recall_at_10': r.recall_at_10,
                        'bleu_score': r.bleu_score,
                        'response_time_ms': r.response_time_ms,
                        'confidence_score': r.confidence_score,
                        'human_rating': r.human_rating,
                        'notes': r.notes
                    }
                    for r in results.individual_results
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(results_dict, f, indent=2)
            
            logger.info("Saved benchmark results", file_path=file_path)
            
        except Exception as e:
            logger.error("Failed to save results", file_path=file_path, error=str(e))
            raise
    
    def generate_sample_ground_truth(self, file_path: str) -> None:
        """Generate sample ground truth data for testing."""
        sample_data = [
            {
                "query": "Find high-risk vehicles near Tema Port",
                "expected_vehicle_ids": ["VH-001-2024", "VH-002-2024"],
                "expected_incident_ids": [],
                "expected_response_keywords": ["high-risk", "vehicles", "Tema Port", "monitoring"],
                "relevance_threshold": 0.7,
                "user_role": "officer"
            },
            {
                "query": "Recent smuggling incidents at border crossings",
                "expected_vehicle_ids": [],
                "expected_incident_ids": ["INC-001-2024", "INC-002-2024"],
                "expected_response_keywords": ["smuggling", "border", "incidents", "investigation"],
                "relevance_threshold": 0.8,
                "user_role": "analyst"
            },
            {
                "query": "System performance metrics and alerts",
                "expected_vehicle_ids": [],
                "expected_incident_ids": [],
                "expected_response_keywords": ["system", "performance", "metrics", "alerts", "monitoring"],
                "relevance_threshold": 0.6,
                "user_role": "administrator"
            }
        ]
        
        with open(file_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info("Generated sample ground truth data", file_path=file_path)
