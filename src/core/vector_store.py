"""FAISS vector store implementation for vehicle and incident data."""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import faiss
from sentence_transformers import SentenceTransformer

from ..utils.config import get_settings
from ..utils.logging import get_logger
from ..models.vehicle import Vehicle
from ..models.incident import Incident

logger = get_logger(__name__)
settings = get_settings()


class VectorStore:
    """
    FAISS-based vector store for similarity search.
    
    Implements the vector similarity component of the hybrid RAG pipeline:
    Input → Vector Similarity (FAISS) → Rank-Based Filtering → Context Assembly → LLM Processing → Output
    """
    
    def __init__(self, dimension: int = None, index_path: str = None):
        """
        Initialize the vector store.
        
        Args:
            dimension: Vector dimension (defaults to embedding model dimension)
            index_path: Path to save/load FAISS index
        """
        self.dimension = dimension or settings.vector_dimension
        self.index_path = index_path or settings.faiss_index_path
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        
        # Initialize FAISS index with IVFPQ for large-scale similarity search
        # as specified in guidelines
        self.index = None
        self.metadata_store = {}  # Store metadata for each vector
        self.id_to_index = {}     # Map document IDs to index positions
        self.index_to_id = {}     # Map index positions to document IDs
        
        self._initialize_index()
        
    def _initialize_index(self) -> None:
        """Initialize or load FAISS index."""
        try:
            if os.path.exists(f"{self.index_path}/faiss.index"):
                self._load_index()
                logger.info("Loaded existing FAISS index", path=self.index_path)
            else:
                self._create_new_index()
                logger.info("Created new FAISS index", dimension=self.dimension)
        except Exception as e:
            logger.error("Failed to initialize FAISS index", error=str(e))
            self._create_new_index()
    
    def _create_new_index(self) -> None:
        """Create a new FAISS index."""
        # Use IVFPQ index for large-scale similarity search as per guidelines
        quantizer = faiss.IndexFlatL2(self.dimension)
        self.index = faiss.IndexIVFPQ(quantizer, self.dimension, 100, 8, 8)
        
        # Initialize with random vectors if needed
        if not self.index.is_trained:
            # Generate some random training data
            training_data = np.random.random((1000, self.dimension)).astype('float32')
            self.index.train(training_data)
            
        os.makedirs(self.index_path, exist_ok=True)
    
    def _load_index(self) -> None:
        """Load existing FAISS index and metadata."""
        try:
            self.index = faiss.read_index(f"{self.index_path}/faiss.index")
            
            # Load metadata
            with open(f"{self.index_path}/metadata.pkl", 'rb') as f:
                data = pickle.load(f)
                self.metadata_store = data.get('metadata_store', {})
                self.id_to_index = data.get('id_to_index', {})
                self.index_to_id = data.get('index_to_id', {})
                
        except Exception as e:
            logger.error("Failed to load FAISS index", error=str(e))
            raise
    
    def _save_index(self) -> None:
        """Save FAISS index and metadata."""
        try:
            os.makedirs(self.index_path, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}/faiss.index")
            
            # Save metadata
            metadata = {
                'metadata_store': self.metadata_store,
                'id_to_index': self.id_to_index,
                'index_to_id': self.index_to_id
            }
            with open(f"{self.index_path}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
                
            logger.info("Saved FAISS index", path=self.index_path)
            
        except Exception as e:
            logger.error("Failed to save FAISS index", error=str(e))
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for text using sentence transformer.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0].astype('float32')
        except Exception as e:
            logger.error("Failed to generate embedding", text=text[:100], error=str(e))
            # Return zero vector as fallback
            return np.zeros(self.dimension, dtype='float32')
    
    def add_vehicle(self, vehicle: Vehicle) -> None:
        """
        Add vehicle to vector store.
        
        Args:
            vehicle: Vehicle object to add
        """
        try:
            # Create searchable text from vehicle data
            searchable_text = self._vehicle_to_text(vehicle)
            
            # Generate embedding
            embedding = self._generate_embedding(searchable_text)
            
            # Add to index
            self._add_vector(vehicle.vehicle_id, embedding, {
                'type': 'vehicle',
                'data': vehicle.dict(),
                'searchable_text': searchable_text
            })
            
            logger.info("Added vehicle to vector store", vehicle_id=vehicle.vehicle_id)
            
        except Exception as e:
            logger.error("Failed to add vehicle", vehicle_id=vehicle.vehicle_id, error=str(e))
    
    def add_incident(self, incident: Incident) -> None:
        """
        Add incident to vector store.
        
        Args:
            incident: Incident object to add
        """
        try:
            # Create searchable text from incident data
            searchable_text = self._incident_to_text(incident)
            
            # Generate embedding
            embedding = self._generate_embedding(searchable_text)
            
            # Add to index
            self._add_vector(incident.incident_id, embedding, {
                'type': 'incident',
                'data': incident.dict(),
                'searchable_text': searchable_text
            })
            
            logger.info("Added incident to vector store", incident_id=incident.incident_id)
            
        except Exception as e:
            logger.error("Failed to add incident", incident_id=incident.incident_id, error=str(e))
    
    def _add_vector(self, doc_id: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Add vector to FAISS index.
        
        Args:
            doc_id: Document identifier
            embedding: Vector embedding
            metadata: Associated metadata
        """
        # Get next index position
        index_pos = self.index.ntotal
        
        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))
        
        # Store metadata
        self.metadata_store[index_pos] = metadata
        self.id_to_index[doc_id] = index_pos
        self.index_to_id[index_pos] = doc_id
    
    def search(
        self, 
        query: str, 
        k: int = 10, 
        filter_type: Optional[str] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter_type: Filter by document type ('vehicle' or 'incident')
            
        Returns:
            List of (doc_id, similarity_score, metadata) tuples
        """
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search FAISS index
            # Use larger k for filtering if needed
            search_k = k * 5 if filter_type else k
            distances, indices = self.index.search(query_embedding.reshape(1, -1), search_k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx == -1:  # Invalid index
                    continue
                    
                metadata = self.metadata_store.get(idx, {})
                
                # Apply type filter
                if filter_type and metadata.get('type') != filter_type:
                    continue
                
                # Convert L2 distance to similarity score (0-1)
                similarity_score = 1.0 / (1.0 + distance)
                
                doc_id = self.index_to_id.get(idx, f"unknown_{idx}")
                results.append((doc_id, similarity_score, metadata))
                
                if len(results) >= k:
                    break
            
            logger.info(
                "Vector search completed",
                query=query[:100],
                results_count=len(results),
                filter_type=filter_type
            )
            
            return results
            
        except Exception as e:
            logger.error("Vector search failed", query=query[:100], error=str(e))
            return []
    
    def _vehicle_to_text(self, vehicle: Vehicle) -> str:
        """
        Convert vehicle object to searchable text.
        
        Args:
            vehicle: Vehicle object
            
        Returns:
            Searchable text representation
        """
        parts = [
            f"Vehicle ID: {vehicle.vehicle_id}",
            f"Registration: {vehicle.registration_number}",
            f"Owner: {vehicle.owner_details.name}",
            f"Make: {vehicle.make or 'Unknown'}",
            f"Model: {vehicle.model or 'Unknown'}",
            f"Year: {vehicle.year or 'Unknown'}",
            f"Color: {vehicle.color or 'Unknown'}",
            f"Status: {vehicle.status}",
            f"Risk Score: {vehicle.risk_score}",
        ]
        
        if vehicle.alert_flags:
            parts.append(f"Alert Flags: {', '.join(vehicle.alert_flags)}")
            
        if vehicle.last_seen_location:
            parts.append(f"Last Location: {vehicle.last_seen_location.address or 'Unknown'}")
        
        return " | ".join(parts)
    
    def _incident_to_text(self, incident: Incident) -> str:
        """
        Convert incident object to searchable text.
        
        Args:
            incident: Incident object
            
        Returns:
            Searchable text representation
        """
        parts = [
            f"Incident ID: {incident.incident_id}",
            f"Vehicle: {incident.vehicle_involved}",
            f"Type: {incident.incident_type}",
            f"Severity: {incident.severity_level}",
            f"Status: {incident.status}",
            f"Description: {incident.description}",
            f"Location: {incident.location.address or 'Unknown'}",
            f"Reported by: {incident.reported_by or 'Unknown'}",
        ]
        
        if incident.investigation_notes:
            parts.append(f"Notes: {' '.join(incident.investigation_notes)}")
        
        return " | ".join(parts)
    
    def rebuild_index(self) -> None:
        """Rebuild the FAISS index for data freshness as per guidelines."""
        logger.info("Starting index rebuild")
        
        # Save current data
        old_metadata = self.metadata_store.copy()
        
        # Create new index
        self._create_new_index()
        self.metadata_store = {}
        self.id_to_index = {}
        self.index_to_id = {}
        
        # Re-add all documents
        for metadata in old_metadata.values():
            if metadata.get('type') == 'vehicle':
                vehicle = Vehicle(**metadata['data'])
                self.add_vehicle(vehicle)
            elif metadata.get('type') == 'incident':
                incident = Incident(**metadata['data'])
                self.add_incident(incident)
        
        # Save rebuilt index
        self._save_index()
        
        logger.info("Index rebuild completed", total_documents=self.index.ntotal)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        vehicle_count = sum(1 for m in self.metadata_store.values() if m.get('type') == 'vehicle')
        incident_count = sum(1 for m in self.metadata_store.values() if m.get('type') == 'incident')
        
        return {
            'total_documents': self.index.ntotal if self.index else 0,
            'vehicle_count': vehicle_count,
            'incident_count': incident_count,
            'dimension': self.dimension,
            'index_trained': self.index.is_trained if self.index else False,
        }
    
    def save(self) -> None:
        """Save the vector store."""
        self._save_index()
    
    def close(self) -> None:
        """Close and save the vector store."""
        self.save()
