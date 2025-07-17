"""
Modern ColBERT Qdrant Retriever
==============================

Qdrant retrieval implementation for Modern ColBERT with native multi-vector support.
This module provides efficient retrieval using Modern ColBERT query embeddings
with Qdrant's native multi-vector functionality and MaxSim comparator.
"""

from typing import List, Tuple, Optional, Dict, Any
import logging
import numpy as np

try:
    from qdrant_client import QdrantClient
except ImportError:
    raise ImportError("qdrant-client is required. Install with: pip install lateness")

logger = logging.getLogger(__name__)


class ModernColBERTQdrantRetriever:
    """
    Modern ColBERT Qdrant Retriever with native multi-vector support
    
    This class handles retrieval using Modern ColBERT query embeddings
    from Qdrant vector database with native multi-vector support and MaxSim.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to search
        
    Example:
        ```python
        from qdrant_client import QdrantClient
        from lateness import ModernColBERT, QdrantRetriever
        
        client = QdrantClient("localhost", port=6333)
        retriever = QdrantRetriever(client, "my_collection")
        colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
        
        # Search with query embeddings
        query_embeddings = colbert.encode_queries(["What is AI?"])
        results = retriever.search(query_embeddings, top_k=10)
        
        # Search with text query
        results = retriever.search_with_query("What is AI?", colbert, top_k=10)
        
        # Simple search (auto-loads model)
        results = retriever.search_simple("What is AI?", top_k=10)
        ```
    """
    
    def __init__(self, client: QdrantClient, collection_name: str, vector_name: str = "colbert"):
        """Initialize Modern ColBERT Qdrant Retriever"""
        self.client = client
        self.collection_name = collection_name
        self.vector_name = vector_name
        
        logger.info(f"Initialized Modern ColBERT Qdrant Retriever for collection: {collection_name}")
    
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def search(self, query_embeddings: np.ndarray, top_k: int = 10, 
               score_threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Search using pre-encoded Modern ColBERT query embeddings with native multi-vector support
        
        Args:
            query_embeddings: Pre-encoded query embeddings from Modern ColBERT
            top_k: Number of top results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        if not self.collection_exists():
            raise ValueError(f"Collection {self.collection_name} does not exist")
        
        # Handle single query embeddings: shape (seq_len, hidden_size)
        if len(query_embeddings.shape) == 2:
            # Convert to multi-vector format for native MaxSim
            query_multi_vector = [token_embedding.tolist() for token_embedding in query_embeddings]
        elif len(query_embeddings.shape) == 3 and query_embeddings.shape[0] == 1:
            # Single query in batch: shape (1, seq_len, hidden_size)
            query_multi_vector = [token_embedding.tolist() for token_embedding in query_embeddings[0]]
        else:
            raise ValueError(f"Unsupported query embeddings shape: {query_embeddings.shape}")
        
        logger.debug(f"Query shape: {query_embeddings.shape}")
        logger.debug(f"Query multi-vector length: {len(query_multi_vector)}")
        logger.debug(f"First token embedding length: {len(query_multi_vector[0])}")
        
        try:
            # Use native multi-vector search with MaxSim
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_multi_vector,  # Multi-vector query for native MaxSim
                using=self.vector_name,  # Use configurable vector name
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold
            )
            
            logger.debug(f"Search returned {len(search_results.points)} results")
            if search_results.points:
                logger.debug(f"First result score: {search_results.points[0].score}")
                logger.debug(f"First result doc_id: {search_results.points[0].payload.get('doc_id')}")
            
            # Extract results
            results = []
            for point in search_results.points:
                doc_id = point.payload["doc_id"]
                score = point.score
                results.append((doc_id, score))
            
            return results
            
        except Exception as e:
            logger.error(f"Native multi-vector search failed: {e}")
            
            # Fallback to single vector search using first token
            try:
                logger.info("Trying fallback single vector search...")
                test_vector = query_embeddings[0].tolist() if len(query_embeddings.shape) == 2 else query_embeddings[0][0].tolist()
                
                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=test_vector,
                    vector_name=self.vector_name,  # Use configurable vector name
                    limit=min(top_k, 10),
                    with_payload=True,
                    with_vectors=False,
                    score_threshold=score_threshold
                )
                
                logger.info(f"Fallback search returned {len(search_results)} results")
                results = []
                for point in search_results:
                    doc_id = point.payload["doc_id"]
                    score = point.score
                    results.append((doc_id, score))
                
                return results
                
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def search_with_query(self, query_text: str, colbert_model, top_k: int = 10,
                         score_threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Search with text query using Modern ColBERT model
        
        Args:
            query_text: Query text string
            colbert_model: Modern ColBERT model instance
            top_k: Number of top results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        # Encode query using Modern ColBERT
        query_embeddings = colbert_model.encode_queries([query_text], batch_size=1, to_cpu=True)
        
        # Handle both torch and numpy backends
        if hasattr(query_embeddings, 'numpy'):
            query_embeddings = query_embeddings.numpy()
        
        # Get first query if batch
        if len(query_embeddings.shape) == 3:
            query_embeddings = query_embeddings[0]
        
        # Perform search
        return self.search(query_embeddings, top_k, score_threshold)
    
    def search_simple(self, query_text: str, top_k: int = 10,
                     score_threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Simple search with automatic model loading
        
        Args:
            query_text: Query text string
            top_k: Number of top results to return
            score_threshold: Minimum score threshold for results
            
        Returns:
            List of (doc_id, score) tuples sorted by score
        """
        # Import here to avoid circular import
        from .. import ModernColBERT
        
        # Use default model for simple search
        colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
        
        return self.search_with_query(query_text, colbert, top_k, score_threshold)
    
    def batch_search(self, queries: List[str], colbert_model, top_k: int = 10,
                    score_threshold: Optional[float] = None) -> Dict[str, List[Tuple[str, float]]]:
        """
        Batch search for multiple queries
        
        Args:
            queries: List of query text strings
            colbert_model: Modern ColBERT model instance
            top_k: Number of top results to return per query
            score_threshold: Minimum score threshold for results
            
        Returns:
            Dictionary mapping query index to list of (doc_id, score) tuples
        """
        results = {}
        
        # Encode all queries at once
        query_embeddings = colbert_model.encode_queries(queries, to_cpu=True)
        
        # Handle both torch and numpy backends
        if hasattr(query_embeddings, 'numpy'):
            query_embeddings = query_embeddings.numpy()
        
        # Search for each query
        for i, query_emb in enumerate(query_embeddings):
            results[str(i)] = self.search(query_emb, top_k, score_threshold)
        
        return results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve document by ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            Document payload if found, None otherwise
        """
        try:
            # Search by payload filter
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter={
                    "must": [
                        {
                            "key": "doc_id",
                            "match": {"value": doc_id}
                        }
                    ]
                },
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            
            if search_results[0]:
                return search_results[0][0].payload
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve document {doc_id}: {e}")
            return None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "points_count": info.points_count,
                "vectors_count": info.vectors_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def __repr__(self) -> str:
        return f"ModernColBERTQdrantRetriever(collection='{self.collection_name}')"