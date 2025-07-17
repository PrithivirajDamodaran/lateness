"""
Modern ColBERT Qdrant Indexer
============================

Qdrant indexing implementation for Modern ColBERT with native multi-vector support.
This module provides efficient indexing of documents using Modern ColBERT embeddings
with Qdrant's native multi-vector functionality and MaxSim comparator.
"""

import uuid
from typing import Dict, List, Optional, Any, Union
import logging
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        VectorParams, Distance, CollectionInfo, PointStruct,
        MultiVectorConfig, MultiVectorComparator
    )
except ImportError:
    raise ImportError("qdrant-client is required. Install with: pip install lateness")

logger = logging.getLogger(__name__)


class ModernColBERTQdrantIndexer:
    """
    Modern ColBERT Qdrant Indexer with native multi-vector support
    
    This class handles indexing of documents using Modern ColBERT embeddings
    into Qdrant vector database with native multi-vector support and MaxSim.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection to create/use
        vector_size: Size of embedding vectors (default: 128)
        
    Example:
        ```python
        from qdrant_client import QdrantClient
        from lateness import ModernColBERT, QdrantIndexer
        
        client = QdrantClient("localhost", port=6333)
        indexer = QdrantIndexer(client, "my_collection")
        colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
        
        # Create collection
        indexer.create_collection()
        
        # Index documents
        corpus = {
            "doc_1": {"title": "Title 1", "text": "Document content 1"},
            "doc_2": {"title": "Title 2", "text": "Document content 2"}
        }
        indexer.index_documents(corpus, colbert, batch_size=32)
        ```
    """
    
    def __init__(self, client: QdrantClient, collection_name: str, vector_size: int = 128, vector_name: str = "colbert"):
        """Initialize Modern ColBERT Qdrant Indexer"""
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.vector_name = vector_name
        
        logger.info(f"Initialized Modern ColBERT Qdrant Indexer for collection: {collection_name}")
    
    def create_collection(self, force_recreate: bool = False) -> None:
        """
        Create Qdrant collection with native multi-vector support for Modern ColBERT
        
        Args:
            force_recreate: Whether to recreate collection if it exists
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(col.name == self.collection_name for col in collections)
            
            if collection_exists:
                if force_recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} already exists")
                    return
            
            # Create collection with multi-vector configuration
            logger.info(f"Creating collection {self.collection_name} with multi-vector support...")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    self.vector_name: VectorParams(
                        size=self.vector_size,
                        distance=Distance.DOT,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM
                        )
                    )
                }
            )
            
            logger.info(f"✅ Collection {self.collection_name} created successfully with MaxSim multi-vector support")
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.collection_name}: {e}")
            raise
    
    def collection_exists(self) -> bool:
        """Check if collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(col.name == self.collection_name for col in collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def get_collection_info(self) -> Optional[CollectionInfo]:
        """Get collection information"""
        try:
            return self.client.get_collection(self.collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def index_documents(self, corpus: Dict[str, Dict[str, str]], colbert_model, 
                       batch_size: int = 32, show_progress: bool = True) -> None:
        """
        Index documents using Modern ColBERT embeddings
        
        Args:
            corpus: Dictionary of {doc_id: {"title": str, "text": str}}
            colbert_model: Modern ColBERT model instance
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
        """
        if not self.collection_exists():
            raise ValueError(f"Collection {self.collection_name} does not exist. Call create_collection() first.")
        
        logger.info(f"Indexing {len(corpus)} documents with Modern ColBERT...")
        
        # Prepare documents for encoding
        doc_ids = list(corpus.keys())
        documents = []
        
        for doc_id in doc_ids:
            doc_data = corpus[doc_id]
            # Combine title and text
            title = doc_data.get("title", "")
            text = doc_data.get("text", "")
            
            if title:
                full_text = f"{title}. {text}".strip()
            else:
                full_text = text.strip()
            
            documents.append(full_text)
        
        # Process documents in batches
        points = []
        
        progress_bar = tqdm(range(0, len(documents), batch_size), 
                          desc="Encoding document batches", 
                          disable=not show_progress)
        
        for i in progress_bar:
            batch_docs = documents[i:i + batch_size]
            batch_ids = doc_ids[i:i + batch_size]
            
            # Encode documents with Modern ColBERT
            try:
                doc_embeddings = colbert_model.encode_documents(
                    batch_docs, 
                    batch_size=len(batch_docs),
                    keep_dims=False,  # Get list of individual document embeddings
                    to_cpu=True
                )
                
                # Create points for Qdrant
                for j, (doc_id, embeddings) in enumerate(zip(batch_ids, doc_embeddings)):
                    # Convert to list of lists for multi-vector
                    if hasattr(embeddings, 'tolist'):
                        embedding_vectors = embeddings.tolist()
                    else:
                        embedding_vectors = embeddings
                    
                    # Create point with multi-vector embeddings
                    point = PointStruct(
                        id=str(uuid.uuid4()),  # Generate unique ID
                        vector={
                            self.vector_name: embedding_vectors
                        },
                        payload={
                            "doc_id": doc_id,
                            "title": corpus[doc_id].get("title", ""),
                            "text": corpus[doc_id].get("text", ""),
                            "full_text": batch_docs[j]
                        }
                    )
                    points.append(point)
                
            except Exception as e:
                logger.error(f"Failed to encode batch starting at index {i}: {e}")
                continue
        
        # Upload points to Qdrant
        if points:
            logger.info(f"Uploading {len(points)} points to Qdrant...")
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"✅ Successfully indexed {len(points)} documents")
                
            except Exception as e:
                logger.error(f"Failed to upload points to Qdrant: {e}")
                raise
        else:
            logger.warning("No points to upload")
    
    def index_documents_simple(self, documents: List[str], show_progress: bool = True) -> None:
        """
        Simple document indexing for a list of strings
        
        Args:
            documents: List of document strings
            show_progress: Whether to show progress bar
        """
        # Convert to corpus format
        corpus = {}
        for i, doc in enumerate(documents):
            corpus[f"doc_{i}"] = {"title": "", "text": doc}
        
        # Import here to avoid circular import
        from .. import ModernColBERT
        
        # Use default model for simple indexing
        colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
        
        self.index_documents(corpus, colbert, show_progress=show_progress)
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents by their IDs
        
        Args:
            doc_ids: List of document IDs to delete
        """
        try:
            # Delete by payload filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={
                    "filter": {
                        "must": [
                            {
                                "key": "doc_id",
                                "match": {"any": doc_ids}
                            }
                        ]
                    }
                }
            )
            logger.info(f"Deleted {len(doc_ids)} documents")
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection"""
        try:
            # Delete all points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector={"filter": {"must": []}}  # Empty filter matches all
            )
            logger.info(f"Cleared all documents from collection {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise
    
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
        return f"ModernColBERTQdrantIndexer(collection='{self.collection_name}', vector_size={self.vector_size})"