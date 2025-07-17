"""
Modern ColBERT Retrieval Module
==============================

Retrieval components for Modern ColBERT with Qdrant vector database integration.
"""

from .qdrant_retriever import ModernColBERTQdrantRetriever

__all__ = ["ModernColBERTQdrantRetriever"]