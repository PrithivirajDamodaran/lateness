"""
Lateness - Modern ColBERT for Late Interaction
==============================================

Smart backend selection based on installation:
- pip install lateness → ONNX backend (lightweight retrieval)  
- pip install lateness[index] → PyTorch backend (heavy indexing)

Example Usage:
   ```python
   # RETRIEVAL SERVICE (lightweight)
   pip install lateness
   from lateness import ModernColBERT
   colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")  # ONNX
   
   # INDEXING SERVICE (heavy)
   pip install lateness[index] 
   from lateness import ModernColBERT
   colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")  # PyTorch
   ```
"""

import os
from .models.model_manager import download_model_from_hf

# Conservative backend detection - default to ONNX, require explicit PyTorch
def _detect_backend():
    """Detect which backend to use - defaults to ONNX for reliability"""
    
    # Check for explicit PyTorch request - MUST be exactly 'true'
    use_torch_env = os.environ.get('LATENESS_USE_TORCH', '').lower()
    use_torch = use_torch_env == 'true'
    
    if use_torch:
        try:
            import torch
            import transformers
            print("🚀 Using PyTorch backend (LATENESS_USE_TORCH=true)")
            from .backends.torch_colbert import ModernColBERT
            return ModernColBERT
        except ImportError:
            print("❌ PyTorch backend requested but dependencies missing")
            print("💡 Install with: pip install lateness[index]")
            print("🚀 Falling back to ONNX backend")
            # Fall through to ONNX
    else:
        # Give clear feedback about why ONNX is being used
        if use_torch_env and use_torch_env != 'true':
            print(f"🚀 Using ONNX backend (LATENESS_USE_TORCH='{use_torch_env}' - only 'true' enables PyTorch)")
        else:
            print("🚀 Using ONNX backend (default, for GPU accelerated indexing, install lateness[index] and set LATENESS_USE_TORCH=true)")
    
    # Default to ONNX backend
    from .backends.onnx_colbert import ModernColBERT
    return ModernColBERT

# Export the appropriate ModernColBERT based on installation
ModernColBERT = _detect_backend()

# Qdrant integration (always available since qdrant-client is in base requirements)
from .indexing.qdrant_indexer import ModernColBERTQdrantIndexer as QdrantIndexer
from .retrieval.qdrant_retriever import ModernColBERTQdrantRetriever as QdrantRetriever

__version__ = "0.1.17"
__author__ = "Modern ColBERT Team"
__description__ = "Modern ColBERT for Late Interaction with native multi-vector support"

__all__ = [
    "ModernColBERT", 
    "download_model_from_hf",
    "QdrantIndexer", 
    "QdrantRetriever"
]