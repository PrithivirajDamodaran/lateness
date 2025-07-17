"""
Modern ColBERT Backends
=======================

Backend implementations for Modern ColBERT:
- torch_colbert.py: PyTorch backend for GPU indexing
- onnx_colbert.py: ONNX backend for CPU retrieval
"""

# Backends are imported dynamically based on available dependencies
# See lateness/__init__.py for backend selection logic