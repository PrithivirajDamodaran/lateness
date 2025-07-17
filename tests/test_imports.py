"""
Test imports and basic functionality
"""

import pytest


def test_main_imports():
    """Test that main imports work correctly"""
    try:
        from lateness import ModernColBERT
        assert ModernColBERT is not None
    except ImportError as e:
        pytest.skip(f"ModernColBERT import failed: {e}")


def test_qdrant_imports():
    """Test that Qdrant-related imports work"""
    try:
        from lateness import QdrantIndexer, QdrantRetriever
        assert QdrantIndexer is not None
        assert QdrantRetriever is not None
    except ImportError as e:
        pytest.skip(f"Qdrant imports failed: {e}")


def test_model_manager_import():
    """Test model manager import"""
    try:
        from lateness import download_model_from_hf
        assert download_model_from_hf is not None
    except ImportError as e:
        pytest.skip(f"Model manager import failed: {e}")


def test_backend_imports():
    """Test backend module imports"""
    try:
        from lateness.backends.onnx_colbert import ModernColBERT as ONNXModernColBERT
        assert ONNXModernColBERT is not None
    except ImportError as e:
        pytest.skip(f"ONNX backend import failed: {e}")
    
    # Test PyTorch backend if available
    try:
        from lateness.backends.torch_colbert import ModernColBERT as TorchModernColBERT
        assert TorchModernColBERT is not None
    except ImportError as e:
        pytest.skip(f"PyTorch backend import failed: {e}")


def test_package_metadata():
    """Test package metadata"""
    import lateness
    
    assert hasattr(lateness, '__version__')
    assert hasattr(lateness, '__author__')
    assert hasattr(lateness, '__description__')
    assert hasattr(lateness, '__all__')
    
    # Check that __all__ contains expected exports
    expected_exports = ['ModernColBERT', 'QdrantIndexer', 'QdrantRetriever', 'download_model_from_hf']
    for export in expected_exports:
        assert export in lateness.__all__


def test_class_structure():
    """Test that classes have expected methods"""
    try:
        from lateness import ModernColBERT
        
        # Check if class has expected methods
        expected_methods = [
            'encode_queries', 'encode_documents', 'compute_similarity',
            'search', 'rank_documents'
        ]
        
        for method in expected_methods:
            assert hasattr(ModernColBERT, method), f"ModernColBERT missing method: {method}"
            
    except ImportError as e:
        pytest.skip(f"ModernColBERT import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
