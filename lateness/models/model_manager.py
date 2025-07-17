"""
Modern ColBERT Model Management
==============================

Utilities for downloading and managing Modern ColBERT models from Hugging Face Hub.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Optional
from huggingface_hub import snapshot_download

# Setup logging
logger = logging.getLogger(__name__)


def download_model_from_hf(model_name: str, cache_dir: Optional[str] = None) -> Tuple[str, str]:
    """
    Download Modern ColBERT model from Hugging Face Hub
    
    Args:
        model_name: Name of the model on Hugging Face Hub
        cache_dir: Directory to cache downloaded models
        
    Returns:
        Tuple of (model_path, onnx_model_path)
        
    Raises:
        FileNotFoundError: If ONNX model is not found in the downloaded model
        Exception: If download fails
    """
    logger.info(f"Downloading Modern ColBERT model {model_name} from Hugging Face...")
    
    try:
        # Download the entire model repository
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_files_only=False
        )
        
        # Construct ONNX model path
        onnx_model_path = os.path.join(model_path, "onnx", "model.onnx")
        
        # Verify ONNX model exists
        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")
        
        logger.info(f"Modern ColBERT model downloaded successfully to {model_path}")
        logger.info(f"ONNX model found at {onnx_model_path}")
        
        return model_path, onnx_model_path
        
    except Exception as e:
        logger.error(f"Failed to download Modern ColBERT model {model_name}: {e}")
        raise


def get_model_cache_path(model_name: str, cache_dir: str = "./models") -> str:
    """
    Get the cache path for a model without downloading
    
    Args:
        model_name: Name of the model
        cache_dir: Cache directory
        
    Returns:
        Expected cache path for the model
    """
    # This follows the huggingface_hub caching structure
    cache_path = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    return str(cache_path)


def is_model_cached(model_name: str, cache_dir: str = "./models") -> bool:
    """
    Check if a model is already cached locally
    
    Args:
        model_name: Name of the model
        cache_dir: Cache directory
        
    Returns:
        True if model is cached, False otherwise
    """
    cache_path = get_model_cache_path(model_name, cache_dir)
    onnx_path = os.path.join(cache_path, "snapshots", "*", "onnx", "model.onnx")
    
    # Check if any snapshot contains the ONNX model
    import glob
    return len(glob.glob(onnx_path)) > 0