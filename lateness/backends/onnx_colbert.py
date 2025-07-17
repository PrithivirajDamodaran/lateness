"""
ONNX Modern ColBERT Implementation
=================================

Lightweight backend for fast CPU retrieval.
Available with: pip install lateness (base install)
"""

import numpy as np
import onnxruntime as ort
from tokenizers import AddedToken, Tokenizer
import json
import string
from pathlib import Path
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
from ..models.model_manager import download_model_from_hf


class ModernColBERT:
    """Modern ColBERT ONNX Backend (Lightweight Retrieval)"""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None,
                 max_query_len: int = 256, max_doc_len: int = 300,
                 providers: Optional[List[str]] = None):
        
        # Download model if needed
        if not Path(model_name).exists():
            print(f"🔄 Downloading model: {model_name}")
            model_path, onnx_path = download_model_from_hf(model_name, cache_dir)
        else:
            model_path = model_name
            onnx_path = Path(model_name) / "onnx" / "model.onnx"
        
        self.model_dir = Path(model_path)
        self.tokenizer = self._get_tokenizer(max_length=512)
        self.max_query_len = max_query_len
        self.max_doc_len = max_doc_len
        self.backend_type = "onnx"
        
        # Setup inference configuration
        self.Q_PID = self.tokenizer.token_to_id("[unused0]")
        self.D_PID = self.tokenizer.token_to_id("[unused1]") 
        self.mask_token_id = self.tokenizer.token_to_id("[MASK]")
        
        if None in [self.Q_PID, self.D_PID, self.mask_token_id]:
            raise ValueError("Could not find required special tokens in tokenizer")
        
        # Setup post-tokenization punctuation masking
        self.skip_ids = set()
        for c in string.punctuation:
            encoded = self.tokenizer.encode(c, add_special_tokens=False)
            if len(encoded.ids) > 0:
                self.skip_ids.add(encoded.ids[0])
        
        print(f"Identified {len(self.skip_ids)} punctuation token IDs to skip")
        
        # Initialize ONNX Runtime session
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        try:
            self.session = ort.InferenceSession(str(onnx_path), providers=providers)
            print(f"✅ ONNX ColBERT loaded with providers: {self.session.get_providers()}")
        except Exception as e:
            print(f"❌ Error loading ONNX: {e}")
            self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            print(f"✅ ONNX ColBERT loaded (CPU only)")
        
        print(f"Query max length: {max_query_len}, Document max length: {max_doc_len}")

    def _get_tokenizer(self, max_length: int = 512) -> Tokenizer:
        """Initialize tokenizer"""
        with open(str(self.model_dir / "config.json")) as config_file:
            config = json.load(config_file)
        with open(str(self.model_dir / "tokenizer_config.json")) as tokenizer_config_file:
            tokenizer_config = json.load(tokenizer_config_file)
        with open(str(self.model_dir / "special_tokens_map.json")) as tokens_map_file:
            tokens_map = json.load(tokens_map_file)
        
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        tokenizer.enable_truncation(max_length=min(tokenizer_config["model_max_length"], max_length))
        tokenizer.enable_padding(pad_id=config["pad_token_id"], pad_token=tokenizer_config["pad_token"])
        
        for token in tokens_map.values():
            if isinstance(token, str):
                tokenizer.add_special_tokens([token])
            elif isinstance(token, dict):
                tokenizer.add_special_tokens([AddedToken(**token)])
        
        return tokenizer

    def _encode_batch(self, ids: np.ndarray, mask: np.ndarray, to_cpu: bool = False) -> np.ndarray:
        """Internal encoding function"""
        # Create position IDs
        pos = np.arange(ids.shape[1])[None, :].repeat(ids.shape[0], axis=0)
        
        # ONNX inference
        inputs = {
            "input_ids": ids.astype(np.int64),
            "attention_mask": mask.astype(np.int64),
            "position_ids": pos.astype(np.int64)
        }
        
        outputs = self.session.run(["last_hidden_state"], inputs)
        return outputs[0]

    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None, 
                      to_cpu: bool = False) -> np.ndarray:
        """Encode queries"""
        print(f"Encoding {len(queries)} queries...")
        
        # Tokenize with query prefix
        encoded_queries = self.tokenizer.encode_batch(queries, add_special_tokens=True)
        id_lists = [[self.Q_PID] + encoded.ids for encoded in encoded_queries]
        
        # Apply dynamic augmentation with length cap
        cap = self.max_query_len or 511
        id_lists = [_dynamic_augment(ids, self.mask_token_id, cap) for ids in id_lists]
        
        # Manual padding
        max_len = max(len(ids) for ids in id_lists)
        batch_size_actual = len(id_lists)
        
        ids = np.zeros((batch_size_actual, max_len), dtype=np.int64)
        mask = np.zeros((batch_size_actual, max_len), dtype=np.int64)
        
        for i, id_list in enumerate(id_lists):
            ids[i, :len(id_list)] = id_list
            mask[i, :len(id_list)] = 1
        
        # Process in batches if specified
        if batch_size:
            reps = []
            for i, a in tqdm(_split_into_batches(ids, mask, batch_size), desc="Encoding query batches"):
                reps.append(self._encode_batch(i, a, to_cpu))
            return np.concatenate(reps, axis=0)
        
        return self._encode_batch(ids, mask, to_cpu)

    def encode_documents(self, documents: List[str], batch_size: Optional[int] = None,
                        keep_dims: bool = True, to_cpu: bool = False) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode documents"""
        print(f"Encoding {len(documents)} documents...")
        
        # Encode documents individually to preserve natural lengths
        encoded_docs = []
        for doc in documents:
            encoded = self.tokenizer.encode(doc, add_special_tokens=True)
            encoded_docs.append(encoded)
        
        id_lists = []
        for encoded in encoded_docs:
            ids = encoded.ids
            # Truncate to max_doc_len - 1
            if len(ids) > self.max_doc_len - 1:
                ids = ids[:self.max_doc_len - 1]
            # Add D_PID prefix
            ids = [self.D_PID] + ids
            id_lists.append(ids)
        
        # Manual padding
        max_len = max(len(ids) for ids in id_lists)
        batch_size_actual = len(id_lists)
        
        ids = np.zeros((batch_size_actual, max_len), dtype=np.int64)
        mask = np.zeros((batch_size_actual, max_len), dtype=np.int64)
        
        for i, id_list in enumerate(id_lists):
            ids[i, :len(id_list)] = id_list
            mask[i, :len(id_list)] = 1
        
        # Apply post-tokenization punctuation masking
        for skip_id in self.skip_ids:
            mask[ids == skip_id] = 0
        
        # Process in batches if specified
        if batch_size:
            ids_s, mask_s, rev = _sort_by_length(ids, mask, batch_size)
            reps = []
            
            for i, a in tqdm(_split_into_batches(ids_s, mask_s, batch_size), desc="Encoding document batches"):
                rep = self._encode_batch(i, a, to_cpu)
                if not keep_dims:
                    m = a.astype(bool)
                    rep = [r[m[idx]] for idx, r in enumerate(rep)]
                reps.append(rep)
            
            if keep_dims:
                return _stack_3D_arrays(reps)[rev]
            else:
                flat = [d for g in reps for d in g]
                return [flat[i] for i in rev.tolist()]
        
        # Single batch processing
        rep = self._encode_batch(ids, mask, to_cpu)
        if not keep_dims:
            m = mask.astype(bool)
            rep = [r[m[idx]] for idx, r in enumerate(rep)]
        
        return rep

    @staticmethod
    def compute_similarity(q_reps: np.ndarray, p_reps: np.ndarray) -> np.ndarray:
        """Compute Modern ColBERT similarity"""
        token_scores = np.einsum("qin,pjn->qipj", q_reps, p_reps)
        scores = np.max(token_scores, axis=-1)
        scores = np.sum(scores, axis=1)
        return scores

    def search(self, queries: List[str], documents: List[str],
               batch_size: Optional[int] = None, return_scores: bool = True):
        """End-to-end search"""
        # Encode queries and documents
        q_reps = self.encode_queries(queries, batch_size=batch_size, to_cpu=True)
        p_reps = self.encode_documents(documents, batch_size=batch_size, to_cpu=True)
        
        if return_scores:
            # Compute similarities
            print("Computing similarities...")
            scores = self.compute_similarity(q_reps, p_reps)
            return scores, q_reps, p_reps
        
        return q_reps, p_reps

    def rank_documents(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple]:
        """Rank documents"""
        scores, _, _ = self.search([query], documents, return_scores=True)
        scores = scores.squeeze(0)
        
        # Get top-k results
        top_indices = np.argsort(scores)[::-1][:min(top_k, len(documents))]
        
        results = []
        for idx in top_indices:
            results.append((int(idx), float(scores[idx]), documents[idx]))
        
        return results


# Helper Functions
def _split_into_batches(ids: np.ndarray, mask: np.ndarray, bsize: int):
    return [(ids[i:i + bsize], mask[i:i + bsize])
            for i in range(0, ids.shape[0], bsize)]

def _sort_by_length(ids: np.ndarray, mask: np.ndarray, bsize: int):
    if ids.shape[0] <= bsize:
        return ids, mask, np.arange(ids.shape[0])
    
    lengths = mask.sum(-1)
    order = np.argsort(lengths)
    reverse = np.argsort(order)
    return ids[order], mask[order], reverse

def _dynamic_augment(ids: List[int], mask_id: int, max_cap: int = None) -> List[int]:
    if max_cap is not None and len(ids) > max_cap:
        return ids[:max_cap]
    
    q_len = len(ids)
    target = max(32, ((q_len + 31) // 32) * 32)
    if target - q_len < 8:
        target = q_len + 8
    if max_cap is not None:
        target = min(target, max_cap)
    return ids + [mask_id] * (target - q_len)

def _stack_3D_arrays(groups):
    bsize = sum(x.shape[0] for x in groups)
    maxlen = max(x.shape[1] for x in groups)
    hdim = groups[0].shape[2]
    out = np.zeros((bsize, maxlen, hdim), dtype=groups[0].dtype)
    ptr = 0
    for g in groups:
        out[ptr:ptr + g.shape[0], :g.shape[1]] = g
        ptr += g.shape[0]
    return out