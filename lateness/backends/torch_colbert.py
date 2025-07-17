"""
PyTorch Modern ColBERT Implementation
====================================

Heavy backend for GPU indexing workloads.
Only available with: pip install lateness[index]
"""

import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from tqdm import tqdm
from typing import List, Tuple, Union, Optional
import string
import os
from ..models.model_manager import download_model_from_hf


class TaggingHead(nn.Module):
    def __init__(self, input_size, num_labels):
        super().__init__()
        self.classifier = nn.Linear(input_size, num_labels, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, x):
        return self.classifier(x)


class ColBERT(PreTrainedModel):
    config_class = AutoConfig
    base_model_prefix = "backbone"
    
    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        hidden_dim = config.hidden_size
        self.heads = nn.ModuleDict({
            "col_pooling": TaggingHead(hidden_dim, num_labels=128)
        })
        
        # Inference settings (will be set when loading for inference)
        self.tokenizer = None
        self.max_query_len = 256
        self.max_doc_len = 300
        self.Q_PID = None
        self.D_PID = None
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask=None, position_ids=None, return_dict=False, **kwargs):
        kwargs.pop("token_type_ids", None)
        
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
            **kwargs
        )
        
        reps = outputs.last_hidden_state
        reps = torch.nn.functional.normalize(reps, p=2, dim=2)
        reps *= attention_mask[:, :, None].float()
        logits = self.heads["col_pooling"](reps)
        
        if return_dict:
            return BaseModelOutput(last_hidden_state=logits)
        return logits
    
    @classmethod
    def load_for_inference(cls, model_name_or_path: str, max_query_len: int = 256, 
                          max_doc_len: int = 300, device: str = None):
        """Load ColBERT model with tokenizer for inference"""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Download model if needed
        if not os.path.exists(model_name_or_path):
            print(f"🔄 Downloading model: {model_name_or_path}")
            model_name_or_path, _ = download_model_from_hf(model_name_or_path)
        
        try:
            # Load model and tokenizer
            print(f"Loading model from: {model_name_or_path}")
            config = AutoConfig.from_pretrained(model_name_or_path)
            model = cls.from_pretrained(model_name_or_path, config=config)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Setup inference configuration
            model.tokenizer = tokenizer
            model.max_query_len = max_query_len
            model.max_doc_len = max_doc_len
            model.Q_PID = tokenizer.convert_tokens_to_ids("[unused0]")
            model.D_PID = tokenizer.convert_tokens_to_ids("[unused1]")
            
            # Setup post-tokenization punctuation masking
            model.skip_ids = {tokenizer.encode(c, add_special_tokens=False)[0]
                             for c in string.punctuation}
            
            model.to(device)
            model.eval()
            
            print(f"✅ PyTorch ColBERT loaded on {device}")
            print(f"Query max length: {max_query_len}, Document max length: {max_doc_len}")
            
            return model
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _encode_batch(self, ids: torch.Tensor, mask: torch.Tensor, to_cpu: bool = False):
        """Internal encoding function"""
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded for inference. Use ColBERT.load_for_inference()")
        
        ids, mask = ids.to(self.device), mask.to(self.device)
        pos = torch.arange(ids.size(1), device=self.device).unsqueeze(0).expand_as(ids)
        
        with torch.no_grad():
            rep = self(input_ids=ids, attention_mask=mask, position_ids=pos)
        
        return rep.cpu() if to_cpu else rep
    
    def encode_queries(self, queries: List[str], batch_size: Optional[int] = None, to_cpu: bool = False):
        """Encode queries for ColBERT retrieval"""
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded for inference. Use ColBERT.load_for_inference()")
        
        print(f"Encoding {len(queries)} queries...")
        
        # Tokenize with query prefix
        enc = self.tokenizer(queries, add_special_tokens=True, truncation=False)
        id_lists = [[self.Q_PID] + ids for ids in enc["input_ids"]]
        
        # Apply dynamic augmentation with length cap
        cap = self.max_query_len or (self.tokenizer.model_max_length - 1)
        id_lists = [_dynamic_augment(ids, self.tokenizer.mask_token_id, cap) for ids in id_lists]
        
        # Pad sequences
        padded = self.tokenizer.pad({"input_ids": id_lists}, padding=True, return_tensors="pt")
        ids, mask = padded["input_ids"], padded["attention_mask"]
        
        # Process in batches if specified
        if batch_size:
            reps = []
            for i, a in tqdm(_split_into_batches(ids, mask, batch_size), desc="Encoding query batches"):
                reps.append(self._encode_batch(i, a, to_cpu))
            return torch.cat(reps)
        
        return self._encode_batch(ids, mask, to_cpu)
    
    def encode_documents(self, documents: List[str], batch_size: Optional[int] = None, 
                        keep_dims: bool = True, to_cpu: bool = False):
        """Encode documents for ColBERT retrieval with post-tokenization punctuation masking"""
        if self.tokenizer is None:
            raise RuntimeError("Model not loaded for inference. Use ColBERT.load_for_inference()")
        
        print(f"Encoding {len(documents)} documents...")
        
        # Tokenize documents WITHOUT removing punctuation (post-tokenization masking)
        enc = self.tokenizer(documents, add_special_tokens=True, 
                           truncation=True, max_length=self.max_doc_len - 1)
        id_lists = [[self.D_PID] + ids for ids in enc["input_ids"]]
        
        # Pad sequences
        padded = self.tokenizer.pad({"input_ids": id_lists}, padding=True, return_tensors="pt")
        ids, mask = padded["input_ids"], padded["attention_mask"]
        
        # Apply post-tokenization punctuation masking
        mask[torch.isin(ids, torch.tensor(list(self.skip_ids), device=ids.device))] = 0
        
        # Process in batches if specified
        if batch_size:
            ids_s, mask_s, rev = _sort_by_length(ids, mask, batch_size)
            reps = []
            
            for i, a in tqdm(_split_into_batches(ids_s, mask_s, batch_size), desc="Encoding document batches"):
                rep = self._encode_batch(i, a, to_cpu)
                if not keep_dims:
                    # Convert to list of variable-length tensors
                    m = a.cpu().bool() if to_cpu else a.bool()
                    rep = [r[m[idx]] for idx, r in enumerate(rep)]
                reps.append(rep)
            
            if keep_dims:
                return _stack_3D_tensors(reps)[rev]
            else:
                # Flatten and reorder
                flat = [d for g in reps for d in g]
                return [flat[i] for i in rev.tolist()]
        
        # Single batch processing
        rep = self._encode_batch(ids, mask, to_cpu)
        if not keep_dims:
            m = mask.cpu().bool() if to_cpu else mask.bool()
            rep = [r[m[idx]] for idx, r in enumerate(rep)]
        
        return rep
    
    @staticmethod
    def compute_similarity(q_reps: torch.Tensor, p_reps: torch.Tensor):
        """Compute ColBERT-style max similarity between queries and passages"""
        token_scores = torch.einsum("qin,pjn->qipj", q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1)
        return scores
    
    def search(self, queries: List[str], documents: List[str], 
               batch_size: Optional[int] = None, return_scores: bool = True):
        """End-to-end search: encode queries and documents, compute similarities"""
        # Encode queries and documents
        q_reps = self.encode_queries(queries, batch_size=batch_size, to_cpu=True)
        p_reps = self.encode_documents(documents, batch_size=batch_size, to_cpu=True)
        
        if return_scores:
            # Compute similarities
            print("Computing similarities...")
            scores = self.compute_similarity(q_reps, p_reps)
            return scores, q_reps, p_reps
        
        return q_reps, p_reps
    
    def rank_documents(self, query: str, documents: List[str], top_k: int = 10):
        """Rank documents for a single query"""
        scores, _, _ = self.search([query], documents, return_scores=True)
        scores = scores.squeeze(0)  # Remove query dimension
        
        # Get top-k results
        top_indices = torch.topk(scores, min(top_k, len(documents))).indices
        
        results = []
        for idx in top_indices:
            results.append((idx.item(), scores[idx].item(), documents[idx.item()]))
        
        return results


# Wrapper class for consistent API
class ModernColBERT:
    """Modern ColBERT PyTorch Backend (Heavy Indexing)"""
    
    def __init__(self, model_name: str, cache_dir: Optional[str] = None,
                 max_query_len: int = 256, max_doc_len: int = 300,
                 device: Optional[str] = None):
        self._model = ColBERT.load_for_inference(
            model_name, max_query_len, max_doc_len, device
        )
        self.backend_type = "torch"
    
    def encode_queries(self, queries: List[str], **kwargs):
        return self._model.encode_queries(queries, **kwargs)
    
    def encode_documents(self, documents: List[str], **kwargs):
        return self._model.encode_documents(documents, **kwargs)
    
    @staticmethod
    def compute_similarity(q_reps, p_reps):
        return ColBERT.compute_similarity(q_reps, p_reps)
    
    def search(self, queries: List[str], documents: List[str], **kwargs):
        return self._model.search(queries, documents, **kwargs)
    
    def rank_documents(self, query: str, documents: List[str], top_k: int = 10):
        return self._model.rank_documents(query, documents, top_k)


# Helper Functions
def _split_into_batches(ids: torch.Tensor, mask: torch.Tensor, bsize: int):
    return [(ids[i:i + bsize], mask[i:i + bsize])
            for i in range(0, ids.size(0), bsize)]

def _sort_by_length(ids: torch.Tensor, mask: torch.Tensor, bsize: int):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))
    
    lengths = mask.sum(-1)
    order = lengths.sort().indices
    reverse = order.sort().indices
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

def _stack_3D_tensors(groups):
    bsize = sum(x.size(0) for x in groups)
    maxlen = max(x.size(1) for x in groups)
    hdim = groups[0].size(2)
    out = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)
    ptr = 0
    for g in groups:
        out[ptr:ptr + g.size(0), :g.size(1)] = g
        ptr += g.size(0)
    return out