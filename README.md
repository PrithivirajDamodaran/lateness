# lateness

A simple-by-design python lib that works in two modes: 1.) allows you to do cheap and lightweight retrieval and 2.) heavy GPU accelerated indexing using ModernColBERT -  `prithivida/modern_colbert_base_en_v1`, (the 2nd best ColBERT in the world) into vectorDBs that offers native multi-vector support like Qdrant, Vespa and more..


## Why Modern-Colbert Models ?

- They are based on ModernBERT and efficient.
- Top 2 ColBERTs in the world are ModernBERT based.
- They support long context, 8K.


| Dataset / Model | GTE-ModernColBERT<br/>(Lighton AI) | <span style="background-color: #a4f9d9; padding: 4px; border-radius: 4px;">(Ours)</span> | ColBERT-small<br/>(Answer AI, reproduced by Lighton) | jina-colbert-v2 | ColBERTv2.0<br/>Stanford |
|:-----------------|:-----------------:|:-----------------:|:------------------------:|:---------------:|:------------:|
| **BEIR Average**     | **54.75** (🥇)   | <span style="background-color:#a4f9d9; padding: 4px; border-radius: 4px;">**54.19** (🥈)</span>       | 53.14                    | 52.30 | 49.48 |

PS: Jina and Stanford did not run eval on CQADupstack and MSMARCO hence we skipped to make it fair.

## Features

- **Dual Backend Architecture**: ONNX for fast retrieval, PyTorch for GPU indexing
- **Native Multi-Vector Support**: Optimized for Qdrant's MaxSim comparator
- **Smart Installation**: Lightweight retrieval or heavy indexing based on your needs
- **Production Ready**: Separate deployment targets for different workloads

## Quick Start

### Installation

```bash
# Lightweight retrieval (ONNX + Qdrant)
pip install lateness

# Heavy indexing (PyTorch + Transformers + ONNX + Qdrant)
pip install lateness[index]
```

### Backend Selection

### Basic Usage

**Default Installation (ONNX Backend):**
```python

# pip install lateness
from lateness import ModernColBERT
colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
# Output:
# 🚀 Using ONNX backend Using ONNX backend (default, for GPU accelerated indexing, install lateness[index] and set LATENESS_USE_TORCH=true)
# 🔄 Downloading model: prithivida/modern_colbert_base_en_v1
# ✅ ONNX ColBERT loaded with providers: ['CPUExecutionProvider']
# Query max length: 256, Document max length: 300
```

```python
from lateness import ModernColBERT
colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")

documents = [
    "PyTorch is an open-source machine learning framework that provides tensor computations with GPU acceleration and deep neural networks built on tape-based autograd system.",
    "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications across clusters of machines.",
    "REST APIs follow representational state transfer architectural style using HTTP methods like GET, POST, PUT, DELETE for stateless client-server communication.",
]

queries = [
    "How to build real-time data pipelines?",
    "What are the benefits of microservices?",
    "How to implement efficient web APIs?"
]



query_embeddings = colbert.encode_queries(queries)
doc_embeddings = colbert.encode_documents(documents)
scores = ModernColBERT.compute_similarity(query_embeddings, doc_embeddings)
print(scores)
```


**Index Installation (PyTorch Backend):**
```python
# pip install lateness[index]
import os
os.environ['LATENESS_USE_TORCH'] = 'true'
from lateness import ModernColBERT

colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
# Output:
# 🚀 Using PyTorch backend (LATENESS_USE_TORCH=true)
# 🔄 Downloading model: prithivida/modern_colbert_base_en_v1
# Loading model from: /root/.cache/huggingface/hub/models--prithivida--modern_colbert_base_en_v1/...
# ✅ PyTorch ColBERT loaded on cuda
# Query max length: 256, Document max length: 300
```

**Complete Example with Qdrant:**

For a complete working example with Qdrant integration, environment setup, and testing instructions, see the [examples/qdrant folder](./examples/qdrant/).

The examples include:
- Environment setup and testing
- Local Qdrant server management
- Complete indexing and retrieval workflows
- Both ONNX and PyTorch backend examples

## Architecture

### Two Deployment Models

**Retrieval Service (Lightweight)**
```bash
pip install lateness
```
- ONNX backend (fast CPU inference)
- Qdrant integration
- ~50MB total dependencies
- Perfect for user-facing search APIs

**Indexing Service (Heavy)**
```bash
pip install lateness[index]
```
- PyTorch backend (GPU acceleration)
- Full Transformers support
- ~2GB+ dependencies
- Perfect for batch document processing

### Backend Selection

The package uses environment variables for backend control:

- **Default behavior** → ONNX backend (CPU retrieval)
- **`LATENESS_USE_TORCH=true`** → PyTorch backend (GPU indexing)

**Note:** PyTorch backend requires `pip install lateness[index]` to install PyTorch dependencies.


## License

Apache License 2.0

## Contributing

Contributions welcome! Please check our [contributing guidelines](CONTRIBUTING.md).
