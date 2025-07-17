# Lateness - Modern ColBERT for Late Interaction

A simple-by-design python lib that works in two modes: 1.) allows you to do cheap and lightweight retrieval and 2.) heavy GPU accelerated indexing using ModernColBERT -  `prithivida/modern_colbert_base_en_v1`, (the 2nd best ColBERT in the world) into vectorDBs that offers native multi-vector support like Qdrant, Vespa and more..


## Why moderncolberts ?

- Top 2 ColBERTs in the world are moderncolberts
- They support long context, 8K.

| Dataset / Model | GTE-ModernColBERT<br/>(Lighton AI) |(Ours) | ColBERT-small<br/>(Answer AI, reproduced by Lighton) | jina-colbert-v2 | ColBERTv2.0 <br/> Stanford |
|:-----------------|:-----------------:|:-----------------:|:------------------------:|:---------------:|:------------:|
| **Outfit type**     | AI Lab with PhDs <br/>    | Indie Researcher, <br/> No PhD, No GPU budgets :-)      | AI Lab with PhDs                      | AI Lab with PhDs <br/>|  Academia with PhDs |
| **BEIR Average**     | **54.75** (🥇)   | **54.19** (🥈)       | 53.14                    | 52.30 | 49.48 |
| **FiQA2018**    | **48.51**         | 43.96             | 41.01                    | 40.8 | 35.6 |
| **NFCorpus**    | **37.93**         | 37.23             | 36.86                    | 34.6 | 33.8 |
| **TREC-COVID**  | 83.59             | 83.4             | 83.14                    | **83.4** | 73.3 |
| **Touche2020**  | **31.23**         | 29.32             | 24.95                    | 27.4 | 26.3 |
| **ArguAna**     | 48.51             | **52.05**         | 46.76                    | 36.6 | 46.3 |
| **QuoraRetrieval** | 86.61          | 87.54             | 87.89                | **88.7** | 85.2 |
| **SCIDOCS**     | 19.06             | **19.42**         | 18.72                    | 18.6 | 15.4 |
| **SciFact**     | 76.34             | **76.44**             | 74.02                    | 67.8 | 69.3 |
| **NQ**          | 61.8          | 61.68            | 59.42                    | 64.0 | 56.2 |
| **ClimateFEVER** | 30.62            | 28.29             | **32.83**                    | 23.9 | 17.6 |
| **HotpotQA**    | **77.32**         | 76.667             | 76.88                    | 76.6 | 66.7 |
| **DBPedia**     | **48.03**         | 46.31             | 46.36                    | 47.1 | 44.6 |
| **FEVER**       | 87.44             | 88.106             | **88.66**                    | 80.5 | 78.5 |



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
