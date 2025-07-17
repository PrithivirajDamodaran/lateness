# Lateness Local Testing Examples

This folder contains examples for **local testing and development** with Lateness using the **ONNX backend only**. These examples are designed for quick testing of Qdrant indexing and retrieval on your local machine.

> **⚠️ Important**: These examples use the lightweight ONNX backend for local testing. For production indexing with GPU acceleration, see the [Production Setup](#-production-setup) section below.

## 📋 Prerequisites

- Python 3.8+
- Docker (for running Qdrant locally)
- 4GB+ RAM recommended

## 🚀 Local Environment Setup

### Option 1: Python Virtual Environment

```bash
# Create virtual environment
python -m venv lateness-local
source lateness-local/bin/activate  # On Windows: lateness-local\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Lateness (ONNX backend only - for local testing)
pip install lateness

# Verify installation
python -c "from lateness import ModernColBERT; print('✅ Lateness installed successfully')"
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n lateness-local python=3.9
conda activate lateness-local

# Install Lateness (ONNX backend only - for local testing)
pip install lateness

# Verify installation
python -c "from lateness import ModernColBERT; print('✅ Lateness installed successfully')"
```


## 🔧 Local Qdrant Setup

### 1. Start Local Qdrant Server

```bash
# Make script executable (Linux/Mac)
chmod +x qdrant_manager.sh

# Start Qdrant server locally
./qdrant_manager.sh start

# Verify Qdrant is running
./qdrant_manager.sh status
```

**Expected output:**
```
✅ Qdrant is running on http://localhost:6333
✅ Health check: {"status":"ok"}
```

### 2. Test Environment

Run the comprehensive environment test:

```bash
python test_qdrant_setup.py
```

**Expected output:**
```
🔍 Testing Lateness Local Environment
=====================================
✅ Python version: 3.9.x
✅ Lateness package: 0.1.22
✅ Qdrant client: Connected to localhost:6333
✅ ONNX backend: Available
✅ Model download: Success
✅ All tests passed! Ready for local testing.
```

## 📁 Local Testing Examples

### `basic_usage.py` - Complete Local Example

This example demonstrates local indexing and retrieval using ONNX backend:

```bash
# Run local example (uses ONNX backend by default)
python basic_usage.py
```

**What it does:**
- ✅ Initializes ModernColBERT with ONNX backend
- ✅ Creates local Qdrant collection
- ✅ Indexes sample documents locally
- ✅ Performs similarity search
- ✅ Shows retrieval results

**Expected output:**
```
🚀 Using ONNX backend Using ONNX backend (default, for GPU accelerated indexing, install lateness[index] and set LATENESS_USE_TORCH=true)
🔄 Downloading model: prithivida/modern_colbert_base_en_v1
✅ ONNX ColBERT loaded with providers: ['CPUExecutionProvider']
📊 Creating local Qdrant collection...
🔍 Indexing 5 documents locally...
✅ Local indexing completed
🔎 Testing local retrieval...
🎯 Top results found locally
```

## 🧪 Local Testing Workflow

### Step-by-Step Local Testing

1. **Setup Environment**
   ```bash
   python -m venv lateness-local
   source lateness-local/bin/activate
   pip install lateness
   ```

2. **Start Local Services**
   ```bash
   ./qdrant_manager.sh start
   ```

3. **Verify Dependencies**
   ```bash
   python test_qdrant_setup.py
   ```

4. **Run Local Example**
   ```bash
   python basic_usage.py
   ```

5. **Clean Up**
   ```bash
   ./qdrant_manager.sh stop
   deactivate
   ```

## 🔧 Local Qdrant Management

### Qdrant Manager Commands

```bash
# Start local Qdrant
./qdrant_manager.sh start

# Check status
./qdrant_manager.sh status

# View logs
./qdrant_manager.sh logs

# Stop Qdrant
./qdrant_manager.sh stop

# Clean up (removes all data)
./qdrant_manager.sh clean
```

### Manual Docker Commands (Alternative)

```bash
# Start Qdrant locally
docker run -d -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant-local qdrant/qdrant

# Check health
curl http://localhost:6333/health

# Stop and remove
docker stop qdrant-local && docker rm qdrant-local
```


## 🚀 Production Setup

> **For production indexing and retrieval**, you need a different setup:

### Production Indexing (GPU Acceleration)

```bash
# Production environment setup
pip install lateness[index]  # Includes PyTorch + Transformers

# Set environment variable for PyTorch backend
export LATENESS_USE_TORCH=true

# Use production Qdrant cluster
# - Deploy Qdrant cluster (not local Docker)
# - Use GPU-enabled servers
# - Configure proper scaling and monitoring
```

### Production Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Indexing       │    │  Qdrant         │    │  Retrieval      │
│  Service        │    │  Cluster        │    │  Service        │
│                 │    │                 │    │                 │
│ lateness[index] │───▶│  Production     │◀───│  lateness       │
│ PyTorch + GPU   │    │  Multi-node     │    │  ONNX + CPU     │
│ Heavy Indexing  │    │  Persistent     │    │  Fast Queries   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

**Production Checklist:**
- ✅ Use `lateness[index]` for indexing services
- ✅ Set `LATENESS_USE_TORCH=true` for GPU acceleration
- ✅ Deploy Qdrant cluster (not local Docker)
- ✅ Use separate indexing and retrieval services
- ✅ Configure proper monitoring and scaling
- ✅ Use production-grade infrastructure

## 📝 Local vs Production

| Aspect | Local Testing | Production |
|--------|---------------|------------|
| **Installation** | `pip install lateness` | `pip install lateness[index]` |
| **Backend** | ONNX (CPU) | PyTorch (GPU) |
| **Qdrant** | Local Docker | Production Cluster |
| **Performance** | Basic testing | High-performance |
| **Use Case** | Development/Testing | Production workloads |

## 🔗 Next Steps

1. **Complete local testing** with these examples
2. **Understand the workflow** before production deployment
3. **Scale to production** using `lateness[index]` and proper infrastructure
4. **Monitor performance** and optimize based on your use case

## 📚 Additional Resources

- [Main Documentation](../README.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Production Deployment Guide](https://qdrant.tech/documentation/cloud/)

## 🤝 Need Help?

For local testing issues:
1. Run `python test_qdrant_setup.py` to diagnose problems
2. Check the troubleshooting section above
3. Ensure you're using the correct local setup (not production commands)

For production deployment:
1. Review the production setup section
2. Consider professional support for large-scale deployments
3. Test thoroughly in staging before production
