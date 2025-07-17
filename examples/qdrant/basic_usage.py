"""
Basic Usage Example for Lateness - Modern ColBERT Package
=========================================================

This example demonstrates the core functionality of the lateness package:
1. One-line installation: pip install lateness
2. Simple import and usage
3. Modern ColBERT encoding
4. Qdrant indexing and retrieval

"""

from lateness import ModernColBERT, QdrantIndexer, QdrantRetriever
from qdrant_client import QdrantClient


def main():
    print("🚀 Lateness - Modern ColBERT Basic Usage Example")
    print("=" * 50)
    
    # Step 1: Initialize Modern ColBERT (auto-downloads model)
    print("📥 Loading Modern ColBERT model...")
    colbert = ModernColBERT("prithivida/modern_colbert_base_en_v1")
    print("✅ Modern ColBERT loaded successfully!")
    
    # Step 2: Prepare sample documents
    documents = [
        "PyTorch is an open-source machine learning framework that provides tensor computations with GPU acceleration and deep neural networks built on tape-based autograd system.",
        "Kubernetes is a container orchestration platform that automates deployment, scaling, and management of containerized applications across clusters of machines.",
        "REST APIs follow representational state transfer architectural style using HTTP methods like GET, POST, PUT, DELETE for stateless client-server communication.",
        "Docker containers package applications with their dependencies into lightweight, portable units that can run consistently across different computing environments.",
        "PostgreSQL is an advanced open-source relational database system known for extensibility, SQL compliance, and support for JSON, arrays, and custom data types.",
        "Redis is an in-memory data structure store used as database, cache, and message broker supporting strings, hashes, lists, sets, and sorted sets.",
        "Microservices architecture decomposes applications into small, independent services that communicate via APIs, enabling scalability and technology diversity.",
        "GraphQL is a query language and runtime for APIs that allows clients to request exactly the data they need in a single request with strong type system.",
        "Apache Kafka is a distributed event streaming platform capable of handling trillions of events per day for real-time data pipelines and streaming applications.",
        "React is a JavaScript library for building user interfaces using component-based architecture with virtual DOM for efficient rendering and state management.",
        "Machine learning pipelines automate the workflow of data preprocessing, feature engineering, model training, validation, and deployment for ML systems.",
        "Vector databases store and retrieve high-dimensional embeddings for similarity search, recommendation systems, and semantic search applications.",
        "NGINX is a high-performance web server and reverse proxy that handles HTTP requests, load balancing, SSL termination, and serves static content efficiently.",
        "Elasticsearch is a distributed search and analytics engine built on Apache Lucene for full-text search, structured search, and analytics at scale.",
        "JWT tokens provide secure information transmission between parties as compact, URL-safe JSON objects that are digitally signed using HMAC or RSA algorithms."
    ]
    
    print(f"📚 Prepared {len(documents)} sample documents")
    
    # Step 3: Setup Qdrant client and indexer
    print("🔧 Setting up Qdrant connection...")
    client = QdrantClient("localhost", port=6333)  # Assumes Qdrant is running locally
    
    collection_name = "lateness_demo"
    indexer = QdrantIndexer(client, collection_name)
    retriever = QdrantRetriever(client, collection_name)
    
    # Step 4: Create collection and index documents
    print("📊 Creating Qdrant collection...")
    indexer.create_collection()
    
    print("🔍 Indexing documents with Modern ColBERT...")
    # Convert documents to corpus format
    corpus = {}
    for i, doc in enumerate(documents):
        corpus[f"doc_{i}"] = {"title": "", "text": doc}
    
    indexer.index_documents(corpus, colbert, batch_size=4)
    print("✅ Documents indexed successfully!")
    
    # Step 5: Perform searches
    print("\n🔎 Performing searches...")
    queries = [
        "How to deploy containers at scale?",
        "What is the best database for JSON data?", 
        "How to build real-time data pipelines?",
        "What are the benefits of microservices?",
        "How to implement efficient web APIs?"
    ]
    
    for query in queries:
        print(f"\n📝 Query: '{query}'")
        results = retriever.search_with_query(query, colbert, top_k=3)
        
        print("🎯 Top results:")
        for i, (doc_id, score) in enumerate(results):
            doc_idx = int(doc_id.split('_')[1])
            print(f"  {i+1}. Score: {score:.4f}")
            print(f"     Document: {documents[doc_idx][:100]}...")
    
    print("\n🎉 Demo completed successfully!")
    print("\n💡 Key Features Demonstrated:")
    

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Make sure:")
        print("   1. Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        print("   2. Package is installed: pip install lateness")
        print("   3. You have internet connection for model download")
