from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Base dependencies (ONNX + Qdrant - lightweight retrieval)
base_requirements = [
    "numpy>=1.21.0",
    "onnxruntime>=1.15.0",
    "tokenizers>=0.13.0",
    "huggingface-hub>=0.16.0",
    "tqdm>=4.64.0",
    "qdrant-client>=1.6.0",  # Vector DB is essential
]

# Heavy indexing dependencies (PyTorch + Transformers for GPU indexing)
index_requirements = [
    "torch>=1.13.0",
    "transformers>=4.21.0",
] + base_requirements

setup(
    name="lateness",
    version="0.1.0",
    author="donkey stereotype",
    author_email="prithivi@donkeystereotype.com",
    description="Modern ColBERT for Late Interaction with native multi-vector support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moderncolbert/lateness",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require={
        "index": index_requirements,  # PyTorch + Transformers for GPU indexing
        "all": index_requirements,    # Same as index (everything included)
    },
    keywords="colbert, retrieval, embeddings, information-retrieval, nlp, onnx, pytorch, qdrant",
    project_urls={
        "Bug Reports": "https://github.com/moderncolbert/lateness/issues",
        "Source": "https://github.com/moderncolbert/lateness",
        "Documentation": "https://moderncolbert.github.io/lateness/",
    },
)