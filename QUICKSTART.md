# Quick Start Guide

## Prerequisites
- Python 3.8+
- Virtual environment already activated at `venv`
- Dependencies already installed

## Step 1: Verify Installation

The dependencies should already be installed. If not:
```powershell
pip install -r requirements.txt
```

## Step 2: Test the System

Run the test script to verify everything works:
```powershell
python test_system.py
```

This will:
- Load 1971 documents from the local tar file
- Generate embeddings (takes ~2 minutes)
- Run clustering
- Test search functionality
- Display sample cluster analysis

## Step 3: Start the API Server

```powershell
uvicorn app.main:app --reload
```

The API will be available at: http://localhost:8000

Important: the first startup is slow because the app loads the dataset, loads the sentence-transformer model, and generates embeddings. Wait until Uvicorn prints that it is running on `http://127.0.0.1:8000` before using `/docs`.

### Try the Interactive Docs
Open http://localhost:8000/docs in your browser for Swagger UI

### Example API Calls

**Semantic Search:**
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "computer graphics rendering"}'
```

**List All Clusters:**
```bash
curl "http://localhost:8000/clusters"
```

**Get Boundary Cases:**
```bash
curl "http://localhost:8000/boundary-cases?min_entropy=2.0&limit=10"
```

## Step 4: Run Detailed Cluster Analysis

```powershell
python -m app.analyze_clusters
```

This will show:
- Statistical analysis of clustering
- Category distribution across clusters
- Most uncertain documents (boundary cases)
- Representative documents per cluster
- Cross-category semantic grouping

## Interactive Python Analysis

```python
from app.embeddings import clean_docs, documents_with_metadata
from app.clustering import (
    get_document_cluster_distribution,
    get_top_k_clusters,
    get_cluster_entropy,
    print_cluster_summary,
    n_clusters
)
from app.search import run_search

# Search for documents
result = run_search("artificial intelligence")
print(f"Found: {result['category']}")
print(f"Clusters: {result['top_clusters']}")
print(f"Entropy: {result['cluster_entropy']:.3f}")

# Analyze a specific document
doc_idx = 100
memberships = get_document_cluster_distribution(doc_idx)
print(f"Cluster memberships: {memberships}")

top_3 = get_top_k_clusters(doc_idx, k=3)
print(f"Top 3 clusters: {top_3}")

entropy = get_cluster_entropy(doc_idx)
print(f"Entropy: {entropy:.3f} bits")
if entropy >= 2.0:
    print("This is a boundary case!")

# Analyze a cluster
print_cluster_summary(cluster_id=5, num_examples=5)
```

## Understanding the Output

### Fuzzy Partition Coefficient (FPC)
- Range: [1/n_clusters, 1]
- Our FPC ≈ 0.09 (baseline would be 0.083 for 12 clusters)
- Interpretation: Memberships are fuzzy but meaningful

### Cluster Entropy
- Low (<1.5 bits): Document clearly belongs to one cluster
- Medium (1.5-2.5 bits): Spans 2-3 clusters
- High (>2.5 bits): Genuinely multi-topic (boundary case)

### Cluster Memberships
Each document has 12 membership values summing to 1.0:
```
[0.14, 0.07, 0.06, 0.09, 0.07, 0.08, 0.07, 0.08, 0.09, 0.07, 0.08, 0.09]
```
This document primarily belongs to cluster 0 (0.14) but also has some affinity with clusters 3, 8, 11.

## Common Tasks

### Find Documents Similar to a Query
```python
from app.search import run_search

result = run_search("quantum physics", k=5)
print(result['text'])
```

### Find All Boundary Cases
```python
from app.clustering import get_boundary_documents

boundary_docs = get_boundary_documents(min_entropy_threshold=2.5)
for doc_idx, entropy, top_clusters in boundary_docs[:10]:
    print(f"Doc {doc_idx}: entropy={entropy:.3f}, clusters={top_clusters}")
```

### Analyze Category Distribution in a Cluster
``` python
from app.clustering import analyze_cluster_categories

categories = analyze_cluster_categories(cluster_id=5)
for cat in categories:
    print(f"{cat['category']}: {cat['count']} docs, avg membership: {cat['avg_membership']:.3f}")
```

## Troubleshooting

### Clustering Takes Too Long
- Normal: ~5 seconds for clustering after embeddings are generated
- First run takes longer due to embedding generation (~2 minutes)
- Subsequent imports reuse cached embeddings

### Memory Issues
- System uses ~500MB RAM
- If running low on memory, reduce batch size in embeddings.py

### API Not Starting
- Make sure port 8000 is not in use
- Try: `uvicorn app.main:app --reload --port 8001`

### Windows: `[WinError 10013]` when starting Uvicorn
- This commonly means another process is already listening on the same port
- In this project, a previous `uvicorn app.main:app --host 127.0.0.1 --port 8000` instance may still be running in the background
- Check the port owner with: `Get-NetTCPConnection -LocalPort 8000 | Select-Object LocalAddress,LocalPort,State,OwningProcess`
- Inspect the process with: `Get-CimInstance Win32_Process -Filter "ProcessId = <PID>" | Select-Object ProcessId,CommandLine`
- If it is an old server you no longer need, stop it with: `Stop-Process -Id <PID>`
- Otherwise start this app on a different port, for example: `uvicorn app.main:app --reload --port 8001`

### Swagger UI Shows "Failed to fetch"
- Make sure the FastAPI server is actually running on `http://127.0.0.1:8000`
- If you started the app manually, wait for the initial embedding generation to finish before opening `/docs`
- If you are calling the API from another browser page, frontend app, or VS Code preview, that is a cross-origin request; this project now enables development CORS, but the server still must be reachable
- `filter_category` should be a full category name such as `talk.politics.guns`; a value like `gun` will not filter, though it should not by itself cause a fetch failure

## Next Steps

1. Explore cluster summaries: `print_cluster_summary(cluster_id)` for each of the 12 clusters
2. Investigate boundary cases: What makes them span multiple topics?
3. Compare cluster assignments vs original categories: Do they make semantic sense?
4. Test search with various queries to see cluster distributions

## Files Reference

- **embeddings.py**: Dataset loading, cleaning, embedding generation, FAISS index
- **clustering.py**: Fuzzy clustering, analysis functions
- **search.py**: Semantic search with cluster information
- **main.py**: FastAPI endpoints
- **analyze_clusters.py**: Comprehensive analysis script
- **test_system.py**: System verification

## Questions?

Check the detailed justifications in:
- Code comments (every major decision is documented)
- [README.md](README.md) for architecture overview
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for all changes
