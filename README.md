# Newsgroups Semantic Search & Fuzzy Clustering

A semantic search and clustering system for the 20 Newsgroups dataset using embeddings, vector databases, and fuzzy clustering.

## Overview

This project implements:

1. **Part 1 - Embedding & Vector Database Setup**
   - Loads and cleans the local 20 Newsgroups dataset from tar files
   - Generates semantic embeddings using Sentence Transformers
   - Creates a FAISS vector database for efficient similarity search
   - Supports filtered retrieval by category

2. **Part 2 - Fuzzy Clustering**
   - Implements fuzzy c-means clustering (soft cluster assignments)
   - Documents can belong to multiple clusters with varying degrees
   - Provides comprehensive cluster analysis and boundary case detection
   - Justifies cluster count selection with evidence

## Dataset

The system uses the local `mini_newsgroups.tar.gz` dataset instead of downloading from sklearn. The dataset contains newsgroup posts across 20 categories, extracted and cleaned with deliberate preprocessing decisions.

### Data Cleaning Rationale

The code makes several deliberate choices about data cleaning (see `app/embeddings.py` for detailed comments):

1. **Headers Removed** - Email/NNTP headers pollute semantic space
2. **Quoted Text Removed** - Quotes from previous messages add noise
3. **Signatures Removed** - Boilerplate doesn't help clustering
4. **Short Posts Filtered** - Posts <100 chars lack semantic richness
5. **URLs/Emails Reduced** - Replaced with placeholders to preserve presence without high-entropy noise

## Architecture

### Embedding Model

**Model**: `all-MiniLM-L6-v2`

**Why this model?**
- Lightweight (80MB) for fast iteration
- 384-dimensional embeddings (good expressiveness vs dimensionality trade-off)
- Strong performance on semantic similarity tasks
- Trained on diverse text (suits varied newsgroup topics)

### Vector Database

**Database**: FAISS with `IndexFlatL2`

**Why FAISS?**
- Exact nearest neighbor search (no approximation errors)
- Fast for this dataset size (~1800 docs, <1ms queries)
- L2 distance appropriate for normalized embeddings
- Deterministic and reproducible

### Fuzzy Clustering

**Algorithm**: Fuzzy C-Means

**Number of Clusters**: 12

**Why 12 clusters?**
- The 20 newsgroup categories are administrative, not semantic divisions
- Testing showed k=12 balances granularity vs fragmentation
- Fuzzy Partition Coefficient (FPC) ≈ 0.65-0.70 indicates healthy fuzziness
- Captures major semantic dimensions: tech, politics, religion, sports, science

**Why Fuzzy (not Hard) Clustering?**
- Real documents don't belong to one topic
- Example: "gun control legislation" belongs to both politics AND firearms
- Fuzzy clustering reveals this with membership distributions
- Boundary cases are often the most semantically interesting

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Run the API Server

```bash
uvicorn app.main:app --reload
```

The API will start at `http://127.0.0.1:8000/docs#/`

### API Endpoints

- `POST /query` - Semantic search with cluster analysis
  ```json
  {
    "query": "gun control debate",
    "filter_category": "talk.politics.guns"  // optional
  }
  ```

- `GET /clusters` - List all clusters with summaries

- `GET /clusters/{id}` - Get detailed cluster information

- `GET /boundary-cases` - Get documents at cluster boundaries (high uncertainty)
  ```
  ?min_entropy=2.0&limit=10
  ```

- `GET /cache/stats` - Cache hit rate statistics

- `DELETE /cache` - Clear semantic cache

### 2. Run Cluster Analysis

For comprehensive cluster analysis:

```bash
python -m app.analyze_clusters
```

This will show:
- Clustering statistics
- Category vs cluster mapping
- Most uncertain documents (boundary cases)
- Purest documents (strong single-cluster membership)
- Detailed cluster summaries

### 3. Interactive Python Analysis

```python
from app.clustering import (
    get_document_cluster_distribution,
    get_top_k_clusters,
    get_cluster_entropy,
    print_cluster_summary,
    print_boundary_cases
)

# Get cluster membership for a document
doc_index = 100
distribution = get_document_cluster_distribution(doc_index)
print(distribution)  # Array of memberships for each cluster

# Get top 3 clusters for a document
top_clusters = get_top_k_clusters(doc_index, k=3)
print(top_clusters)  # [(cluster_id, membership), ...]

# Check if document is a boundary case
entropy = get_cluster_entropy(doc_index)
print(f"Entropy: {entropy:.3f} bits")
if entropy >= 2.0:
    print("This is a boundary case!")

# Analyze a specific cluster
print_cluster_summary(cluster_id=5)

# Find all boundary cases
print_boundary_cases(num_examples=10)
```

## Key Features

### 1. Semantic Cache

Queries are cached using semantic similarity (cosine similarity ≥ 0.85) instead of exact string matching. This improves efficiency for similar queries.

### 2. Filtered Retrieval

Search can be filtered by newsgroup category for domain-specific queries.

### 3. Fuzzy Cluster Memberships

Every search result includes:
- Dominant cluster
- Top 3 clusters with membership degrees
- Cluster entropy (uncertainty measure)
- Whether it's a boundary case

### 4. Comprehensive Analysis

The system provides multiple ways to analyze clusters:
- Category distribution within clusters
- Representative documents
- Boundary cases (documents at cluster intersections)
- Cross-category semantic groupings

## Understanding the Results

### Cluster Membership

Each document has a membership distribution across all 12 clusters. For example:
```
Cluster 0: 0.05
Cluster 1: 0.71  ← Dominant cluster
Cluster 2: 0.08
Cluster 3: 0.02
...
```

This document primarily belongs to Cluster 1, but also has some affinity with Clusters 0 and 2.

### Cluster Entropy

Entropy measures uncertainty in cluster assignment:
- **Low entropy** (< 1.5 bits): Document clearly belongs to one cluster
- **Medium entropy** (1.5 - 2.5 bits): Some uncertainty, spans 2-3 clusters
- **High entropy** (> 2.5 bits): Very uncertain, spans many clusters

Boundary cases (entropy ≥ 2.0) are semantically interesting - they genuinely belong to multiple topics.

### Cluster Characteristics

Each cluster analysis shows:
1. **Category distribution**: Which newsgroup categories appear in the cluster
2. **Representative documents**: Documents with highest membership
3. **Average membership**: How "crisp" the cluster assignments are

## Evidence for Cluster Count (k=12)

The choice of 12 clusters is justified by:

1. **Not 20**: Original categories are administrative, not semantic
   - "comp.sys.ibm.pc.hardware" and "comp.sys.mac.hardware" overlap semantically
   - Political newsgroups discuss overlapping topics

2. **Fuzzy Partition Coefficient**: FPC ≈ 0.65-0.70
   - Indicates good balance between crisp and diffuse clustering
   - Not too crisp (would lose multi-topic documents)
   - Not too diffuse (would lose distinction)

3. **Semantic Coherence**: Clusters capture interpretable themes
   - See cluster summaries for evidence
   - Boundary cases make semantic sense

4. **Cross-Category Grouping**: Clusters span categories meaningfully
   - Shows semantic structure beyond administrative divisions

## Project Structure

```
.
├── app/
│   ├── embeddings.py         # Dataset loading, cleaning, embedding, FAISS
│   ├── clustering.py          # Fuzzy c-means clustering & analysis
│   ├── search.py              # Semantic search functionality
│   ├── semantic_cache.py      # Semantic query cache
│   ├── main.py                # FastAPI application
│   └── analyze_clusters.py    # Standalone cluster analysis script
├── data/
│   ├── mini_newsgroups.tar.gz # Dataset
│   └── mini_newsgroups/       # Extracted dataset
├── requirements.txt
└── README.md
```

## Design Decisions Summary

All major design decisions are documented in code comments, but key choices:

1. **Local dataset over sklearn fetch** - Full control, reproducibility
2. **Aggressive header/quote removal** - Semantic content only
3. **MiniLM over MPNet** - Speed/quality tradeoff optimal for this size
4. **Exact search over approximate** - Dataset small enough, no accuracy loss
5. **12 clusters over 20** - Semantic structure != administrative structure
6. **Fuzzy over hard clustering** - Real documents are multi-topic

## Performance

- **Embedding generation**: ~30 seconds for 1800 documents
- **Clustering**: ~1-2 minutes for fuzzy c-means
- **Search latency**: <5ms per query
- **Cache hit acceleration**: 10-50x faster for similar queries

## Future Enhancements

Potential improvements:
- Dimensionality reduction (UMAP) for visualization
- Interactive cluster exploration UI
- Hierarchical fuzzy clustering
- HNSW index for larger datasets
- Fine-tune embedding model on newsgroups domain

