from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from .semantic_cache import SemanticCache
from .search import run_search
from .clustering import (
    print_cluster_summary,
    get_boundary_documents,
    analyze_cluster_categories,
    get_representative_documents,
    n_clusters
)
from .embeddings import get_document_metadata, clean_docs

import tarfile
import os

# Extract dataset if needed (handled in embeddings.py too, but keeping for safety)
if not os.path.exists("data/mini_newsgroups"):
    tar_path = "data/mini_newsgroups.tar.gz"
    if os.path.exists(tar_path):
        print("Extracting mini_newsgroups.tar.gz...")
        with tarfile.open(tar_path) as tar:
            tar.extractall("data/")

app = FastAPI(
    title="Newsgroups Semantic Search & Clustering API",
    description="Semantic search with fuzzy clustering over 20 Newsgroups dataset"
)

cache = SemanticCache()


class QueryRequest(BaseModel):
    query: str
    filter_category: Optional[str] = None


@app.post("/query")
def query_api(request: QueryRequest):
    """
    Semantic search with cluster analysis.
    
    Returns the most similar document with its fuzzy cluster memberships.
    """
    # Check cache
    lookup = cache.lookup(request.query)
    
    if lookup["hit"]:
        return {
            "query": request.query,
            "cache_hit": True,
            "matched_query": lookup["matched_query"],
            "similarity_score": lookup["similarity"],
            "result": lookup["result"]
        }
    
    # Run search
    result_data = run_search(request.query, filter_category=request.filter_category)
    
    # Store in cache
    cache.store(
        request.query,
        result_data['embedding'],
        result_data
    )
    
    # Format response
    response = {
        "query": request.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": {
            "text": result_data['text'][:500] + "..." if len(result_data['text']) > 500 else result_data['text'],
            "category": result_data['category'],
            "file_id": result_data['file_id'],
            "dominant_cluster": result_data['dominant_cluster'],
            "top_clusters": result_data['top_clusters'],
            "cluster_entropy": result_data['cluster_entropy'],
            "is_boundary_case": result_data['cluster_entropy'] >= 2.0
        }
    }
    
    return response


@app.get("/clusters")
def list_clusters():
    """Get information about all clusters"""
    cluster_info = []
    
    for cluster_id in range(n_clusters):
        categories = analyze_cluster_categories(cluster_id)
        representatives = get_representative_documents(cluster_id, top_n=3)
        
        cluster_info.append({
            "cluster_id": cluster_id,
            "top_categories": categories[:3],
            "representative_docs": [
                {
                    "doc_index": doc_idx,
                    "membership": membership,
                    "preview": clean_docs[doc_idx][:100] + "..."
                }
                for doc_idx, membership in representatives
            ]
        })
    
    return {"clusters": cluster_info, "total_clusters": n_clusters}


@app.get("/clusters/{cluster_id}")
def get_cluster(cluster_id: int):
    """Get detailed information about a specific cluster"""
    if cluster_id < 0 or cluster_id >= n_clusters:
        raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found. Valid range: 0-{n_clusters-1}")
    
    categories = analyze_cluster_categories(cluster_id)
    representatives = get_representative_documents(cluster_id, top_n=5)
    
    return {
        "cluster_id": cluster_id,
        "categories": categories,
        "representative_documents": [
            {
                "doc_index": doc_idx,
                "membership": membership,
                "text": clean_docs[doc_idx][:300] + "...",
                "metadata": get_document_metadata(doc_idx)
            }
            for doc_idx, membership in representatives
        ]
    }


@app.get("/boundary-cases")
def get_boundary_cases(min_entropy: float = 2.0, limit: int = 10):
    """
    Get documents that sit at cluster boundaries (high uncertainty).
    
    These are semantically interesting - they genuinely belong to multiple topics.
    """
    boundary_docs = get_boundary_documents(min_entropy_threshold=min_entropy)[:limit]
    
    result = []
    for doc_idx, entropy, top_clusters in boundary_docs:
        metadata = get_document_metadata(doc_idx)
        result.append({
            "doc_index": doc_idx,
            "entropy": entropy,
            "top_clusters": top_clusters,
            "category": metadata['category'],
            "text_preview": clean_docs[doc_idx][:200] + "..."
        })
    
    return {
        "boundary_cases": result,
        "total_found": len(boundary_docs)
    }


@app.get("/cache/stats")
def cache_stats():
    """Get cache statistics"""
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    """Clear the semantic cache"""
    cache.clear()
    return {"message": "Cache cleared"}


@app.get("/")
def root():
    """API info"""
    return {
        "message": "Newsgroups Semantic Search & Fuzzy Clustering API",
        "endpoints": {
            "POST /query": "Semantic search with cluster analysis",
            "GET /clusters": "List all clusters",
            "GET /clusters/{id}": "Get cluster details",
            "GET /boundary-cases": "Get documents at cluster boundaries",
            "GET /cache/stats": "Cache statistics",
            "DELETE /cache": "Clear cache"
        }
    }