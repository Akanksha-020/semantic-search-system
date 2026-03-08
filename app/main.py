from fastapi import FastAPI
from pydantic import BaseModel

from .semantic_cache import SemanticCache
from .search import run_search

import tarfile
import os

if not os.path.exists("data/mini_newsgroups"):
    with tarfile.open("data/mini_newsgroups.tar.gz") as tar:
        tar.extractall("data/")

app = FastAPI()

cache = SemanticCache()

class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_api(request: QueryRequest):

    lookup = cache.lookup(request.query)

    if lookup["hit"]:
        return {
            "query": request.query,
            "cache_hit": True,
            "matched_query": lookup["matched_query"],
            "similarity_score": lookup["similarity"],
            "result": lookup["result"],
            "dominant_cluster": lookup["cluster"]
        }

    result, cluster, embedding = run_search(request.query)

    cache.store(request.query, embedding, result, cluster)

    return {
        "query": request.query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": cluster
    }


@app.get("/cache/stats")
def cache_stats():
    return cache.stats()


@app.delete("/cache")
def clear_cache():
    cache.clear()

    return {"message": "Cache cleared"}
