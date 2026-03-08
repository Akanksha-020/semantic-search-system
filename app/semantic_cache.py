import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:
    """
    Semantic cache for query results.
    
    Uses cosine similarity to match semantically similar queries
    instead of exact string matching.
    """

    def __init__(self, threshold=0.85):
        self.cache = []
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def lookup(self, query):
        """
        Look up a query in the cache.
        
        Returns cached result if a semantically similar query exists.
        """
        query_vec = self.model.encode([query])

        for entry in self.cache:
            sim = cosine_similarity(query_vec, entry["embedding"])[0][0]

            if sim >= self.threshold:
                self.hit_count += 1

                return {
                    "hit": True,
                    "matched_query": entry["query"],
                    "similarity": float(sim),
                    "result": entry["result"]
                }

        self.miss_count += 1
        return {"hit": False, "embedding": query_vec}

    def store(self, query, embedding, result):
        """
        Store a query and its result in the cache.
        
        Args:
            query: query string
            embedding: query embedding
            result: result data (dict or any serializable object)
        """
        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):
        """Get cache statistics"""
        total = len(self.cache)
        hits = self.hit_count
        misses = self.miss_count

        rate = hits / (hits + misses) if (hits + misses) else 0

        return {
            "total_entries": total,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": rate
        }

    def clear(self):
        """Clear all cache entries and statistics"""
        self.cache = []
        self.hit_count = 0
        self.miss_count = 0