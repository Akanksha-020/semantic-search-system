import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticCache:

    def __init__(self, threshold=0.85):
        self.cache = []
        self.threshold = threshold
        self.hit_count = 0
        self.miss_count = 0
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def lookup(self, query):

        query_vec = self.model.encode([query])

        for entry in self.cache:

            sim = cosine_similarity(query_vec, entry["embedding"])[0][0]

            if sim >= self.threshold:
                self.hit_count += 1

                return {
                    "hit": True,
                    "matched_query": entry["query"],
                    "similarity": float(sim),
                    "result": entry["result"],
                    "cluster": entry["cluster"]
                }

        self.miss_count += 1
        return {"hit": False, "embedding": query_vec}

    def store(self, query, embedding, result, cluster):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result,
            "cluster": cluster
        })

    def stats(self):

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

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0