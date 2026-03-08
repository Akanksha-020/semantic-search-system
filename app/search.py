from sentence_transformers import SentenceTransformer
from .embeddings import search_vector, clean_docs
from .clustering import get_dominant_cluster

model = SentenceTransformer("all-MiniLM-L6-v2")

def run_search(query):

    q_embed = model.encode([query]).astype("float32")

    results = search_vector(q_embed, k=1)

    doc_index = int(results[0][0])

    result_text = clean_docs[doc_index]

    cluster = get_dominant_cluster(doc_index)

    return result_text, cluster, q_embed