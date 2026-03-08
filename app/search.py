from sentence_transformers import SentenceTransformer
from .embeddings import search_vector, clean_docs, get_document_metadata
from .clustering import get_dominant_cluster, get_top_k_clusters, get_cluster_entropy

model = SentenceTransformer("all-MiniLM-L6-v2")

def run_search(query, filter_category=None, k=1):
    """
    Run semantic search on the newsgroups corpus.
    
    Args:
        query: search query string
        filter_category: optional category name to filter results
        k: number of results to return
    
    Returns:
        result_data: dict with search results and cluster information
    """
    # Encode query
    q_embed = model.encode([query]).astype("float32")
    
    # Search vector database
    results = search_vector(q_embed, k=k, filter_category=filter_category)
    
    doc_index = int(results[0][0])
    
    # Get document and metadata
    result_text = clean_docs[doc_index]
    metadata = get_document_metadata(doc_index)
    
    # Get cluster information
    dominant_cluster = get_dominant_cluster(doc_index)
    top_clusters = get_top_k_clusters(doc_index, k=3)
    entropy = get_cluster_entropy(doc_index)
    
    result_data = {
        'text': result_text,
        'category': metadata['category'],
        'file_id': metadata['file_id'],
        'dominant_cluster': dominant_cluster,
        'top_clusters': top_clusters,
        'cluster_entropy': float(entropy),
        'embedding': q_embed
    }
    
    return result_data