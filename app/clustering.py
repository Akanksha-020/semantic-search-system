import numpy as np
import skfuzzy as fuzz
from .embeddings import embeddings

# number of clusters
n_clusters = 12

# transpose for fuzzy cmeans
data = embeddings.T

cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    data,
    c=n_clusters,
    m=2,
    error=0.005,
    maxiter=1000
)

# u = membership matrix

def get_document_cluster_distribution(doc_index):
    return u[:, doc_index]

def get_dominant_cluster(doc_index):
    return int(np.argmax(u[:, doc_index]))