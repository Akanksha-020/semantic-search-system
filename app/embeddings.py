import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import faiss

# Load dataset
data = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
documents = data.data

# Cleaning step
clean_docs = []
for doc in documents:
    doc = doc.strip()
    if len(doc) > 50:  # remove extremely short noisy posts
        clean_docs.append(doc)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
embeddings = model.encode(clean_docs, show_progress_bar=True)

# Convert to numpy
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

def search_vector(query_embedding, k=5):
    D, I = index.search(query_embedding, k)
    return I