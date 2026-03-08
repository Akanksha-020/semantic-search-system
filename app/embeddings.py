import os
import re
import numpy as np
import tarfile
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# =============================================================================
# PART 1: EMBEDDING & VECTOR DATABASE SETUP
# =============================================================================

# --- Dataset Loading ---
# We use the local mini_newsgroups dataset (tar.gz) instead of downloading
# from sklearn to have full control over what data we process and to work
# with the actual provided dataset.

def load_newsgroups_from_tar():
    """
    Load newsgroups from local tar.gz file or already extracted directory.
    Returns: list of (document_text, category, file_id) tuples
    """
    base_path = Path("data/mini_newsgroups")
    tar_path = Path("data/mini_newsgroups.tar.gz")
    
    documents = []
    
    # Extract if needed
    if not base_path.exists() and tar_path.exists():
        print("Extracting mini_newsgroups.tar.gz...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall("data/")
    
    # Load from extracted directory
    if base_path.exists():
        for category_dir in sorted(base_path.iterdir()):
            if category_dir.is_dir():
                category = category_dir.name
                for doc_file in sorted(category_dir.iterdir()):
                    if doc_file.is_file():
                        try:
                            # Try UTF-8 first, fall back to latin-1 for legacy encoding
                            # (newsgroup posts are often encoded in various formats)
                            try:
                                text = doc_file.read_text(encoding='utf-8', errors='ignore')
                            except:
                                text = doc_file.read_text(encoding='latin-1', errors='ignore')
                            
                            documents.append((text, category, doc_file.name))
                        except Exception as e:
                            print(f"Warning: Could not read {doc_file}: {e}")
    
    return documents


def clean_newsgroup_document(text):
    """
    Clean a newsgroup post for semantic embedding.
    
    Design decisions and justifications:
    
    1. HEADERS REMOVED: Email/NNTP headers (From, Subject, etc.) contain metadata
       but pollute semantic space. We want content similarity, not sender similarity.
       Exception: We keep Subject if present as it's semantically meaningful.
    
    2. QUOTED TEXT REMOVED: Lines starting with > or | are quotes from previous
       messages. They add noise and dilute the document's original semantic signal.
    
    3. SIGNATURES REMOVED: Common signature markers (--) indicate boilerplate.
       "John Smith, john@example.com" doesn't help semantic clustering.
    
    4. SHORT POSTS FILTERED: Posts <100 chars after cleaning are likely noise,
       one-liners, or test posts. They lack semantic richness for meaningful
       clustering and waste embedding model capacity.
    
    5. WHITESPACE NORMALIZED: Multiple newlines/spaces collapsed to improve
       embedding quality. The model doesn't need to learn "some posts have
       extra blank lines."
    
    6. URL PATTERNS REDUCED: Full URLs are high-entropy noise. We keep the
       concept "URL" but not the specific address (which is rarely semantically
       meaningful for clustering).
    """
    
    lines = text.split('\n')
    cleaned_lines = []
    in_header = True
    subject = None
    
    for line in lines:
        line = line.strip()
        
        # Extract subject from headers (semantically useful)
        if in_header and line.startswith('Subject:'):
            subject = line[8:].strip()
        
        # Skip header section (until first blank line)
        if in_header:
            if line == '':
                in_header = False
            continue
        
        # Skip quoted text (> or |)
        if line.startswith('>') or line.startswith('|'):
            continue
        
        # Stop at signature marker
        if line == '--':
            break
        
        # Skip empty lines temporarily (we'll normalize whitespace later)
        if not line:
            continue
        
        cleaned_lines.append(line)
    
    # Reconstruct document
    body = ' '.join(cleaned_lines)
    
    # Add subject back (it's semantically important)
    if subject:
        body = subject + '. ' + body
    
    # Reduce URLs to placeholder (preserve that a URL existed, not which one)
    body = re.sub(r'http[s]?://\S+', '[URL]', body)
    body = re.sub(r'www\.\S+', '[URL]', body)
    
    # Reduce email addresses similarly
    body = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', body)
    
    # Normalize whitespace
    body = re.sub(r'\s+', ' ', body).strip()
    
    return body


# Load and clean dataset
print("Loading newsgroups dataset...")
raw_documents = load_newsgroups_from_tar()
print(f"Loaded {len(raw_documents)} raw documents")

documents_with_metadata = []
clean_docs = []

for text, category, file_id in raw_documents:
    cleaned_text = clean_newsgroup_document(text)
    
    # Filter out very short documents (insufficient semantic content)
    if len(cleaned_text) >= 100:
        clean_docs.append(cleaned_text)
        documents_with_metadata.append({
            'text': cleaned_text,
            'category': category,
            'file_id': file_id
        })

print(f"After cleaning: {len(clean_docs)} documents (removed {len(raw_documents) - len(clean_docs)} noisy/short posts)")

# --- Embedding Model Selection ---
# Using 'all-MiniLM-L6-v2': 
# - Lightweight (80MB) for fast iteration and deployment
# - 384-dimensional embeddings (good balance of expressiveness vs. curse of dimensionality)
# - Trained on diverse text (not domain-specific), which suits the varied newsgroup topics
# - Strong performance on semantic similarity tasks per MTEB benchmarks
# 
# Alternative considered: 'all-mpnet-base-v2' (better quality, but slower inference)
# Decision: For this dataset size (~1800 docs) and fuzzy clustering downstream,
# the speed/quality tradeoff of MiniLM is optimal.

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded: all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(clean_docs, show_progress_bar=True, batch_size=32)
embeddings = np.array(embeddings).astype("float32")
print(f"Embeddings shape: {embeddings.shape}")

# --- Vector Database Setup ---
# Using FAISS IndexFlatL2:
# - Exact nearest neighbor search (no approximation errors for this dataset size)
# - L2 distance is appropriate for normalized embeddings from sentence-transformers
# - Simple and deterministic (good for reproducibility)
# 
# For production at scale (>100k docs), consider IndexIVFFlat or IndexHNSWFlat
# But for ~1800 docs, flat index is fast enough (<1ms query time) and has zero recall loss

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print(f"FAISS index created with {index.ntotal} vectors")

# Create category mapping for filtered retrieval
categories = [doc['category'] for doc in documents_with_metadata]
category_to_indices = {}
for idx, category in enumerate(categories):
    if category not in category_to_indices:
        category_to_indices[category] = []
    category_to_indices[category].append(idx)


def search_vector(query_embedding, k=5, filter_category=None):
    """
    Search for similar documents.
    
    Args:
        query_embedding: numpy array of shape (1, dimension)
        k: number of results to return
        filter_category: optional category name to filter results
    
    Returns:
        I: indices of matching documents
    """
    if filter_category and filter_category in category_to_indices:
        # Filtered retrieval: only search within specific category
        # Build a temporary index with only docs from that category
        category_indices = category_to_indices[filter_category]
        category_embeddings = embeddings[category_indices]
        
        temp_index = faiss.IndexFlatL2(dimension)
        temp_index.add(category_embeddings)
        
        D, I = temp_index.search(query_embedding, min(k, len(category_indices)))
        # Map back to global indices
        I = np.array([[category_indices[i] for i in I[0]]])
        return I
    else:
        # Global search
        D, I = index.search(query_embedding, k)
        return I


def get_document_metadata(doc_index):
    """Get metadata for a document by index"""
    if 0 <= doc_index < len(documents_with_metadata):
        return documents_with_metadata[doc_index]
    return None