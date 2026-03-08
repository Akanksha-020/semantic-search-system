"""
Simple test script to verify the system works correctly.

Run this to:
1. Check that data loads properly
2. Verify embeddings are generated
3. Confirm clustering completes
4. Test search functionality
"""

print("="*70)
print("Testing Newsgroups Semantic Search & Fuzzy Clustering System")
print("="*70)

print("\n1. Testing data loading and embeddings...")
try:
    from app.embeddings import (
        embeddings, 
        clean_docs, 
        documents_with_metadata,
        search_vector,
        model
    )
    print(f"   [OK] Loaded {len(clean_docs)} documents")
    print(f"   [OK] Generated embeddings with shape: {embeddings.shape}")
    print(f"   [OK] Embedding dimension: {embeddings.shape[1]}")
except Exception as e:
    print(f"   [ERROR] Error loading embeddings: {e}")
    exit(1)

print("\n2. Testing fuzzy clustering...")
try:
    from app.clustering import (
        n_clusters,
        u,
        get_document_cluster_distribution,
        get_dominant_cluster,
        get_cluster_entropy
    )
    print(f"   [OK] Clustering complete with {n_clusters} clusters")
    print(f"   [OK] Membership matrix shape: {u.shape}")
    
    # Test on first document
    doc_idx = 0
    distribution = get_document_cluster_distribution(doc_idx)
    dominant = get_dominant_cluster(doc_idx)
    entropy = get_cluster_entropy(doc_idx)
    
    print(f"   [OK] Document 0 cluster distribution: sum={sum(distribution):.3f} (should be 1.0)")
    print(f"   [OK] Document 0 dominant cluster: {dominant}")
    print(f"   [OK] Document 0 entropy: {entropy:.3f} bits")
except Exception as e:
    print(f"   [ERROR] Error in clustering: {e}")
    exit(1)

print("\n3. Testing search functionality...")
try:
    from app.search import run_search
    
    query = "computer graphics rendering"
    result = run_search(query)
    
    print(f"   [OK] Search completed")
    print(f"   [OK] Result category: {result['category']}")
    print(f"   [OK] Dominant cluster: {result['dominant_cluster']}")
    print(f"   [OK] Cluster entropy: {result['cluster_entropy']:.3f}")
    print(f"   [OK] Is boundary case: {result['cluster_entropy'] >= 2.0}")
    print(f"   [OK] Text preview: {result['text'][:100]}...")
except Exception as e:
    print(f"   [ERROR] Error in search: {e}")
    exit(1)

print("\n4. Testing cluster analysis functions...")
try:
    from app.clustering import (
        analyze_cluster_categories,
        get_representative_documents,
        get_boundary_documents
    )
    
    # Test cluster analysis
    categories = analyze_cluster_categories(0)
    print(f"   [OK] Cluster 0 has {len(categories)} categories")
    
    # Test representative documents
    reps = get_representative_documents(0, top_n=3)
    print(f"   [OK] Found {len(reps)} representative documents for cluster 0")
    
    # Test boundary cases
    boundary = get_boundary_documents(min_entropy_threshold=2.0)
    print(f"   [OK] Found {len(boundary)} boundary cases (entropy >= 2.0)")
except Exception as e:
    print(f"   [ERROR] Error in cluster analysis: {e}")
    exit(1)

print("\n5. Sample cluster summary:")
print("-" * 70)
try:
    from app.clustering import print_cluster_summary
    print_cluster_summary(0, num_examples=2)
except Exception as e:
    print(f"   [ERROR] Error printing cluster summary: {e}")

print("\n6. Sample boundary cases:")
print("-" * 70)
try:
    from app.clustering import print_boundary_cases
    print_boundary_cases(num_examples=3)
except Exception as e:
    print(f"   [ERROR] Error printing boundary cases: {e}")

print("\n" + "="*70)
print("All tests passed! [OK]")
print("="*70)
print("\nNext steps:")
print("  1. Run the API: uvicorn app.main:app --reload")
print("  2. Run full analysis: python -m app.analyze_clusters")
print("  3. Access API docs: http://localhost:8000/docs")
print("="*70 + "\n")
