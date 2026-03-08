"""
Cluster Analysis Script

Run this independently to analyze the fuzzy clustering results in detail.

Usage:
    python -m app.analyze_clusters
"""

from .clustering import (
    print_cluster_summary,
    print_boundary_cases,
    get_boundary_documents,
    analyze_cluster_categories,
    get_representative_documents,
    n_clusters,
    u,
    get_cluster_entropy
)
from .embeddings import clean_docs, documents_with_metadata
import numpy as np


def analyze_all_clusters(detailed=False):
    """
    Comprehensive analysis of all clusters.
    
    Args:
        detailed: if True, show more examples per cluster
    """
    print("\n" + "="*80)
    print(" " * 20 + "COMPREHENSIVE CLUSTER ANALYSIS")
    print("="*80 + "\n")
    
    num_examples = 5 if detailed else 3
    
    for cluster_id in range(n_clusters):
        print_cluster_summary(cluster_id, num_examples=num_examples)
        print()
    
    print("\n" + "="*80)
    print(" " * 25 + "BOUNDARY CASE ANALYSIS")
    print("="*80)
    
    print_boundary_cases(num_examples=10 if detailed else 5)


def cluster_statistics():
    """
    Print statistical analysis of the clustering.
    """
    print("\n" + "="*80)
    print(" " * 25 + "CLUSTERING STATISTICS")
    print("="*80 + "\n")
    
    # Dominant cluster assignments
    dominant_clusters = np.argmax(u, axis=0)
    
    # Distribution of dominant clusters
    unique, counts = np.unique(dominant_clusters, return_counts=True)
    print("Documents per cluster (by dominant assignment):")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / len(dominant_clusters)) * 100
        print(f"  Cluster {cluster_id:2d}: {count:4d} documents ({percentage:5.2f}%)")
    
    print()
    
    # Membership statistics
    print("Membership strength statistics:")
    max_memberships = np.max(u, axis=0)
    print(f"  Mean maximum membership: {np.mean(max_memberships):.3f}")
    print(f"  Median maximum membership: {np.median(max_memberships):.3f}")
    print(f"  Min maximum membership: {np.min(max_memberships):.3f}")
    print(f"  Max maximum membership: {np.max(max_memberships):.3f}")
    
    print()
    
    # Entropy statistics
    entropies = [get_cluster_entropy(i) for i in range(len(clean_docs))]
    print("Cluster assignment entropy statistics:")
    print(f"  Mean entropy: {np.mean(entropies):.3f} bits")
    print(f"  Median entropy: {np.median(entropies):.3f} bits")
    print(f"  Min entropy: {np.min(entropies):.3f} bits (most certain)")
    print(f"  Max entropy: {np.max(entropies):.3f} bits (most uncertain)")
    
    # Count boundary cases
    boundary_count = sum(1 for e in entropies if e >= 2.0)
    print(f"\n  Documents with entropy >= 2.0 (boundary cases): {boundary_count} ({boundary_count/len(entropies)*100:.1f}%)")
    
    print()
    
    # Cross-category analysis
    print("Cross-category semantic grouping:")
    print("  (Clusters that significantly span multiple original newsgroup categories)")
    print()
    
    for cluster_id in range(n_clusters):
        category_analysis = analyze_cluster_categories(cluster_id)
        if len(category_analysis) >= 3:
            # Check if cluster has significant representation from multiple categories
            top_3_counts = [item['count'] for item in category_analysis[:3]]
            if top_3_counts[1] >= 0.3 * top_3_counts[0]:  # Second category has at least 30% of first
                print(f"  Cluster {cluster_id}:")
                for item in category_analysis[:3]:
                    print(f"    - {item['category']:<35} {item['count']:3d} docs ({item['avg_membership']:.3f} avg membership)")
                print()


def find_interesting_documents():
    """
    Find and display particularly interesting documents:
    - Highest entropy (most uncertain)
    - Most evenly distributed across clusters
    - Pure examples (very high membership in single cluster)
    """
    print("\n" + "="*80)
    print(" " * 22 + "INTERESTING DOCUMENT CASES")
    print("="*80 + "\n")
    
    # Most uncertain documents
    print("1. MOST UNCERTAIN DOCUMENTS (highest entropy):\n")
    entropies = [(i, get_cluster_entropy(i)) for i in range(len(clean_docs))]
    entropies.sort(key=lambda x: x[1], reverse=True)
    
    for i, (doc_idx, entropy) in enumerate(entropies[:5], 1):
        metadata = documents_with_metadata[doc_idx]
        memberships = u[:, doc_idx]
        top_3_clusters = np.argsort(memberships)[::-1][:3]
        
        print(f"   {i}. Entropy: {entropy:.3f} bits | Category: {metadata['category']}")
        print(f"      Top cluster memberships:")
        for cluster_id in top_3_clusters:
            print(f"        Cluster {cluster_id}: {memberships[cluster_id]:.3f}")
        text_preview = clean_docs[doc_idx][:150].replace('\n', ' ')
        print(f"      Preview: {text_preview}...")
        print()
    
    # Purest documents (lowest entropy)
    print("\n2. PUREST DOCUMENTS (lowest entropy, strongest single cluster membership):\n")
    entropies.sort(key=lambda x: x[1])
    
    for i, (doc_idx, entropy) in enumerate(entropies[:5], 1):
        metadata = documents_with_metadata[doc_idx]
        memberships = u[:, doc_idx]
        dominant_cluster = np.argmax(memberships)
        
        print(f"   {i}. Entropy: {entropy:.3f} bits | Cluster {dominant_cluster} membership: {memberships[dominant_cluster]:.3f}")
        print(f"      Category: {metadata['category']}")
        text_preview = clean_docs[doc_idx][:150].replace('\n', ' ')
        print(f"      Preview: {text_preview}...")
        print()


def cluster_confusion_matrix():
    """
    Show how original newsgroup categories are distributed across clusters.
    This reveals whether clusters capture semantic themes that cross categorical boundaries.
    """
    print("\n" + "="*80)
    print(" " * 20 + "CLUSTER vs CATEGORY MAPPING")
    print("="*80 + "\n")
    
    # Get all unique categories
    all_categories = sorted(set(doc['category'] for doc in documents_with_metadata))
    
    print("Top 3 clusters for each original newsgroup category:\n")
    
    for category in all_categories:
        # Get all documents in this category
        category_docs = [i for i, doc in enumerate(documents_with_metadata) if doc['category'] == category]
        
        if not category_docs:
            continue
        
        # Aggregate cluster memberships for this category
        cluster_memberships = np.zeros(n_clusters)
        for doc_idx in category_docs:
            cluster_memberships += u[:, doc_idx]
        
        # Normalize
        cluster_memberships /= len(category_docs)
        
        # Get top 3 clusters
        top_clusters = np.argsort(cluster_memberships)[::-1][:3]
        
        print(f"  {category:<35}")
        for cluster_id in top_clusters:
            print(f"    → Cluster {cluster_id:2d}: {cluster_memberships[cluster_id]:.3f} avg membership")
        print()


if __name__ == "__main__":
    print("\nStarting cluster analysis...")
    print(f"Total documents: {len(clean_docs)}")
    print(f"Total clusters: {n_clusters}")
    
    # Run all analyses
    cluster_statistics()
    cluster_confusion_matrix()
    find_interesting_documents()
    
    # Optional: detailed cluster summaries
    print("\n" + "="*80)
    response = input("Show detailed analysis of all clusters? (y/n): ")
    if response.lower() == 'y':
        analyze_all_clusters(detailed=True)
    else:
        print("\nTo see full cluster analysis, run:")
        print("  from app.clustering import print_cluster_summary")
        print("  print_cluster_summary(cluster_id)")
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")
