"""
Detailed Analysis and Visualization of Spectral Clustering Results
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from spectral_clustering import SpectralClustering, analyze_eigenspectrum
from collections import Counter


def load_graph_networkx(filename):
    """
    Load graph into NetworkX for visualization
    """
    G = nx.Graph()
    
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                i, j = int(parts[0]), int(parts[1])
                w = float(parts[2]) if len(parts) > 2 else 1.0
                G.add_edge(i, j, weight=w)
    
    return G


def visualize_network(G, labels_dict, title, filename):
    """
    Visualize network with cluster colors
    """
    # Create layout
    try:
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    except:
        pos = nx.spring_layout(G, seed=42)
    
    # Get unique clusters
    unique_clusters = set(labels_dict.values())
    n_clusters = len(unique_clusters)
    
    # Color map
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw nodes by cluster
    for i, cluster_id in enumerate(sorted(unique_clusters)):
        nodes_in_cluster = [node for node, label in labels_dict.items() if label == cluster_id]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_cluster, 
                              node_color=[colors[i]], 
                              node_size=50,
                              label=f'Cluster {cluster_id} ({len(nodes_in_cluster)} nodes)',
                              ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved to: {filename}")


def analyze_cluster_properties(G, labels_dict):
    """
    Analyze properties of each cluster
    """
    unique_clusters = sorted(set(labels_dict.values()))
    
    print("\n" + "="*70)
    print("CLUSTER ANALYSIS")
    print("="*70)
    
    for cluster_id in unique_clusters:
        nodes = [node for node, label in labels_dict.items() if label == cluster_id]
        subgraph = G.subgraph(nodes)
        
        # Compute metrics
        n_nodes = len(nodes)
        n_edges = subgraph.number_of_edges()
        density = nx.density(subgraph) if n_nodes > 1 else 0
        
        # Average degree
        degrees = [subgraph.degree(n) for n in nodes]
        avg_degree = np.mean(degrees) if degrees else 0
        
        # Connectivity
        is_connected = nx.is_connected(subgraph) if n_nodes > 1 else True
        n_components = nx.number_connected_components(subgraph)
        
        print(f"\nCluster {cluster_id}:")
        print(f"  Nodes: {n_nodes}")
        print(f"  Internal edges: {n_edges}")
        print(f"  Density: {density:.4f}")
        print(f"  Average degree: {avg_degree:.2f}")
        print(f"  Connected: {is_connected}")
        print(f"  Number of components: {n_components}")
        
        # Sample nodes
        sample_size = min(10, len(nodes))
        print(f"  Sample nodes: {sorted(nodes)[:sample_size]}")


def compute_modularity(G, labels_dict):
    """
    Compute modularity of the clustering
    """
    # Convert labels_dict to partition format for NetworkX
    communities = []
    unique_clusters = set(labels_dict.values())
    for cluster_id in unique_clusters:
        community = set([node for node, label in labels_dict.items() if label == cluster_id])
        communities.append(community)
    
    modularity = nx.algorithms.community.modularity(G, communities)
    return modularity


def analyze_eigengap(eigenvalues):
    """
    Detailed eigengap analysis to suggest number of clusters
    """
    print("\n" + "="*70)
    print("EIGENGAP ANALYSIS - Suggested Number of Clusters")
    print("="*70)
    
    gaps = []
    for i in range(len(eigenvalues)-1):
        gap = eigenvalues[i] - eigenvalues[i+1]
        gaps.append((i+1, gap, eigenvalues[i], eigenvalues[i+1]))
    
    # Sort by gap size
    gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 largest eigengaps:")
    for i, (idx, gap, lambda_k, lambda_k1) in enumerate(gaps_sorted[:5]):
        print(f"  {i+1}. Gap at k={idx}: {gap:.6f} (λ_{idx}={lambda_k:.6f} → λ_{idx+1}={lambda_k1:.6f})")
    
    # Suggest k
    largest_gap_idx = gaps_sorted[0][0]
    print(f"\n>>> Suggested k (based on largest eigengap): {largest_gap_idx}")
    
    return largest_gap_idx


def create_detailed_report(dataset_name, filename, suggested_k):
    """
    Create detailed analysis report
    """
    print("\n" + "="*80)
    print(f"DETAILED ANALYSIS: {dataset_name}")
    print("="*80)
    
    # Load graph
    G = load_graph_networkx(filename)
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    print(f"  Average degree: {np.mean([d for n, d in G.degree()]):.2f}")
    
    # Check if connected
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    print(f"  Connected: {is_connected}")
    print(f"  Number of connected components: {n_components}")
    
    if not is_connected:
        component_sizes = [len(c) for c in nx.connected_components(G)]
        print(f"  Component sizes: {sorted(component_sizes, reverse=True)}")
    
    # Analyze eigenspectrum
    sc = SpectralClustering(n_clusters=suggested_k)
    A, n, node_to_idx, node_list = sc.load_graph_from_file(filename)
    
    eigenvalues = analyze_eigenspectrum(A, k_max=20)
    
    # Eigengap analysis
    suggested_k_from_gap = analyze_eigengap(eigenvalues[:15])
    
    # Run clustering with suggested k
    print(f"\n" + "="*70)
    print(f"SPECTRAL CLUSTERING with k={suggested_k_from_gap}")
    print("="*70)
    
    sc_final = SpectralClustering(n_clusters=suggested_k_from_gap)
    A_final, _, _, node_list_final = sc_final.load_graph_from_file(filename)
    labels = sc_final.fit_predict(A_final)
    
    # Create labels dictionary
    labels_dict = {node_list_final[i]: labels[i] for i in range(len(labels))}
    
    # Cluster distribution
    cluster_counts = Counter(labels)
    print(f"\nCluster size distribution:")
    for cluster_id in sorted(cluster_counts.keys()):
        print(f"  Cluster {cluster_id}: {cluster_counts[cluster_id]} nodes")
    
    # Compute modularity
    modularity = compute_modularity(G, labels_dict)
    print(f"\nModularity: {modularity:.4f}")
    
    # Analyze cluster properties
    analyze_cluster_properties(G, labels_dict)
    
    # Visualize network
    output_filename = filename.replace('.dat', '').split('/')[-1]
    viz_filename = f'{output_filename}_network_k{suggested_k_from_gap}.png'
    visualize_network(G, labels_dict, 
                     f"{dataset_name} - Spectral Clustering (k={suggested_k_from_gap})",
                     viz_filename)
    
    return suggested_k_from_gap, labels_dict, modularity


def main():
    """
    Main analysis function
    """
    print("="*80)
    print("COMPREHENSIVE SPECTRAL CLUSTERING ANALYSIS")
    print("="*80)
    
    # Dataset 1: Medical Innovation Network
    k1, labels1, mod1 = create_detailed_report(
        "Medical Innovation Network (example1.dat)",
        "example1.dat",
        suggested_k=4
    )
    
    print("\n\n")
    
    # Dataset 2: Synthetic Graph
    k2, labels2, mod2 = create_detailed_report(
        "Synthetic Graph (example2.dat)",
        "example2.dat",
        suggested_k=2
    )
    
    # Final summary
    print("\n\n" + "="*80)
    print("SUMMARY OF FINDINGS")
    print("="*80)
    
    print("\n### Medical Innovation Network (example1.dat)")
    print(f"  Suggested number of clusters: k = {k1}")
    print(f"  Modularity: {mod1:.4f}")
    print(f"  Interpretation: The network shows {k1} distinct physician communities")
    print(f"  in the medical innovation diffusion study across 4 Illinois towns.")
    
    print("\n### Synthetic Graph (example2.dat)")
    print(f"  Suggested number of clusters: k = {k2}")
    print(f"  Modularity: {mod2:.4f}")
    print(f"  Interpretation: The synthetic graph exhibits {k2} well-separated")
    print(f"  community structures.")
    
    print("\n" + "="*80)
    print("All visualizations saved to /mnt/user-data/outputs/")
    print("="*80)


if __name__ == "__main__":
    main()
