"""
Spectral Clustering Implementation
Based on "On Spectral Clustering: Analysis and an algorithm" by Ng, Jordan, and Weiss
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import defaultdict


class SpectralClustering:
    """
    Spectral Clustering using the algorithm from Ng, Jordan, and Weiss (2001)
    """
    
    def __init__(self, n_clusters, sigma=1.0):
        """
        Initialize Spectral Clustering
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters to find
        sigma : float
            Parameter for Gaussian similarity (for weighted graphs)
        """
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.labels_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        
    def load_graph_from_file(self, filename):
        """
        Load graph from file
        
        For example1.dat: format is "i,j" (edge list, unweighted)
        For example2.dat: format is "i,j,weight" (edge list, weighted)
        
        Returns:
        --------
        A : scipy.sparse matrix
            Affinity matrix
        n : int
            Number of nodes
        """
        edges = []
        weights = []
        nodes = set()
        
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    i, j = int(parts[0]), int(parts[1])
                    w = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    edges.append((i, j))
                    weights.append(w)
                    nodes.add(i)
                    nodes.add(j)
        
        # Create mapping from node IDs to indices (0-based)
        node_list = sorted(nodes)
        n = len(node_list)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        
        # Build affinity matrix A
        row_indices = []
        col_indices = []
        data = []
        
        for (i, j), w in zip(edges, weights):
            idx_i = node_to_idx[i]
            idx_j = node_to_idx[j]
            
            # Gaussian similarity if sigma is provided and graph is weighted
            if self.sigma is not None and w != 1.0:
                similarity = np.exp(-w**2 / (2 * self.sigma**2))
            else:
                similarity = 1.0  # For unweighted graphs
            
            # Symmetric matrix
            row_indices.extend([idx_i, idx_j])
            col_indices.extend([idx_j, idx_i])
            data.extend([similarity, similarity])
        
        A = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        return A, n, node_to_idx, node_list
    
    def fit(self, A):
        """
        Fit the spectral clustering model
        
        Algorithm from Ng, Jordan, Weiss (2001):
        1. Form the affinity matrix A
        2. Define D to be the diagonal matrix whose (i,i)-element is the sum of A's i-th row
        3. Form the normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
        4. Find the k largest eigenvectors of L, and form matrix X
        5. Form matrix Y by renormalizing each of X's rows to have unit length
        6. Treating each row of Y as a point in R^k, cluster into k clusters via K-means
        7. Assign the original point s_i to cluster j if row i of Y was assigned to cluster j
        
        Parameters:
        -----------
        A : scipy.sparse matrix
            Affinity matrix (n x n)
        """
        n = A.shape[0]
        
        # Step 2: Compute degree matrix D
        # D[i,i] = sum of A's i-th row
        d = np.array(A.sum(axis=1)).flatten()
        
        # Avoid division by zero
        d[d == 0] = 1e-10
        
        # Step 3: Form normalized Laplacian L = D^(-1/2) * A * D^(-1/2)
        # This is equivalent to the normalized adjacency matrix
        D_inv_sqrt = np.sqrt(1.0 / d)
        
        # L = D^(-1/2) * A * D^(-1/2)
        # For sparse matrices, we compute this efficiently
        L = A.copy()
        L = L.tocsr()
        
        # Multiply rows by D^(-1/2)
        for i in range(n):
            L.data[L.indptr[i]:L.indptr[i+1]] *= D_inv_sqrt[i]
        
        # Multiply columns by D^(-1/2)
        L = L.tocsc()
        for j in range(n):
            L.data[L.indptr[j]:L.indptr[j+1]] *= D_inv_sqrt[j]
        L = L.tocsr()
        
        # Step 4: Find k largest eigenvectors
        # Note: we find k largest eigenvalues of L
        try:
            eigenvalues, eigenvectors = eigsh(L, k=self.n_clusters, which='LA')
        except:
            # If sparse solver fails, convert to dense
            L_dense = L.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L_dense)
            # Get k largest
            idx = eigenvalues.argsort()[-self.n_clusters:][::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        
        # Step 5: Normalize rows of eigenvector matrix to unit length
        X = eigenvectors
        Y = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        
        # Step 6: Run K-means on rows of Y
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        self.labels_ = kmeans.fit_predict(Y)
        
        return self
    
    def fit_predict(self, A):
        """
        Fit the model and return cluster labels
        """
        self.fit(A)
        return self.labels_


def analyze_eigenspectrum(A, k_max=10):
    """
    Analyze the eigenspectrum of the normalized Laplacian
    to determine the number of clusters
    
    Parameters:
    -----------
    A : scipy.sparse matrix
        Affinity matrix
    k_max : int
        Maximum number of eigenvalues to compute
    
    Returns:
    --------
    eigenvalues : array
        Computed eigenvalues
    """
    n = A.shape[0]
    
    # Compute degree matrix
    d = np.array(A.sum(axis=1)).flatten()
    d[d == 0] = 1e-10
    D_inv_sqrt = np.sqrt(1.0 / d)
    
    # Form normalized Laplacian
    L = A.copy().tocsr()
    
    # Multiply rows by D^(-1/2)
    for i in range(n):
        L.data[L.indptr[i]:L.indptr[i+1]] *= D_inv_sqrt[i]
    
    # Multiply columns by D^(-1/2)
    L = L.tocsc()
    for j in range(n):
        L.data[L.indptr[j]:L.indptr[j+1]] *= D_inv_sqrt[j]
    L = L.tocsr()
    
    # Compute eigenvalues
    try:
        eigenvalues, _ = eigsh(L, k=min(k_max, n-2), which='LA')
        eigenvalues = np.sort(eigenvalues)[::-1]
    except:
        # Fallback to dense computation
        L_dense = L.toarray()
        eigenvalues, _ = np.linalg.eigh(L_dense)
        eigenvalues = np.sort(eigenvalues)[::-1][:k_max]
    
    return eigenvalues


def visualize_results(labels, node_list, title="Spectral Clustering Results"):
    """
    Visualize clustering results
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    print(f"\n{title}")
    print("=" * 60)
    print(f"Number of clusters: {n_clusters}")
    print(f"Total nodes: {len(labels)}")
    
    for cluster_id in unique_labels:
        nodes_in_cluster = [node_list[i] for i, label in enumerate(labels) if label == cluster_id]
        print(f"\nCluster {cluster_id}: {len(nodes_in_cluster)} nodes")
        print(f"  Nodes: {nodes_in_cluster[:20]}{'...' if len(nodes_in_cluster) > 20 else ''}")


def plot_eigenspectrum(eigenvalues, title="Eigenspectrum"):
    """
    Plot eigenvalue spectrum to help determine number of clusters
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(eigenvalues)+1), eigenvalues, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Eigenvalue Index', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def main():
    """
    Main function to analyze both datasets
    """
    print("Spectral Clustering Analysis")
    print("=" * 80)
    
    # Analyze example1.dat
    print("\n\n### DATASET 1: example1.dat (Medical Innovation Network) ###\n")
    
    # Load graph
    sc1 = SpectralClustering(n_clusters=4)  # We'll try different k values
    A1, n1, node_to_idx1, node_list1 = sc1.load_graph_from_file('example1.dat')
    
    print(f"Graph loaded: {n1} nodes, {A1.nnz//2} edges")
    
    # Analyze eigenspectrum
    print("\nAnalyzing eigenspectrum...")
    eigenvalues1 = analyze_eigenspectrum(A1, k_max=15)
    print(f"Top 15 eigenvalues: {eigenvalues1}")
    print(f"\nEigengap analysis:")
    for i in range(len(eigenvalues1)-1):
        gap = eigenvalues1[i] - eigenvalues1[i+1]
        print(f"  Gap {i+1}: {gap:.6f} (between λ_{i+1}={eigenvalues1[i]:.6f} and λ_{i+2}={eigenvalues1[i+1]:.6f})")
    
    # Plot eigenspectrum
    plt1 = plot_eigenspectrum(eigenvalues1, "Eigenspectrum - Medical Innovation Network")
    plt1.savefig('example1_eigenspectrum.png', dpi=150, bbox_inches='tight')
    print("\nEigenspectrum plot saved to: example1_eigenspectrum.png")
    
    # Try different numbers of clusters
    print("\n" + "="*60)
    print("Testing different numbers of clusters:")
    for k in [2, 3, 4, 5]:
        print(f"\n--- K = {k} clusters ---")
        sc_test = SpectralClustering(n_clusters=k)
        A1_test, _, _, node_list_test = sc_test.load_graph_from_file('example1.dat')
        labels = sc_test.fit_predict(A1_test)
        visualize_results(labels, node_list_test, f"Results with K={k}")
    
    # Analyze example2.dat
    print("\n\n" + "="*80)
    print("### DATASET 2: example2.dat (Synthetic Graph) ###\n")
    
    # Load graph
    sc2 = SpectralClustering(n_clusters=2)
    A2, n2, node_to_idx2, node_list2 = sc2.load_graph_from_file('example2.dat')
    
    print(f"Graph loaded: {n2} nodes, {A2.nnz//2} edges")
    
    # Analyze eigenspectrum
    print("\nAnalyzing eigenspectrum...")
    eigenvalues2 = analyze_eigenspectrum(A2, k_max=15)
    print(f"Top 15 eigenvalues: {eigenvalues2}")
    print(f"\nEigengap analysis:")
    for i in range(len(eigenvalues2)-1):
        gap = eigenvalues2[i] - eigenvalues2[i+1]
        print(f"  Gap {i+1}: {gap:.6f} (between λ_{i+1}={eigenvalues2[i]:.6f} and λ_{i+2}={eigenvalues2[i+1]:.6f})")
    
    # Plot eigenspectrum
    plt2 = plot_eigenspectrum(eigenvalues2, "Eigenspectrum - Synthetic Graph")
    plt2.savefig('example2_eigenspectrum.png', dpi=150, bbox_inches='tight')
    print("\nEigenspectrum plot saved to: example2_eigenspectrum.png")
    
    # Try different numbers of clusters
    print("\n" + "="*60)
    print("Testing different numbers of clusters:")
    for k in [2, 3, 4]:
        print(f"\n--- K = {k} clusters ---")
        sc_test = SpectralClustering(n_clusters=k)
        A2_test, _, _, node_list_test = sc_test.load_graph_from_file('example2.dat')
        labels = sc_test.fit_predict(A2_test)
        visualize_results(labels, node_list_test, f"Results with K={k}")
    
    print("\n" + "="*80)
    print("Analysis complete!")


if __name__ == "__main__":
    main()
