"""
Create publication-quality visualizations comparing eigengaps
"""

import numpy as np
import matplotlib.pyplot as plt
from spectral_clustering import SpectralClustering, analyze_eigenspectrum

# Set publication style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# Dataset 1: Medical Innovation Network
sc1 = SpectralClustering(n_clusters=4)
A1, _, _, _ = sc1.load_graph_from_file('example1.dat')
eigenvalues1 = analyze_eigenspectrum(A1, k_max=15)

# Plot eigenvalues
ax1.plot(range(1, len(eigenvalues1)+1), eigenvalues1, 'bo-', linewidth=2.5, markersize=10)
ax1.axvline(x=4, color='r', linestyle='--', linewidth=2, label='Suggested k=4')
ax1.set_xlabel('Eigenvalue Index k', fontsize=12, fontweight='bold')
ax1.set_ylabel('Eigenvalue λₖ', fontsize=12, fontweight='bold')
ax1.set_title('(a) Medical Innovation Network - Eigenspectrum', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Plot eigengaps
gaps1 = [eigenvalues1[i] - eigenvalues1[i+1] for i in range(len(eigenvalues1)-1)]
ax2.bar(range(1, len(gaps1)+1), gaps1, color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.bar(4, gaps1[3], color='red', alpha=0.8, edgecolor='black', linewidth=1.5, label='Largest gap')
ax2.set_xlabel('Gap Position k', fontsize=12, fontweight='bold')
ax2.set_ylabel('Eigengap (λₖ - λₖ₊₁)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Medical Innovation Network - Eigengaps', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
ax2.legend(fontsize=11)

# Dataset 2: Synthetic Graph
sc2 = SpectralClustering(n_clusters=2)
A2, _, _, _ = sc2.load_graph_from_file('example2.dat')
eigenvalues2 = analyze_eigenspectrum(A2, k_max=15)

# Plot eigenvalues
ax3.plot(range(1, len(eigenvalues2)+1), eigenvalues2, 'go-', linewidth=2.5, markersize=10)
ax3.axvline(x=2, color='r', linestyle='--', linewidth=2, label='Suggested k=2')
ax3.set_xlabel('Eigenvalue Index k', fontsize=12, fontweight='bold')
ax3.set_ylabel('Eigenvalue λₖ', fontsize=12, fontweight='bold')
ax3.set_title('(c) Synthetic Graph - Eigenspectrum', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

# Plot eigengaps
gaps2 = [eigenvalues2[i] - eigenvalues2[i+1] for i in range(len(eigenvalues2)-1)]
ax4.bar(range(1, len(gaps2)+1), gaps2, color='seagreen', alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.bar(2, gaps2[1], color='red', alpha=0.8, edgecolor='black', linewidth=1.5, label='Largest gap')
ax4.set_xlabel('Gap Position k', fontsize=12, fontweight='bold')
ax4.set_ylabel('Eigengap (λₖ - λₖ₊₁)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Synthetic Graph - Eigengaps', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(fontsize=11)

plt.tight_layout()
plt.savefig('eigengap_comparison.png', dpi=300, bbox_inches='tight')
print("Eigengap comparison plot saved to: eigengap_comparison.png")

# Create summary statistics table
print("\n" + "="*70)
print("EIGENGAP ANALYSIS SUMMARY")
print("="*70)

print("\n### Dataset 1: Medical Innovation Network")
print(f"  Top 3 eigenvalues: {eigenvalues1[:3]}")
print(f"  Largest eigengap: {max(gaps1):.6f} at position {gaps1.index(max(gaps1))+1}")
print(f"  Suggested k: {gaps1.index(max(gaps1))+1}")

print("\n### Dataset 2: Synthetic Graph")
print(f"  Top 3 eigenvalues: {eigenvalues2[:3]}")
print(f"  Largest eigengap: {max(gaps2):.6f} at position {gaps2.index(max(gaps2))+1}")
print(f"  Suggested k: {gaps2.index(max(gaps2))+1}")

print("\n" + "="*70)

plt.show()
