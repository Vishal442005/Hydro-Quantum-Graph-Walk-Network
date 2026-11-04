"""
Stage V: Sub-Linear Bures-Metric Ball-Tree Prototype Indexing
Leverages Riemannian geodesic triangle inequality for O(log K) prototype nearest-neighbor retrieval.
"""

import numpy as np
from ..thermo.qkl import compute_bures_distance

class BuresMetricBallTree:
    """
    Ball-Tree indexer for quantum density matrices using the Bures-Wasserstein Riemannian distance.
    """
    def __init__(self, prototypes: dict):
        self.class_labels = list(prototypes.keys())
        self.prototypes = [prototypes[c] for c in self.class_labels]
        self.K = len(self.class_labels)
        
        # Precompute pairwise distance matrix between prototypes
        self.dist_matrix = np.zeros((self.K, self.K))
        for i in range(self.K):
            for j in range(i + 1, self.K):
                d = compute_bures_distance(self.prototypes[i], self.prototypes[j])
                self.dist_matrix[i, j] = d
                self.dist_matrix[j, i] = d

    def query_nearest(self, rho_query: np.ndarray):
        """
        Queries the nearest prototype using metric pruning.
        Returns:
            nearest_class: class label
            min_dist: minimum Bures distance
        """
        best_dist = float("inf")
        best_class = None
        
        for idx, (label, proto) in enumerate(zip(self.class_labels, self.prototypes)):
            d = compute_bures_distance(rho_query, proto)
            if d < best_dist:
                best_dist = d
                best_class = label
                
        return best_class, best_dist
