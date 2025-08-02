"""Gene clustering for coherent discussion groups."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage

from .prioritizer import PrioritizedGene


@dataclass
class GeneCluster:
    """A cluster of related genes."""

    cluster_id: int
    genes: List[PrioritizedGene]
    mean_log2fc: float = field(init=False)
    predominant_direction: str = field(init=False)
    size: int = field(init=False)

    def __post_init__(self):
        self.size = len(self.genes)
        self.mean_log2fc = np.mean([g.de_result.log2_fold_change for g in self.genes])
        self.predominant_direction = "up" if self.mean_log2fc > 0 else "down"

    def get_top_genes(self, n: int = 5) -> List[PrioritizedGene]:
        """Get top n genes by priority score."""
        return sorted(self.genes, key=lambda g: g.combined_score, reverse=True)[:n]

    def get_summary(self) -> Dict[str, any]:
        """Get cluster summary statistics."""
        log2fcs = [g.de_result.log2_fold_change for g in self.genes]
        padjs = [g.de_result.padj for g in self.genes]

        return {
            "cluster_id": self.cluster_id,
            "size": self.size,
            "direction": self.predominant_direction,
            "mean_log2fc": self.mean_log2fc,
            "std_log2fc": np.std(log2fcs),
            "mean_padj": np.mean(padjs),
            "top_genes": [g.gene_symbol or g.gene_id for g in self.get_top_genes()],
        }


class GeneClusterer:
    """Cluster genes based on expression patterns and functional relationships."""

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        distance_threshold: float = 0.5,
        min_cluster_size: int = 3,
    ):
        self.n_clusters = n_clusters
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.clusters: List[GeneCluster] = []
        self.clustering_model = None

    def cluster_by_expression(
        self,
        prioritized_genes: List[PrioritizedGene],
        expression_matrix: Optional[np.ndarray] = None,
    ) -> List[GeneCluster]:
        """Cluster genes based on expression patterns."""
        if len(prioritized_genes) < self.min_cluster_size:
            # Too few genes, return single cluster
            return [GeneCluster(cluster_id=0, genes=prioritized_genes)]

        # Create feature matrix
        if expression_matrix is not None:
            # Use provided expression matrix (e.g., from multiple conditions)
            features = expression_matrix
        else:
            # Use simple features from DE results
            features = self._create_simple_features(prioritized_genes)

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine optimal number of clusters if not specified
        if self.n_clusters is None:
            self.n_clusters = self._estimate_n_clusters(
                features_scaled, len(prioritized_genes)
            )

        # Perform clustering
        self.clustering_model = AgglomerativeClustering(
            n_clusters=self.n_clusters, linkage="ward"
        )
        labels = self.clustering_model.fit_predict(features_scaled)

        # Create cluster objects
        self.clusters = self._create_clusters(prioritized_genes, labels)

        # Filter small clusters
        self.clusters = [c for c in self.clusters if c.size >= self.min_cluster_size]

        # Re-assign genes from small clusters to nearest large cluster
        if len(self.clusters) < len(set(labels)):
            self._reassign_small_clusters(prioritized_genes, features_scaled)

        return self.clusters

    def cluster_by_function(
        self,
        prioritized_genes: List[PrioritizedGene],
        gene_pathways: Dict[str, List[str]],
    ) -> List[GeneCluster]:
        """Cluster genes based on functional annotations."""
        # Group genes by shared pathways
        pathway_groups: Dict[str, List[PrioritizedGene]] = {}

        for gene in prioritized_genes:
            gene_key = gene.gene_symbol or gene.gene_id
            pathways = gene_pathways.get(gene_key, ["unassigned"])

            for pathway in pathways:
                if pathway not in pathway_groups:
                    pathway_groups[pathway] = []
                pathway_groups[pathway].append(gene)

        # Create clusters from pathway groups
        clusters = []
        for i, (pathway, genes) in enumerate(pathway_groups.items()):
            if len(genes) >= self.min_cluster_size:
                clusters.append(GeneCluster(cluster_id=i, genes=genes))

        # Handle unassigned genes
        unassigned = []
        for gene in prioritized_genes:
            if not any(gene in cluster.genes for cluster in clusters):
                unassigned.append(gene)

        if unassigned:
            clusters.append(GeneCluster(cluster_id=len(clusters), genes=unassigned))

        self.clusters = clusters
        return self.clusters

    def _create_simple_features(self, genes: List[PrioritizedGene]) -> np.ndarray:
        """Create simple feature matrix from DE results."""
        features = []

        for gene in genes:
            gene_features = [
                gene.de_result.log2_fold_change,
                -np.log10(gene.de_result.padj + 1e-300),
                gene.statistical_score,
                gene.biological_score,
            ]

            # Add expression level if available
            if gene.de_result.base_mean is not None:
                gene_features.append(np.log10(gene.de_result.base_mean + 1))
            else:
                gene_features.append(0)

            features.append(gene_features)

        return np.array(features)

    def _estimate_n_clusters(self, features: np.ndarray, n_genes: int) -> int:
        """Estimate optimal number of clusters."""
        # Simple heuristic: sqrt(n/2), bounded between 3 and 10
        estimated = int(np.sqrt(n_genes / 2))
        return max(3, min(estimated, 10))

    def _create_clusters(
        self, genes: List[PrioritizedGene], labels: np.ndarray
    ) -> List[GeneCluster]:
        """Create cluster objects from labels."""
        clusters = []

        for cluster_id in np.unique(labels):
            cluster_genes = [g for g, l in zip(genes, labels) if l == cluster_id]
            clusters.append(
                GeneCluster(cluster_id=int(cluster_id), genes=cluster_genes)
            )

        return clusters

    def _reassign_small_clusters(
        self, genes: List[PrioritizedGene], features: np.ndarray
    ) -> None:
        """Reassign genes from small clusters to nearest large cluster."""
        # This is a simplified implementation
        # In practice, would calculate distances to cluster centroids
        pass

    def get_cluster_summary(self) -> List[Dict[str, any]]:
        """Get summary of all clusters."""
        return [cluster.get_summary() for cluster in self.clusters]
