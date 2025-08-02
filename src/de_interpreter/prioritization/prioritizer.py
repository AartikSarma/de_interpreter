"""Gene prioritization based on statistical and biological importance."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

from ..parsers import DEResult, ExperimentalContext


@dataclass
class PrioritizedGene:
    """A gene with prioritization scores."""

    de_result: DEResult
    statistical_score: float
    biological_score: float
    combined_score: float
    rank: Optional[int] = None

    @property
    def gene_id(self) -> str:
        return self.de_result.gene_id

    @property
    def gene_symbol(self) -> Optional[str]:
        return self.de_result.gene_symbol


class GenePrioritizer:
    """Prioritize genes based on multiple criteria."""

    def __init__(
        self,
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        top_n: int = 100,
    ):
        self.padj_threshold = padj_threshold
        self.log2fc_threshold = log2fc_threshold
        self.top_n = top_n
        self.prioritized_genes: List[PrioritizedGene] = []

    def prioritize(
        self,
        de_results: List[DEResult],
        context: Optional[ExperimentalContext] = None,
        disease_genes: Optional[Dict[str, float]] = None,
    ) -> List[PrioritizedGene]:
        """Prioritize genes based on statistical and biological importance."""
        # Filter significant genes
        significant = [
            r
            for r in de_results
            if r.padj < self.padj_threshold
            and abs(r.log2_fold_change) > self.log2fc_threshold
        ]

        if not significant:
            print(
                f"Warning: No genes passed significance thresholds. Relaxing criteria..."
            )
            significant = sorted(de_results, key=lambda x: x.padj)[: self.top_n]

        # Score genes
        prioritized = []
        for de_result in significant:
            stat_score = self._calculate_statistical_score(de_result, de_results)
            bio_score = self._calculate_biological_score(de_result, disease_genes)
            combined = self._combine_scores(stat_score, bio_score)

            prioritized.append(
                PrioritizedGene(
                    de_result=de_result,
                    statistical_score=stat_score,
                    biological_score=bio_score,
                    combined_score=combined,
                )
            )

        # Sort by combined score and assign ranks
        prioritized.sort(key=lambda x: x.combined_score, reverse=True)
        for i, gene in enumerate(prioritized):
            gene.rank = i + 1

        # Keep top N
        self.prioritized_genes = prioritized[: self.top_n]
        return self.prioritized_genes

    def _calculate_statistical_score(
        self, gene: DEResult, all_results: List[DEResult]
    ) -> float:
        """Calculate statistical importance score."""
        # Components:
        # 1. Significance (-log10 padj)
        # 2. Effect size (absolute log2FC)
        # 3. Expression level (if available)
        # 4. Consistency (low variance across replicates if available)

        sig_score = -np.log10(gene.padj + 1e-300)  # Add small value to avoid inf
        effect_score = abs(gene.log2_fold_change)

        # Normalize effect size relative to all genes
        all_effects = [abs(r.log2_fold_change) for r in all_results]
        effect_percentile = stats.percentileofscore(all_effects, effect_score) / 100

        # Expression level component
        expr_score = 0.5  # Default if no expression data
        if gene.base_mean is not None and gene.base_mean > 0:
            all_expr = [
                r.base_mean
                for r in all_results
                if r.base_mean is not None and r.base_mean > 0
            ]
            if all_expr:
                expr_percentile = (
                    stats.percentileofscore(all_expr, gene.base_mean) / 100
                )
                expr_score = expr_percentile

        # Combine with weights
        weights = {"significance": 0.4, "effect_size": 0.4, "expression": 0.2}

        # Normalize significance score (cap at 50 for -log10(padj))
        sig_score_norm = min(sig_score / 50, 1.0)

        score = (
            weights["significance"] * sig_score_norm
            + weights["effect_size"] * effect_percentile
            + weights["expression"] * expr_score
        )

        return score

    def _calculate_biological_score(
        self, gene: DEResult, disease_genes: Optional[Dict[str, float]] = None
    ) -> float:
        """Calculate biological importance score."""
        # Default score
        score = 0.5

        if disease_genes:
            # Check if gene is known to be associated with disease
            gene_key = gene.gene_symbol or gene.gene_id
            if gene_key in disease_genes:
                score = disease_genes[gene_key]

        # Could extend with:
        # - Druggability scores
        # - Pathway centrality
        # - Protein-protein interaction degree
        # - Conservation scores

        return score

    def _combine_scores(self, stat_score: float, bio_score: float) -> float:
        """Combine statistical and biological scores."""
        # Weighted combination
        stat_weight = 0.7
        bio_weight = 0.3

        return stat_weight * stat_score + bio_weight * bio_score

    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics of prioritized genes."""
        if not self.prioritized_genes:
            return {}

        up_genes = [g for g in self.prioritized_genes if g.de_result.is_upregulated]
        down_genes = [
            g for g in self.prioritized_genes if not g.de_result.is_upregulated
        ]

        return {
            "total_prioritized": len(self.prioritized_genes),
            "upregulated": len(up_genes),
            "downregulated": len(down_genes),
            "mean_combined_score": np.mean(
                [g.combined_score for g in self.prioritized_genes]
            ),
            "max_log2fc": max(
                g.de_result.log2_fold_change for g in self.prioritized_genes
            ),
            "min_log2fc": min(
                g.de_result.log2_fold_change for g in self.prioritized_genes
            ),
            "top_genes": [
                {
                    "symbol": g.gene_symbol or g.gene_id,
                    "log2fc": g.de_result.log2_fold_change,
                    "padj": g.de_result.padj,
                    "score": g.combined_score,
                }
                for g in self.prioritized_genes[:10]
            ],
        }
