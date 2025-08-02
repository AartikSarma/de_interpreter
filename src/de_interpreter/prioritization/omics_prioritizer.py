"""Omics-agnostic feature prioritization for multi-omics analysis."""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

from ..parsers.omics_data import OmicsFeature, OmicsExperimentContext, OmicsType


@dataclass
class PrioritizedOmicsFeature:
    """A prioritized omics feature with scoring."""

    omics_feature: OmicsFeature
    statistical_score: float
    biological_score: float
    combined_score: float
    rank: Optional[int] = None

    @property
    def feature_id(self) -> str:
        return self.omics_feature.feature_id

    @property
    def feature_symbol(self) -> Optional[str]:
        return self.omics_feature.feature_symbol

    @property
    def display_name(self) -> str:
        return self.omics_feature.display_name

    @property
    def omics_type(self) -> OmicsType:
        return self.omics_feature.omics_type


class OmicsPrioritizer:
    """Prioritize omics features based on statistical and biological importance."""

    def __init__(
        self,
        padj_threshold: float = 0.05,
        log2fc_threshold: float = 1.0,
        top_n: int = 100,
    ):
        self.padj_threshold = padj_threshold
        self.log2fc_threshold = log2fc_threshold
        self.top_n = top_n
        self.prioritized_features: List[PrioritizedOmicsFeature] = []

    def prioritize(
        self,
        omics_features: List[OmicsFeature],
        context: Optional[OmicsExperimentContext] = None,
        known_features: Optional[Dict[str, float]] = None,
    ) -> List[PrioritizedOmicsFeature]:
        """Prioritize omics features based on multiple criteria."""
        
        # Filter features by significance and effect size
        filtered_features = self._filter_features(omics_features)
        
        if not filtered_features:
            return []

        # Calculate scores
        prioritized = []
        for feature in filtered_features:
            statistical_score = self._calculate_statistical_score(feature)
            biological_score = self._calculate_biological_score(feature, context, known_features)
            combined_score = self._combine_scores(statistical_score, biological_score)

            prioritized_feature = PrioritizedOmicsFeature(
                omics_feature=feature,
                statistical_score=statistical_score,
                biological_score=biological_score,
                combined_score=combined_score,
            )
            prioritized.append(prioritized_feature)

        # Sort by combined score and assign ranks
        prioritized.sort(key=lambda x: x.combined_score, reverse=True)
        for i, feature in enumerate(prioritized):
            feature.rank = i + 1

        # Return top N
        self.prioritized_features = prioritized[: self.top_n]
        return self.prioritized_features

    def _filter_features(self, features: List[OmicsFeature]) -> List[OmicsFeature]:
        """Filter features by significance and effect size."""
        filtered = []
        
        for feature in features:
            # Check significance
            if not feature.is_significant_at_threshold(self.padj_threshold):
                continue
                
            # Check effect size (use omics-appropriate thresholds)
            min_effect_size = self._get_min_effect_size(feature.omics_type)
            if abs(feature.log2_fold_change) < min_effect_size:
                continue
                
            filtered.append(feature)
        
        return filtered

    def _get_min_effect_size(self, omics_type: OmicsType) -> float:
        """Get minimum effect size threshold for different omics types."""
        # Different omics have different typical effect sizes
        thresholds = {
            OmicsType.TRANSCRIPTOMICS: 1.0,    # 2-fold change
            OmicsType.PROTEOMICS: 0.58,       # 1.5-fold change (less dynamic range)
            OmicsType.METABOLOMICS: 0.58,     # 1.5-fold change
            OmicsType.GENOMICS: 0.1,          # Effect sizes often smaller
            OmicsType.METAGENOMICS: 1.0,      # 2-fold change
            OmicsType.EPIGENOMICS: 0.5,       # Moderate changes
            OmicsType.LIPIDOMICS: 0.58,       # 1.5-fold change
        }
        return thresholds.get(omics_type, self.log2fc_threshold)

    def _calculate_statistical_score(self, feature: OmicsFeature) -> float:
        """Calculate statistical significance score."""
        # Combine p-value and effect size
        neg_log_padj = -np.log10(feature.padj + 1e-300)  # Avoid log(0)
        abs_log2fc = abs(feature.log2_fold_change)
        
        # Weight by effect size (larger changes get higher scores)
        statistical_score = neg_log_padj * (1 + abs_log2fc / 5.0)
        
        return min(statistical_score, 100.0)  # Cap at 100

    def _calculate_biological_score(
        self,
        feature: OmicsFeature,
        context: Optional[OmicsExperimentContext],
        known_features: Optional[Dict[str, float]],
    ) -> float:
        """Calculate biological relevance score."""
        score = 1.0  # Base score
        
        # Check if feature is in known disease/pathway list
        if known_features:
            feature_key = feature.feature_symbol or feature.feature_id
            if feature_key in known_features:
                score += known_features[feature_key]
        
        # Omics-specific biological scoring
        score += self._calculate_omics_specific_score(feature, context)
        
        # Annotation-based scoring
        score += self._calculate_annotation_score(feature)
        
        return min(score, 10.0)  # Cap at 10

    def _calculate_omics_specific_score(
        self, 
        feature: OmicsFeature, 
        context: Optional[OmicsExperimentContext]
    ) -> float:
        """Calculate omics-type specific biological scores."""
        score = 0.0
        
        if feature.omics_type == OmicsType.TRANSCRIPTOMICS:
            # Higher scores for well-characterized genes
            if feature.feature_symbol and len(feature.feature_symbol) <= 6:
                score += 0.5  # Likely official gene symbol
        
        elif feature.omics_type == OmicsType.PROTEOMICS:
            # Higher scores for proteins with functional domains
            if "domain" in feature.annotations:
                score += 0.5
        
        elif feature.omics_type == OmicsType.METABOLOMICS:
            # Higher scores for central metabolism
            pathways = feature.annotations.get("pathway", "").lower()
            if any(term in pathways for term in ["glycolysis", "tca", "fatty acid"]):
                score += 1.0
        
        elif feature.omics_type == OmicsType.METAGENOMICS:
            # Higher scores for well-characterized species
            if feature.annotations.get("species"):
                score += 0.5
        
        return score

    def _calculate_annotation_score(self, feature: OmicsFeature) -> float:
        """Score based on functional annotations."""
        score = 0.0
        
        # Pathway annotations
        pathways = feature.annotations.get("pathway", "")
        if pathways:
            score += 0.5
        
        # GO term annotations
        go_terms = feature.annotations.get("go_term", "")
        if go_terms:
            score += 0.3
        
        # Disease-relevant annotations
        disease_terms = ["cancer", "diabetes", "alzheimer", "parkinson", "immune"]
        annotation_text = str(feature.annotations).lower()
        if any(term in annotation_text for term in disease_terms):
            score += 1.0
        
        return score

    def _combine_scores(self, statistical_score: float, biological_score: float) -> float:
        """Combine statistical and biological scores."""
        # Weight statistical significance more heavily, but include biological relevance
        combined = (0.7 * statistical_score) + (0.3 * biological_score)
        return combined

    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics of prioritized features."""
        if not self.prioritized_features:
            return {"upregulated": 0, "downregulated": 0, "total": 0}

        upregulated = sum(
            1 for f in self.prioritized_features if f.omics_feature.is_upregulated
        )
        downregulated = sum(
            1 for f in self.prioritized_features if f.omics_feature.is_downregulated
        )

        return {
            "upregulated": upregulated,
            "downregulated": downregulated,
            "total": len(self.prioritized_features),
            "mean_statistical_score": np.mean([f.statistical_score for f in self.prioritized_features]),
            "mean_biological_score": np.mean([f.biological_score for f in self.prioritized_features]),
            "mean_combined_score": np.mean([f.combined_score for f in self.prioritized_features]),
        }

    def get_omics_specific_summary(self) -> Dict[str, any]:
        """Get omics-specific summary information."""
        if not self.prioritized_features:
            return {}
        
        omics_type = self.prioritized_features[0].omics_type
        feature_type = omics_type.value
        
        # Calculate omics-specific metrics
        log2fcs = [f.omics_feature.log2_fold_change for f in self.prioritized_features]
        padjs = [f.omics_feature.padj for f in self.prioritized_features]
        
        summary = {
            "omics_type": feature_type,
            "feature_type_name": self._get_feature_type_name(omics_type),
            "mean_log2fc": np.mean(log2fcs),
            "median_log2fc": np.median(log2fcs),
            "max_abs_log2fc": np.max(np.abs(log2fcs)),
            "min_padj": np.min(padjs),
            "max_padj": np.max(padjs),
        }
        
        return summary

    def _get_feature_type_name(self, omics_type: OmicsType) -> str:
        """Get human-readable feature type name."""
        names = {
            OmicsType.TRANSCRIPTOMICS: "genes",
            OmicsType.GENOMICS: "genetic variants",
            OmicsType.PROTEOMICS: "proteins", 
            OmicsType.METABOLOMICS: "metabolites",
            OmicsType.METAGENOMICS: "microbial taxa",
            OmicsType.EPIGENOMICS: "genomic regions",
            OmicsType.LIPIDOMICS: "lipids"
        }
        return names.get(omics_type, "features")