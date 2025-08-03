"""Unified prioritizer for all omics types."""

import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np

from ..parsers.omics_data import OmicsFeature, OmicsExperimentContext, OmicsType


@dataclass
class PrioritizedFeature:
    """Container for prioritized feature with scoring information."""
    feature: OmicsFeature
    statistical_score: float
    biological_score: float
    combined_score: float


class OmicsPrioritizer:
    """Unified prioritizer for all omics types."""
    
    def __init__(self):
        # Omics-specific scoring weights
        self.omics_weights = {
            OmicsType.TRANSCRIPTOMICS: {"statistical": 0.6, "biological": 0.4},
            OmicsType.PROTEOMICS: {"statistical": 0.5, "biological": 0.5},
            OmicsType.METABOLOMICS: {"statistical": 0.7, "biological": 0.3},
            OmicsType.GENOMICS: {"statistical": 0.4, "biological": 0.6},
            OmicsType.METAGENOMICS: {"statistical": 0.6, "biological": 0.4},
            OmicsType.EPIGENOMICS: {"statistical": 0.5, "biological": 0.5},
            OmicsType.LIPIDOMICS: {"statistical": 0.7, "biological": 0.3}
        }
    
    def prioritize(
        self,
        features: List[OmicsFeature],
        context: OmicsExperimentContext,
        max_features: Optional[int] = None
    ) -> List[PrioritizedFeature]:
        """Prioritize features based on statistical and biological significance."""
        
        # Filter significant features
        significant_features = [f for f in features if f.padj < 0.05 and abs(f.log2_fold_change) > 1.0]
        
        if not significant_features:
            # If no features meet strict criteria, relax slightly
            significant_features = [f for f in features if f.padj < 0.1 and abs(f.log2_fold_change) > 0.5]
        
        if not significant_features:
            print("Warning: No significant features found with current thresholds")
            return []
        
        prioritized_features = []
        
        for feature in significant_features:
            # Calculate statistical score
            stat_score = self._calculate_statistical_score(feature)
            
            # Calculate biological score
            bio_score = self._calculate_biological_score(feature, context)
            
            # Calculate combined score
            weights = self.omics_weights.get(context.omics_type, {"statistical": 0.6, "biological": 0.4})
            combined_score = (stat_score * weights["statistical"] + bio_score * weights["biological"])
            
            prioritized_features.append(PrioritizedFeature(
                feature=feature,
                statistical_score=stat_score,
                biological_score=bio_score,
                combined_score=combined_score
            ))
        
        # Sort by combined score (descending)
        prioritized_features.sort(key=lambda x: x.combined_score, reverse=True)
        
        # Return max features if specified
        if max_features:
            prioritized_features = prioritized_features[:max_features]
        
        return prioritized_features
    
    def _calculate_statistical_score(self, feature: OmicsFeature) -> float:
        """Calculate statistical significance score."""
        # Convert p-value to -log10 scale
        log_pval = -math.log10(max(feature.padj, 1e-300))  # Avoid log(0)
        
        # Normalize to 0-1 scale (assume max -log10(p) ~ 50)
        pval_score = min(log_pval / 50.0, 1.0)
        
        # Include effect size
        effect_score = min(abs(feature.log2_fold_change) / 10.0, 1.0)  # Normalize by 10 log2FC
        
        # Combine (weighted towards p-value)
        statistical_score = 0.7 * pval_score + 0.3 * effect_score
        
        return min(statistical_score, 1.0)
    
    def _calculate_biological_score(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Calculate biological significance score."""
        # Base score starts at 0.5
        bio_score = 0.5
        
        # Adjust based on omics type and feature characteristics
        if context.omics_type == OmicsType.TRANSCRIPTOMICS:
            bio_score = self._score_transcriptomics_feature(feature, context)
        elif context.omics_type == OmicsType.PROTEOMICS:
            bio_score = self._score_proteomics_feature(feature, context)
        elif context.omics_type == OmicsType.METABOLOMICS:
            bio_score = self._score_metabolomics_feature(feature, context)
        elif context.omics_type == OmicsType.GENOMICS:
            bio_score = self._score_genomics_feature(feature, context)
        elif context.omics_type == OmicsType.METAGENOMICS:
            bio_score = self._score_metagenomics_feature(feature, context)
        elif context.omics_type == OmicsType.EPIGENOMICS:
            bio_score = self._score_epigenomics_feature(feature, context)
        elif context.omics_type == OmicsType.LIPIDOMICS:
            bio_score = self._score_lipidomics_feature(feature, context)
        
        return min(bio_score, 1.0)
    
    def _score_transcriptomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score transcriptomics features (genes)."""
        score = 0.5
        
        # Higher score for known disease genes
        if context.disease and feature.feature_symbol:
            disease_keywords = ["cancer", "tumor", "inflammation", "immune", "disease"]
            if any(keyword in context.disease.lower() for keyword in disease_keywords):
                score += 0.2
        
        # Higher score for well-characterized genes
        if feature.feature_symbol and len(feature.feature_symbol) <= 10:  # Likely official gene symbol
            score += 0.1
        
        # Higher score for larger effect sizes
        if abs(feature.log2_fold_change) > 2:
            score += 0.2
        elif abs(feature.log2_fold_change) > 1.5:
            score += 0.1
        
        return score
    
    def _score_proteomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score proteomics features (proteins)."""
        score = 0.5
        
        # Proteins are directly functional - higher base score
        score += 0.1
        
        # Higher score for secreted/membrane proteins (therapeutic targets)
        if feature.feature_name:
            secreted_keywords = ["secreted", "membrane", "receptor", "kinase", "enzyme"]
            if any(keyword in feature.feature_name.lower() for keyword in secreted_keywords):
                score += 0.2
        
        # Higher score for larger effect sizes
        if abs(feature.log2_fold_change) > 1.5:
            score += 0.2
        elif abs(feature.log2_fold_change) > 1:
            score += 0.1
        
        return score
    
    def _score_metabolomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score metabolomics features (metabolites)."""
        score = 0.5
        
        # Higher score for known biomarkers
        if feature.feature_name:
            biomarker_keywords = ["glucose", "lactate", "amino acid", "fatty acid", "lipid"]
            if any(keyword in feature.feature_name.lower() for keyword in biomarker_keywords):
                score += 0.2
        
        # Metabolites with large changes are often biologically significant
        if abs(feature.log2_fold_change) > 2:
            score += 0.3
        elif abs(feature.log2_fold_change) > 1:
            score += 0.1
        
        return score
    
    def _score_genomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score genomics features (variants, CNVs, etc)."""
        score = 0.5
        
        # Higher score for coding variants
        if feature.feature_name:
            coding_keywords = ["exon", "coding", "missense", "nonsense", "frameshift"]
            if any(keyword in feature.feature_name.lower() for keyword in coding_keywords):
                score += 0.3
        
        # Higher score for known disease-associated genes
        if context.disease and feature.feature_symbol:
            score += 0.2
        
        return score
    
    def _score_metagenomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score metagenomics features (species, genes, pathways)."""
        score = 0.5
        
        # Higher score for pathogenic species
        if feature.feature_name:
            pathogen_keywords = ["pathogen", "virulence", "resistance", "toxin"]
            if any(keyword in feature.feature_name.lower() for keyword in pathogen_keywords):
                score += 0.3
        
        # Higher score for large abundance changes
        if abs(feature.log2_fold_change) > 2:
            score += 0.2
        
        return score
    
    def _score_epigenomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score epigenomics features (methylation, histone marks, etc)."""
        score = 0.5
        
        # Higher score for promoter regions
        if feature.feature_name:
            regulatory_keywords = ["promoter", "enhancer", "cpg", "island", "regulatory"]
            if any(keyword in feature.feature_name.lower() for keyword in regulatory_keywords):
                score += 0.2
        
        # Higher score for larger changes
        if abs(feature.log2_fold_change) > 1:
            score += 0.2
        
        return score
    
    def _score_lipidomics_feature(self, feature: OmicsFeature, context: OmicsExperimentContext) -> float:
        """Score lipidomics features (lipid species)."""
        score = 0.5
        
        # Higher score for bioactive lipids
        if feature.feature_name:
            bioactive_keywords = ["prostaglandin", "leukotriene", "sphingolipid", "ceramide", "cholesterol"]
            if any(keyword in feature.feature_name.lower() for keyword in bioactive_keywords):
                score += 0.3
        
        # Higher score for larger changes
        if abs(feature.log2_fold_change) > 2:
            score += 0.2
        
        return score