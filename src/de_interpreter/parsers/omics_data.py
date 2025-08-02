"""Omics-agnostic data structures for multi-omics analysis."""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pandas as pd
from dataclasses import dataclass
import numpy as np


class OmicsType(Enum):
    """Supported omics data types."""
    TRANSCRIPTOMICS = "transcriptomics"
    GENOMICS = "genomics" 
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    METAGENOMICS = "metagenomics"
    EPIGENOMICS = "epigenomics"
    LIPIDOMICS = "lipidomics"


@dataclass
class OmicsFeature:
    """Container for a single omics feature result."""

    feature_id: str                    # Primary identifier (gene, protein, metabolite, etc.)
    feature_symbol: Optional[str]      # Human-readable symbol/name
    feature_name: Optional[str]        # Full feature name
    omics_type: OmicsType             # Type of omics data
    
    # Core differential analysis metrics
    log2_fold_change: float           # Log2 fold change (or effect size)
    p_value: float                    # Raw p-value
    padj: float                       # Adjusted p-value
    
    # Additional metrics (optional)
    base_mean: Optional[float] = None          # Base expression/abundance
    effect_size: Optional[float] = None       # Alternative effect size metric
    confidence_interval: Optional[tuple] = None  # CI for fold change
    
    # Omics-specific annotations
    annotations: Dict[str, Any] = None        # Pathway, GO terms, etc.
    
    def __post_init__(self):
        if self.annotations is None:
            self.annotations = {}

    @property
    def is_upregulated(self) -> bool:
        """Check if feature is upregulated."""
        return self.log2_fold_change > 0

    @property
    def is_downregulated(self) -> bool:
        """Check if feature is downregulated."""
        return self.log2_fold_change < 0

    @property
    def is_significant(self) -> bool:
        """Check if feature is statistically significant."""
        return self.padj < 0.05

    def is_significant_at_threshold(self, padj_threshold: float = 0.05) -> bool:
        """Check significance at custom threshold."""
        return self.padj < padj_threshold

    @property
    def fold_change(self) -> float:
        """Calculate absolute fold change."""
        return 2 ** abs(self.log2_fold_change)

    @property
    def display_name(self) -> str:
        """Get the best display name for this feature."""
        return self.feature_symbol or self.feature_name or self.feature_id

    def to_search_terms(self) -> List[str]:
        """Generate search terms for literature mining."""
        terms = [self.feature_id]
        
        if self.feature_symbol:
            terms.append(self.feature_symbol)
        if self.feature_name and self.feature_name != self.feature_symbol:
            terms.append(self.feature_name)
            
        return terms


class OmicsColumnMapping:
    """Column mappings for different omics types."""
    
    # Common mappings across all omics
    COMMON_MAPPINGS = {
        "log2FoldChange": [
            "log2FC", "logFC", "log2_fc", "l2fc", "log2fold_change", 
            "log2_fold_change", "fold_change", "fc", "log2ratio"
        ],
        "pvalue": [
            "p_value", "p.value", "PValue", "P", "pval", "p_val"
        ],
        "padj": [
            "p_adj", "p.adj", "padjust", "q_value", "qvalue", "fdr", "FDR", 
            "adj_p", "adjusted_p", "p_adjusted"
        ],
        "baseMean": [
            "base_mean", "mean_expression", "avg_expr", "AveExpr", "baseMean",
            "mean_abundance", "base_abundance", "intensity"
        ]
    }
    
    # Omics-specific feature ID mappings
    FEATURE_ID_MAPPINGS = {
        OmicsType.TRANSCRIPTOMICS: [
            "gene_id", "gene", "ensembl_id", "ENSEMBL", "ID", "transcript_id",
            "feature_id", "gene_symbol", "symbol"
        ],
        OmicsType.GENOMICS: [
            "variant_id", "snp_id", "rsid", "chr_pos", "variant", "mutation_id",
            "genomic_coordinate", "locus"
        ],
        OmicsType.PROTEOMICS: [
            "protein_id", "uniprot_id", "protein_accession", "protein_symbol",
            "protein_name", "peptide_id", "accession"
        ],
        OmicsType.METABOLOMICS: [
            "metabolite_id", "compound_id", "chemical_id", "pubchem_id", 
            "kegg_id", "hmdb_id", "metabolite_name", "compound_name"
        ],
        OmicsType.METAGENOMICS: [
            "species", "taxon_id", "otu_id", "asv_id", "taxonomic_id",
            "organism", "taxa", "microbe_id"
        ],
        OmicsType.EPIGENOMICS: [
            "region_id", "peak_id", "dmr_id", "cpg_id", "chromatin_region",
            "genomic_region", "coordinate"
        ],
        OmicsType.LIPIDOMICS: [
            "lipid_id", "lipid_name", "lipid_class", "fatty_acid", "lipid_species"
        ]
    }
    
    # Omics-specific feature symbol mappings
    FEATURE_SYMBOL_MAPPINGS = {
        OmicsType.TRANSCRIPTOMICS: [
            "gene_symbol", "symbol", "gene_name", "SYMBOL", "GENE", "name"
        ],
        OmicsType.GENOMICS: [
            "gene_symbol", "nearest_gene", "associated_gene", "symbol"
        ],
        OmicsType.PROTEOMICS: [
            "protein_symbol", "gene_symbol", "protein_name", "symbol", "name"
        ],
        OmicsType.METABOLOMICS: [
            "metabolite_name", "compound_name", "chemical_name", "name", "common_name"
        ],
        OmicsType.METAGENOMICS: [
            "species_name", "organism_name", "taxonomic_name", "taxa_name"
        ],
        OmicsType.EPIGENOMICS: [
            "region_name", "peak_name", "associated_gene", "gene_symbol"
        ],
        OmicsType.LIPIDOMICS: [
            "lipid_name", "lipid_symbol", "common_name", "systematic_name"
        ]
    }

    @classmethod
    def get_mappings_for_omics(cls, omics_type: OmicsType) -> Dict[str, List[str]]:
        """Get all column mappings for a specific omics type."""
        mappings = cls.COMMON_MAPPINGS.copy()
        
        # Add omics-specific feature ID mappings
        mappings["feature_id"] = cls.FEATURE_ID_MAPPINGS.get(omics_type, [])
        mappings["feature_symbol"] = cls.FEATURE_SYMBOL_MAPPINGS.get(omics_type, [])
        
        return mappings


@dataclass 
class OmicsExperimentContext:
    """Extended experimental context for multi-omics studies."""
    
    # Basic experiment info
    omics_type: OmicsType
    disease: str
    tissue: str
    cell_type: str
    treatment: str
    control: str
    organism: str
    comparison_description: str
    
    # Study design
    sample_size: Optional[Dict[str, int]] = None
    time_point: Optional[str] = None
    
    # Omics-specific context
    platform: Optional[str] = None              # Sequencing/MS platform
    analysis_method: Optional[str] = None       # Analysis pipeline used
    normalization: Optional[str] = None        # Normalization method
    
    # Additional metadata
    additional_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}
    
    def get_context_string(self) -> str:
        """Generate human-readable context string."""
        return self.comparison_description
    
    def get_omics_context(self) -> str:
        """Get omics-specific context for prompts."""
        omics_contexts = {
            OmicsType.TRANSCRIPTOMICS: "gene expression",
            OmicsType.GENOMICS: "genetic variation", 
            OmicsType.PROTEOMICS: "protein abundance",
            OmicsType.METABOLOMICS: "metabolite levels",
            OmicsType.METAGENOMICS: "microbial abundance",
            OmicsType.EPIGENOMICS: "epigenetic modifications",
            OmicsType.LIPIDOMICS: "lipid composition"
        }
        return omics_contexts.get(self.omics_type, "molecular abundance")
    
    def get_feature_type_name(self) -> str:
        """Get the appropriate feature type name for this omics."""
        feature_names = {
            OmicsType.TRANSCRIPTOMICS: "gene",
            OmicsType.GENOMICS: "variant",
            OmicsType.PROTEOMICS: "protein", 
            OmicsType.METABOLOMICS: "metabolite",
            OmicsType.METAGENOMICS: "microbe",
            OmicsType.EPIGENOMICS: "genomic region",
            OmicsType.LIPIDOMICS: "lipid"
        }
        return feature_names.get(self.omics_type, "feature")