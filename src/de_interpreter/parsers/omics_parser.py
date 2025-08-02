"""Multi-omics parser for differential analysis results."""

from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from .omics_data import OmicsFeature, OmicsType, OmicsColumnMapping, OmicsExperimentContext


class OmicsParser:
    """Parser for multi-omics differential analysis results."""

    def __init__(self, omics_type: OmicsType):
        self.omics_type = omics_type
        self.df: Optional[pd.DataFrame] = None
        self.features: List[OmicsFeature] = []
        self.column_mappings = OmicsColumnMapping.get_mappings_for_omics(omics_type)
        
        # Required columns for differential analysis
        self.required_columns = {"log2FoldChange", "pvalue", "padj"}

    def parse(self, file_path: Path) -> List[OmicsFeature]:
        """Parse omics results from file."""
        self.df = self._read_file(file_path)
        self._validate_columns()
        self._standardize_columns()
        self._clean_data()
        self.features = self._create_features()
        return self.features

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read results from various file formats."""
        suffix = file_path.suffix.lower()

        if suffix in [".csv"]:
            return pd.read_csv(file_path)
        elif suffix in [".tsv", ".txt"]:
            return pd.read_csv(file_path, sep="\t")
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def _validate_columns(self) -> None:
        """Ensure required columns are present."""
        columns = set(self.df.columns)
        missing = []

        for req_col in self.required_columns:
            found = False
            if req_col in columns:
                found = True
            else:
                # Check alternative names
                for alt in self.column_mappings.get(req_col, []):
                    if alt in columns:
                        found = True
                        break

            if not found:
                missing.append(req_col)

        if missing:
            available_cols = list(self.df.columns)
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {available_cols}. "
                f"Expected omics type: {self.omics_type.value}"
            )

    def _standardize_columns(self) -> None:
        """Standardize column names."""
        column_map = {}

        # Map feature identifier columns
        feature_id_col = self._find_column("feature_id")
        if feature_id_col:
            column_map[feature_id_col] = "feature_id"

        feature_symbol_col = self._find_column("feature_symbol")
        if feature_symbol_col:
            column_map[feature_symbol_col] = "feature_symbol"

        # Map standard differential analysis columns
        for standard_col in ["log2FoldChange", "pvalue", "padj", "baseMean"]:
            found_col = self._find_column(standard_col)
            if found_col:
                column_map[found_col] = standard_col

        # Rename columns
        self.df.rename(columns=column_map, inplace=True)

    def _find_column(self, target_col: str) -> Optional[str]:
        """Find column by checking alternative names."""
        columns = self.df.columns
        
        # Check exact match first
        if target_col in columns:
            return target_col
            
        # Check mappings
        for alt in self.column_mappings.get(target_col, []):
            if alt in columns:
                return alt
                
        return None

    def _clean_data(self) -> None:
        """Clean and validate data."""
        # Remove rows with missing critical values
        critical_cols = ["log2FoldChange", "pvalue", "padj"]
        for col in critical_cols:
            if col in self.df.columns:
                self.df = self.df.dropna(subset=[col])

        # Convert numeric columns
        numeric_cols = ["log2FoldChange", "pvalue", "padj", "baseMean"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Remove rows with invalid p-values
        if "pvalue" in self.df.columns:
            self.df = self.df[(self.df["pvalue"] >= 0) & (self.df["pvalue"] <= 1)]
        if "padj" in self.df.columns:
            self.df = self.df[(self.df["padj"] >= 0) & (self.df["padj"] <= 1)]

        # Replace infinite values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(subset=["log2FoldChange"], inplace=True)

    def _create_features(self) -> List[OmicsFeature]:
        """Create OmicsFeature objects from dataframe."""
        features = []

        for _, row in self.df.iterrows():
            # Get feature identifiers
            feature_id = self._get_feature_id(row)
            feature_symbol = row.get("feature_symbol")
            feature_name = self._get_feature_name(row)

            # Core metrics
            log2fc = row["log2FoldChange"]
            pvalue = row["pvalue"]
            padj = row["padj"]
            base_mean = row.get("baseMean")

            # Additional annotations
            annotations = self._extract_annotations(row)

            feature = OmicsFeature(
                feature_id=feature_id,
                feature_symbol=feature_symbol,
                feature_name=feature_name,
                omics_type=self.omics_type,
                log2_fold_change=log2fc,
                p_value=pvalue,
                padj=padj,
                base_mean=base_mean,
                annotations=annotations
            )

            features.append(feature)

        return features

    def _get_feature_id(self, row: pd.Series) -> str:
        """Extract feature ID from row."""
        # Try standardized column first
        if "feature_id" in row and pd.notna(row["feature_id"]):
            return str(row["feature_id"])
        
        # Try omics-specific alternatives
        for col_name in self.column_mappings.get("feature_id", []):
            if col_name in row and pd.notna(row[col_name]):
                return str(row[col_name])
        
        # Fall back to index if no ID found
        return f"feature_{row.name}"

    def _get_feature_name(self, row: pd.Series) -> Optional[str]:
        """Extract feature name from row."""
        # Look for name columns
        name_columns = ["feature_name", "name", "description"]
        for col in name_columns:
            if col in row and pd.notna(row[col]):
                return str(row[col])
        return None

    def _extract_annotations(self, row: pd.Series) -> Dict[str, Any]:
        """Extract additional annotations from row."""
        annotations = {}
        
        # Common annotation columns to capture
        annotation_cols = [
            "pathway", "go_term", "kegg_pathway", "reactome_pathway",
            "protein_class", "metabolite_class", "lipid_class",
            "chromosome", "start", "end", "strand",
            "taxonomy", "kingdom", "phylum", "class", "order", "family", "genus", "species"
        ]
        
        for col in annotation_cols:
            if col in row and pd.notna(row[col]):
                annotations[col] = row[col]
        
        return annotations

    def summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.features:
            return {}

        total_features = len(self.features)
        significant = sum(1 for f in self.features if f.is_significant)
        upregulated = sum(1 for f in self.features if f.is_significant and f.is_upregulated)
        downregulated = sum(1 for f in self.features if f.is_significant and f.is_downregulated)

        log2fcs = [f.log2_fold_change for f in self.features]
        padjs = [f.padj for f in self.features]

        return {
            "total_features": total_features,
            "significant_features": significant,
            "upregulated": upregulated,
            "downregulated": downregulated,
            "mean_log2fc": np.mean(log2fcs),
            "median_log2fc": np.median(log2fcs),
            "min_padj": np.min(padjs),
            "omics_type": self.omics_type.value,
            "feature_type": self._get_feature_type_name()
        }

    def _get_feature_type_name(self) -> str:
        """Get the appropriate feature type name."""
        feature_names = {
            OmicsType.TRANSCRIPTOMICS: "genes",
            OmicsType.GENOMICS: "variants",
            OmicsType.PROTEOMICS: "proteins",
            OmicsType.METABOLOMICS: "metabolites",
            OmicsType.METAGENOMICS: "microbes",
            OmicsType.EPIGENOMICS: "genomic regions",
            OmicsType.LIPIDOMICS: "lipids"
        }
        return feature_names.get(self.omics_type, "features")


class OmicsMetadataParser:
    """Parser for omics experiment metadata."""

    def parse(self, file_path: Path) -> OmicsExperimentContext:
        """Parse metadata from JSON/YAML file."""
        import json
        
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                # Try YAML
                try:
                    import yaml
                    data = yaml.safe_load(f)
                except ImportError:
                    raise ValueError("YAML support requires pyyaml package")

        # Determine omics type
        omics_type_str = data.get("omics_type", "transcriptomics").lower()
        try:
            omics_type = OmicsType(omics_type_str)
        except ValueError:
            # Fall back to transcriptomics if not recognized
            omics_type = OmicsType.TRANSCRIPTOMICS

        return OmicsExperimentContext(
            omics_type=omics_type,
            disease=data.get("disease", ""),
            tissue=data.get("tissue", ""),
            cell_type=data.get("cell_type", ""),
            treatment=data.get("treatment", ""),
            control=data.get("control", ""),
            organism=data.get("organism", "human"),
            comparison_description=data.get("comparison_description", ""),
            sample_size=data.get("sample_size"),
            time_point=data.get("time_point"),
            platform=data.get("platform"),
            analysis_method=data.get("analysis_method"),
            normalization=data.get("normalization"),
            additional_info=data.get("additional_info", {})
        )