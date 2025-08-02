"""Parser for differential expression results."""

from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
from dataclasses import dataclass
import numpy as np


@dataclass
class DEResult:
    """Container for a single differential expression result."""

    gene_id: str
    gene_symbol: Optional[str]
    log2_fold_change: float
    p_value: float
    padj: float
    base_mean: Optional[float] = None
    gene_name: Optional[str] = None

    @property
    def is_upregulated(self) -> bool:
        return self.log2_fold_change > 0

    @property
    def is_significant(self) -> bool:
        return self.padj < 0.05

    def is_significant_at_threshold(self, padj_threshold: float = 0.05) -> bool:
        return self.padj < padj_threshold

    @property
    def fold_change(self) -> float:
        return 2 ** abs(self.log2_fold_change)


class DEParser:
    """Parser for differential expression result files."""

    REQUIRED_COLUMNS = {"log2FoldChange", "pvalue", "padj"}
    COLUMN_MAPPINGS = {
        "log2FoldChange": ["log2FC", "logFC", "log2_fc", "l2fc", "log2fold_change"],
        "pvalue": ["p_value", "p.value", "PValue", "P"],
        "padj": ["p_adj", "p.adj", "padjust", "q_value", "qvalue", "fdr", "FDR"],
        "gene_id": ["gene", "gene_id", "ensembl_id", "ENSEMBL", "ID"],
        "gene_symbol": ["symbol", "gene_symbol", "gene_name", "SYMBOL", "GENE"],
        "baseMean": ["base_mean", "mean_expression", "avg_expr", "AveExpr"],
    }

    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.results: List[DEResult] = []

    def parse(self, file_path: Path) -> List[DEResult]:
        """Parse DE results from file."""
        self.df = self._read_file(file_path)
        self._validate_columns()
        self._standardize_columns()
        self._clean_data()
        self.results = self._create_results()
        return self.results

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read DE results from various file formats."""
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

        for req_col in self.REQUIRED_COLUMNS:
            found = False
            if req_col in columns:
                found = True
            else:
                # Check alternative names
                for alt in self.COLUMN_MAPPINGS.get(req_col, []):
                    if alt in columns:
                        found = True
                        break

            if not found:
                missing.append(req_col)

        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _standardize_columns(self) -> None:
        """Standardize column names to expected format."""
        rename_map = {}

        for std_name, alternatives in self.COLUMN_MAPPINGS.items():
            for col in self.df.columns:
                if col in alternatives:
                    rename_map[col] = std_name
                    break

        self.df = self.df.rename(columns=rename_map)

        # Extract gene identifier
        if "gene_id" not in self.df.columns:
            # Use index if it looks like gene IDs
            if self.df.index.name and "gene" in self.df.index.name.lower():
                self.df["gene_id"] = self.df.index
            else:
                self.df["gene_id"] = self.df.index.astype(str)

    def _clean_data(self) -> None:
        """Clean and filter data."""
        # Remove rows with missing values in required columns
        required = ["gene_id", "log2FoldChange", "pvalue", "padj"]
        self.df = self.df.dropna(subset=[c for c in required if c in self.df.columns])

        # Convert to numeric
        numeric_cols = ["log2FoldChange", "pvalue", "padj", "baseMean"]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        # Remove invalid entries
        self.df = self.df[self.df["padj"].notna()]
        self.df = self.df[~self.df["log2FoldChange"].isna()]

    def _create_results(self) -> List[DEResult]:
        """Create DEResult objects from dataframe."""
        results = []

        for _, row in self.df.iterrows():
            result = DEResult(
                gene_id=str(row["gene_id"]),
                gene_symbol=row.get("gene_symbol"),
                log2_fold_change=float(row["log2FoldChange"]),
                p_value=float(row["pvalue"]),
                padj=float(row["padj"]),
                base_mean=(
                    float(row["baseMean"])
                    if "baseMean" in row and pd.notna(row["baseMean"])
                    else None
                ),
                gene_name=row.get("gene_name"),
            )
            results.append(result)

        return results

    def get_significant_genes(
        self, padj_threshold: float = 0.05, log2fc_threshold: float = 1.0
    ) -> List[DEResult]:
        """Get significantly differentially expressed genes."""
        return [
            r
            for r in self.results
            if r.padj < padj_threshold and abs(r.log2_fold_change) > log2fc_threshold
        ]

    def summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of DE results."""
        sig_genes = self.get_significant_genes()

        return {
            "total_genes": len(self.results),
            "significant_genes": len(sig_genes),
            "upregulated": sum(1 for g in sig_genes if g.is_upregulated),
            "downregulated": sum(1 for g in sig_genes if not g.is_upregulated),
            "max_log2fc": max((r.log2_fold_change for r in self.results), default=0),
            "min_log2fc": min((r.log2_fold_change for r in self.results), default=0),
            "median_padj": np.median([r.padj for r in self.results]),
        }
