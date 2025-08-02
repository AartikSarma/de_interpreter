"""Tests for input parsers."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import json

import sys
sys.path.insert(0, '/Users/aartiksarma/Projects/de_interpreter/src')

from de_interpreter.parsers import DEParser, DEResult, MetadataParser, ExperimentalContext


class TestDEParser:
    """Test differential expression parser."""
    
    def test_parse_csv(self, tmp_path):
        """Test parsing CSV file."""
        # Create test data
        data = {
            'gene_id': ['GENE1', 'GENE2', 'GENE3'],
            'gene_symbol': ['G1', 'G2', 'G3'],
            'log2FoldChange': [2.5, -1.8, 0.3],
            'pvalue': [0.001, 0.01, 0.5],
            'padj': [0.01, 0.05, 0.8],
            'baseMean': [100, 200, 50]
        }
        df = pd.DataFrame(data)
        
        # Save to CSV
        csv_path = tmp_path / "test_de.csv"
        df.to_csv(csv_path, index=False)
        
        # Parse
        parser = DEParser()
        results = parser.parse(csv_path)
        
        # Verify
        assert len(results) == 3
        assert results[0].gene_id == 'GENE1'
        assert results[0].gene_symbol == 'G1'
        assert results[0].log2_fold_change == 2.5
        assert results[0].is_upregulated
        assert results[0].is_significant
    
    def test_column_mapping(self, tmp_path):
        """Test alternative column names."""
        data = {
            'ensembl_id': ['ENSG001', 'ENSG002'],
            'log2FC': [1.5, -2.0],
            'p_value': [0.01, 0.001],
            'FDR': [0.05, 0.01],
            'mean_expression': [150, 300]
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "test_alt_cols.csv"
        df.to_csv(csv_path, index=False)
        
        parser = DEParser()
        results = parser.parse(csv_path)
        
        assert len(results) == 2
        assert results[0].gene_id == 'ENSG001'
        assert results[0].log2_fold_change == 1.5
        assert results[0].padj == 0.05
    
    def test_significant_genes(self, tmp_path):
        """Test filtering significant genes."""
        data = {
            'gene_id': ['G1', 'G2', 'G3', 'G4'],
            'log2FoldChange': [2.0, 0.5, -2.5, 1.5],
            'pvalue': [0.001, 0.1, 0.001, 0.01],
            'padj': [0.01, 0.3, 0.01, 0.06]
        }
        df = pd.DataFrame(data)
        
        csv_path = tmp_path / "test_sig.csv"
        df.to_csv(csv_path, index=False)
        
        parser = DEParser()
        results = parser.parse(csv_path)
        
        sig_genes = parser.get_significant_genes(padj_threshold=0.05, log2fc_threshold=1.0)
        
        assert len(sig_genes) == 2
        assert sig_genes[0].gene_id == 'G1'
        assert sig_genes[1].gene_id == 'G3'


class TestMetadataParser:
    """Test metadata parser."""
    
    def test_parse_json(self, tmp_path):
        """Test parsing JSON metadata."""
        metadata = {
            'disease': 'Cancer',
            'tissue': 'lung',
            'cell_type': 'epithelial',
            'treatment': 'drug_A',
            'control': 'DMSO',
            'organism': 'human'
        }
        
        json_path = tmp_path / "metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f)
        
        parser = MetadataParser()
        context = parser.parse(json_path)
        
        assert context.disease == 'Cancer'
        assert context.tissue == 'lung'
        assert context.cell_type == 'epithelial'
        assert context.treatment == 'drug_A'
    
    def test_context_string(self):
        """Test context string generation."""
        context = ExperimentalContext(
            disease="Alzheimer's disease",
            tissue="brain",
            cell_type="neurons",
            treatment="amyloid-beta",
            control="vehicle",
            time_point="24h"
        )
        
        context_str = context.get_context_string()
        
        assert "amyloid-beta vs vehicle" in context_str
        assert "Alzheimer's disease" in context_str
        assert "brain" in context_str
        assert "24h" in context_str
    
    def test_search_terms(self):
        """Test search term generation."""
        context = ExperimentalContext(
            disease="Parkinson's disease",
            tissue="substantia nigra",
            cell_type="dopaminergic neurons",
            treatment="MPTP"
        )
        
        terms = context.get_search_terms()
        
        assert "Parkinson's disease" in terms
        assert "substantia nigra" in terms
        assert "dopaminergic neurons" in terms
        assert "MPTP" in terms