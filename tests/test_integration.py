"""Integration tests that don't require API calls."""

import sys
sys.path.insert(0, '/Users/aartiksarma/Projects/de_interpreter/src')

import pytest
from pathlib import Path
from de_interpreter.parsers import DEParser, MetadataParser
from de_interpreter.prioritization import GenePrioritizer, GeneClusterer
from de_interpreter.literature import LiteratureCache
from de_interpreter.reporting import ReportGenerator, MarkdownFormatter


class TestIntegration:
    """Test integration between modules."""
    
    def test_full_pipeline_setup(self, tmp_path):
        """Test that the full pipeline can be set up without API calls."""
        # Create sample data
        sample_data = {
            'gene_id': ['GENE1', 'GENE2', 'GENE3', 'GENE4', 'GENE5'],
            'gene_symbol': ['G1', 'G2', 'G3', 'G4', 'G5'],
            'log2FoldChange': [2.0, -1.5, 1.8, -2.2, 1.2],
            'pvalue': [0.001, 0.002, 0.003, 0.001, 0.005],
            'padj': [0.01, 0.02, 0.03, 0.01, 0.04],
            'baseMean': [100, 200, 150, 300, 80]
        }
        
        import pandas as pd
        df = pd.DataFrame(sample_data)
        de_file = tmp_path / "test_de.csv"
        df.to_csv(de_file, index=False)
        
        # Create metadata
        metadata = {
            'disease': 'Test Disease',
            'tissue': 'test_tissue',
            'treatment': 'test_treatment',
            'control': 'control',
            'organism': 'human'
        }
        
        import json
        meta_file = tmp_path / "test_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f)
        
        # Test pipeline components
        de_parser = DEParser()
        meta_parser = MetadataParser()
        
        # Parse data
        de_results = de_parser.parse(de_file)
        context = meta_parser.parse(meta_file)
        
        assert len(de_results) == 5
        assert context.disease == 'Test Disease'
        
        # Test prioritization
        prioritizer = GenePrioritizer(top_n=3)
        prioritized = prioritizer.prioritize(de_results, context)
        
        assert len(prioritized) == 3
        assert all(p.combined_score > 0 for p in prioritized)
        
        # Test clustering (with small min_cluster_size for test)
        clusterer = GeneClusterer(n_clusters=2, min_cluster_size=1)
        clusters = clusterer.cluster_by_expression(prioritized)
        
        assert len(clusters) >= 1
        
        # Test cache
        cache = LiteratureCache()
        assert cache is not None
        
        # Test report generation components
        formatter = MarkdownFormatter()
        report_gen = ReportGenerator(output_dir=tmp_path)
        
        # Test markdown formatting
        table = formatter.format_table(['Gene', 'Log2FC'], [['G1', '2.0'], ['G2', '-1.5']])
        assert 'Gene' in table
        assert 'Log2FC' in table
        
        print("✅ All pipeline components initialized successfully!")
        
    def test_prioritization_scoring(self):
        """Test gene prioritization scoring."""
        from de_interpreter.parsers import DEResult
        from de_interpreter.prioritization import PrioritizedGene
        
        # Create mock DE results
        de_result = DEResult(
            gene_id='TEST1',
            gene_symbol='TEST',
            log2_fold_change=2.5,
            p_value=0.001,
            padj=0.01,
            base_mean=200.0
        )
        
        # Test prioritized gene creation
        prioritized = PrioritizedGene(
            de_result=de_result,
            statistical_score=0.8,
            biological_score=0.6,
            combined_score=0.74,
            rank=1
        )
        
        assert prioritized.gene_id == 'TEST1'
        assert prioritized.gene_symbol == 'TEST'
        assert prioritized.combined_score == 0.74
        assert prioritized.rank == 1