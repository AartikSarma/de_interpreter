"""Integration tests for the complete pipeline."""

import pytest
import asyncio
from pathlib import Path

from de_interpreter.main import SimplifiedPipeline, AnalysisConfig


@pytest.mark.asyncio
class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    async def test_basic_pipeline_without_api(self, create_test_files, basic_config, temp_dir):
        """Test basic pipeline without API calls."""
        # Create test files
        csv_file, metadata_file = create_test_files()
        
        # Update config to use temp directory
        config = AnalysisConfig(
            max_features=2,
            use_cache=False,
            output_dir=str(temp_dir),
            anthropic_api_key=None,  # No API key - should use basic discussions
            progress_callback=None,
            use_scoring=False
        )
        
        pipeline = SimplifiedPipeline(config)
        
        # Run analysis
        report_path = await pipeline.run_analysis(
            de_file=str(csv_file),
            metadata_file=str(metadata_file),
            output_name="test_integration"
        )
        
        # Verify report was created
        assert report_path.exists()
        assert report_path.name == "test_integration.md"
        
        # Check report contains expected content
        content = report_path.read_text()
        assert "## Executive Summary" in content
        assert "TP53" in content or "BRCA1" in content or "MYC" in content
    
    async def test_pipeline_with_mock_scoring(self, create_test_files, temp_dir, mock_progress_callback):
        """Test pipeline with scoring enabled (mocked)."""
        # Create test files
        csv_file, metadata_file = create_test_files()
        
        config = AnalysisConfig(
            max_features=2,
            use_cache=False,
            output_dir=str(temp_dir),
            anthropic_api_key=None,
            progress_callback=mock_progress_callback,
            use_scoring=True,
            scorer_type="tfidf"
        )
        
        pipeline = SimplifiedPipeline(config)
        
        # Run analysis
        report_path = await pipeline.run_analysis(
            de_file=str(csv_file),
            metadata_file=str(metadata_file),
            output_name="test_scoring"
        )
        
        # Verify report was created
        assert report_path.exists()
        
        # Check progress messages were captured
        assert len(mock_progress_callback.messages) > 0
        messages = [msg for msg, pct in mock_progress_callback.messages]
        assert any("Parsing input data" in msg for msg in messages)
        assert any("Prioritizing features" in msg for msg in messages)
    
    async def test_parser_integration(self, create_test_files):
        """Test parser integration with different data formats."""
        from de_interpreter.parsers.parser import OmicsParser
        
        # Test with standard format
        csv_file, metadata_file = create_test_files()
        
        parser = OmicsParser()
        features, context = parser.parse(str(csv_file), str(metadata_file))
        
        # Verify parsing results
        assert len(features) == 3  # Should parse all 3 test genes
        assert context.disease == "cancer"
        assert context.tissue == "tumor"
        
        # Check feature data
        gene_symbols = [f.feature_symbol for f in features]
        assert "TP53" in gene_symbols
        assert "BRCA1" in gene_symbols
        assert "MYC" in gene_symbols
    
    async def test_prioritizer_integration(self, create_test_files):
        """Test prioritizer integration."""
        from de_interpreter.parsers.parser import OmicsParser
        from de_interpreter.prioritization.prioritizer import OmicsPrioritizer
        
        # Parse test data
        csv_file, metadata_file = create_test_files()
        parser = OmicsParser()
        features, context = parser.parse(str(csv_file), str(metadata_file))
        
        # Prioritize features
        prioritizer = OmicsPrioritizer()
        prioritized = prioritizer.prioritize(features, context, max_features=2)
        
        # Verify prioritization
        assert len(prioritized) <= 2
        assert all(hasattr(pf, 'combined_score') for pf in prioritized)
        assert all(hasattr(pf, 'feature') for pf in prioritized)
        
        # Check that features are sorted by score (descending)
        if len(prioritized) > 1:
            scores = [pf.combined_score for pf in prioritized]
            assert scores == sorted(scores, reverse=True)
    
    async def test_report_generation(self, create_test_files, temp_dir):
        """Test report generation."""
        from de_interpreter.reporting.report_generator import ReportGenerator
        from de_interpreter.parsers.parser import OmicsParser
        from de_interpreter.prioritization.prioritizer import OmicsPrioritizer
        
        # Setup test data
        csv_file, metadata_file = create_test_files()
        parser = OmicsParser()
        features, context = parser.parse(str(csv_file), str(metadata_file))
        
        prioritizer = OmicsPrioritizer()
        prioritized = prioritizer.prioritize(features, context, max_features=2)
        
        # Generate report
        report_generator = ReportGenerator(temp_dir)
        
        # Create basic feature discussions
        from de_interpreter.synthesis.synthesizer import FeatureDiscussion
        discussions = []
        for pf in prioritized:
            discussion = FeatureDiscussion(
                feature_id=pf.feature.feature_id,
                feature_symbol=pf.feature.feature_symbol,
                discussion_text=f"Test discussion for {pf.feature.feature_symbol}",
                key_findings=[f"Significant change: {pf.feature.log2_fold_change:.2f}"],
                citations=[],
                therapeutic_implications="Test implications"
            )
            discussions.append(discussion)
        
        # Generate summary data
        feature_summary = {
            "total_features": len(features),
            "significant_features": len([f for f in features if f.padj < 0.05])
        }
        
        omics_summary = {
            "upregulated": len([f for f in features if f.log2_fold_change > 0]),
            "downregulated": len([f for f in features if f.log2_fold_change < 0])
        }
        
        report_path = report_generator.generate_omics_report(
            executive_summary="Test executive summary",
            feature_discussions=discussions,
            context=context,
            feature_summary=feature_summary,
            omics_summary=omics_summary,
            analysis_features=prioritized,
            output_name="test_report"
        )
        
        # Verify report
        assert report_path.exists()
        content = report_path.read_text()
        assert "Test executive summary" in content
        assert "TP53" in content or "BRCA1" in content or "MYC" in content