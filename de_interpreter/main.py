"""Simplified main pipeline orchestrator."""

import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any
from dataclasses import dataclass

from .parsers.parser import OmicsParser
from .parsers.omics_data import OmicsExperimentContext, OmicsFeature
from .prioritization.prioritizer import OmicsPrioritizer, PrioritizedFeature
from .literature.pmc_client import PMCClient
from .literature.cache import LiteratureCache
from .synthesis.synthesizer import OmicsSynthesizer, FeatureDiscussion
from .reporting.report_generator import ReportGenerator


@dataclass
class AnalysisConfig:
    """Configuration for analysis pipeline."""
    max_features: int = 25  # Maximum number of features to analyze in detail
    use_cache: bool = True
    cache_dir: str = "cache/literature"
    output_dir: str = "output"
    anthropic_api_key: Optional[str] = None
    progress_callback: Optional[Callable] = None
    # Optional scoring configuration
    use_scoring: bool = False
    scorer_type: str = "tfidf"  # "tfidf", "bm25", "biobert"
    biobert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Optional MeSH enhancement configuration
    use_mesh_enhancement: bool = False
    mesh_terms_count: int = 3


class SimplifiedPipeline:
    """Simplified unified pipeline for all omics types."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.parser = OmicsParser()
        self.prioritizer = OmicsPrioritizer()
        self.cache = LiteratureCache(config.cache_dir) if config.use_cache else None
        self.report_generator = ReportGenerator(Path(config.output_dir))
        
        # Initialize synthesizer if API key provided
        self.synthesizer = None
        if config.anthropic_api_key:
            self.synthesizer = OmicsSynthesizer(
                api_key=config.anthropic_api_key,
                progress_callback=config.progress_callback
            )
    
    async def run_analysis(
        self,
        de_file: str,
        metadata_file: Optional[str] = None,
        output_name: str = "omics_analysis_report"
    ) -> Path:
        """Run complete analysis pipeline."""
        
        try:
            # Step 1: Parse data
            if self.config.progress_callback:
                self.config.progress_callback("Parsing input data...", 5)
            
            features, context = self.parser.parse(de_file, metadata_file)
            print(f"Parsed {len(features)} {context.omics_type.value} features")
            
            # Step 2: Prioritize features
            if self.config.progress_callback:
                self.config.progress_callback("Prioritizing features...", 10)
            
            prioritized_features = self.prioritizer.prioritize(features, context, self.config.max_features)
            analysis_features = prioritized_features[:self.config.max_features]
            print(f"Prioritized {len(prioritized_features)} features, analyzing top {len(analysis_features)}")
            
            # Step 3: Generate summaries
            feature_summary = self._generate_feature_summary(features)
            omics_summary = self._generate_omics_summary(prioritized_features, context)
            
            # Step 4: Literature mining and synthesis (if synthesizer available)
            feature_discussions = []
            executive_summary = ""
            
            if self.synthesizer:
                if self.config.progress_callback:
                    self.config.progress_callback("Searching literature...", 20)
                
                # Literature mining
                async with PMCClient(
                    use_scoring=self.config.use_scoring,
                    scorer_type=self.config.scorer_type,
                    biobert_model=self.config.biobert_model,
                    use_mesh_enhancement=self.config.use_mesh_enhancement,
                    anthropic_api_key=self.config.anthropic_api_key,
                    mesh_terms_count=self.config.mesh_terms_count,
                    progress_callback=self.config.progress_callback
                ) as pmc_client:
                    literature_results = []
                    
                    for feature in analysis_features:
                        query = self._build_literature_query(feature.feature, context)
                        
                        # Check cache first
                        cached_result = None
                        if self.cache:
                            cached_result = self.cache.get(query)
                        
                        if cached_result:
                            literature_results.append(cached_result)
                        else:
                            result = await pmc_client.search(query, limit=5)
                            literature_results.append(result)
                            
                            # Cache result
                            if self.cache:
                                self.cache.set(result)
                
                if self.config.progress_callback:
                    self.config.progress_callback("Synthesizing discussions...", 50)
                
                # Synthesis
                feature_discussions = await self.synthesizer.synthesize_feature_discussions(
                    analysis_features, literature_results, context
                )
                
                executive_summary = await self.synthesizer.generate_executive_summary(
                    feature_discussions, context, feature_summary, omics_summary
                )
            else:
                # Generate basic discussions without synthesis
                feature_discussions = self._generate_basic_discussions(analysis_features, context)
                executive_summary = self._generate_basic_executive_summary(context, feature_summary, omics_summary)
            
            # Step 5: Generate report
            if self.config.progress_callback:
                self.config.progress_callback("Generating report...", 90)
            
            report_path = self.report_generator.generate_omics_report(
                executive_summary=executive_summary,
                feature_discussions=feature_discussions,
                context=context,
                feature_summary=feature_summary,
                omics_summary=omics_summary,
                analysis_features=analysis_features,
                output_name=output_name
            )
            
            if self.config.progress_callback:
                self.config.progress_callback("Analysis complete!", 100)
            
            print(f"Analysis complete! Report saved to: {report_path}")
            return report_path
            
        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            print(error_msg)
            if self.config.progress_callback:
                self.config.progress_callback(error_msg, -1)
            raise
    
    def _generate_feature_summary(self, features: List[OmicsFeature]) -> Dict[str, Any]:
        """Generate feature summary statistics."""
        significant_features = [f for f in features if f.padj < 0.05]
        
        return {
            "total_features": len(features),
            "significant_features": len(significant_features),
            "upregulated": len([f for f in significant_features if f.is_upregulated]),
            "downregulated": len([f for f in significant_features if not f.is_upregulated])
        }
    
    def _generate_omics_summary(self, prioritized_features: List[PrioritizedFeature], context: OmicsExperimentContext) -> Dict[str, Any]:
        """Generate omics-specific summary."""
        significant_features = [pf for pf in prioritized_features if pf.feature.padj < 0.05]
        
        return {
            "feature_type_name": context.get_feature_type_name(),
            "upregulated": len([pf for pf in significant_features if pf.feature.is_upregulated]),
            "downregulated": len([pf for pf in significant_features if not pf.feature.is_upregulated]),
            "top_score": max([pf.combined_score for pf in prioritized_features]) if prioritized_features else 0
        }
    
    def _build_literature_query(self, feature: OmicsFeature, context: OmicsExperimentContext) -> str:
        """Build literature search query."""
        feature_name = feature.display_name
        disease = context.disease or "disease"
        omics_type = context.omics_type.value
        
        return f"{feature_name} AND {disease} AND {omics_type}"
    
    def _generate_basic_discussions(self, features: List[PrioritizedFeature], context: OmicsExperimentContext) -> List[FeatureDiscussion]:
        """Generate basic feature discussions without AI synthesis."""
        discussions = []
        
        for pf in features:
            feature = pf.feature
            direction = "upregulated" if feature.is_upregulated else "downregulated"
            
            discussion_text = f"""
**{feature.display_name}** shows significant {direction} expression with a log2 fold change of {feature.log2_fold_change:.2f} 
(adjusted p-value: {feature.padj:.2e}).

This {context.get_feature_type_name()} has a combined priority score of {pf.combined_score:.2f}, indicating 
{'high' if pf.combined_score > 0.7 else 'moderate' if pf.combined_score > 0.4 else 'low'} biological significance 
in the context of {context.get_context_string()}.

Further literature analysis would be beneficial to understand the functional implications of this expression change.
            """.strip()
            
            discussions.append(FeatureDiscussion(
                feature_id=feature.feature_id,
                feature_symbol=feature.feature_symbol,
                discussion_text=discussion_text,
                key_findings=[
                    f"Significant {direction} expression (padj < 0.05)",
                    f"Effect size: {abs(feature.log2_fold_change):.2f} log2FC",
                    f"Priority score: {pf.combined_score:.2f}"
                ],
                citations=[],
                therapeutic_implications="Literature analysis required for detailed therapeutic insights."
            ))
        
        return discussions
    
    def _generate_basic_executive_summary(self, context: OmicsExperimentContext, feature_summary: Dict[str, Any], omics_summary: Dict[str, Any]) -> str:
        """Generate basic executive summary without AI synthesis."""
        feature_type = context.get_feature_type_name()
        
        return f"""
## Executive Summary

This {context.omics_type.value} analysis identified {feature_summary['significant_features']} significantly altered {feature_type} 
in the comparison of {context.treatment} vs {context.control} in {context.tissue} samples.

### Key Findings:
- **Total {feature_type} analyzed**: {feature_summary['total_features']:,}
- **Significantly altered**: {feature_summary['significant_features']:,}
- **Upregulated**: {omics_summary['upregulated']}
- **Downregulated**: {omics_summary['downregulated']}

### Analysis Context:
- **Omics Type**: {context.omics_type.value.title()}
- **Experimental Design**: {context.get_context_string()}
- **Organism**: {context.organism.title()}

The analysis identified key {feature_type} with potential relevance to {context.disease or 'the experimental condition'}. 
Detailed feature-by-feature discussions are provided below, along with recommendations for further validation and 
functional studies.

*Note: This analysis was performed without AI-powered literature synthesis. For comprehensive biological interpretation 
and therapeutic insights, consider enabling the synthesis features with appropriate API access.*
        """.strip()


async def main():
    """CLI entry point."""
    import argparse
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Multi-Omics Interpretation Pipeline")
    parser.add_argument("--de-file", required=True, help="Path to DE results file")
    parser.add_argument("--metadata", help="Path to metadata file")
    parser.add_argument("--output", default="omics_analysis_report", help="Output file name")
    parser.add_argument("--max-features", type=int, default=25, help="Maximum number of features to analyze in detail")
    parser.add_argument("--no-cache", action="store_true", help="Disable literature caching")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # Scoring options
    parser.add_argument("--use-scoring", action="store_true", help="Enable literature relevance scoring")
    parser.add_argument("--scorer-type", choices=["tfidf", "bm25", "biobert"], default="tfidf", 
                       help="Scoring method (default: tfidf)")
    parser.add_argument("--biobert-model", default="sentence-transformers/all-MiniLM-L6-v2",
                       help="BioBERT model for semantic scoring")
    
    # MeSH enhancement options
    parser.add_argument("--use-mesh", action="store_true", help="Enable MeSH term enhancement for literature searches")
    parser.add_argument("--mesh-terms-count", type=int, default=3, 
                       help="Number of MeSH terms to generate (default: 3)")
    
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not found. Running without AI synthesis.")
    
    # Simple progress callback
    def progress_callback(message: str, percent: int):
        if percent >= 0:
            print(f"[{percent:3d}%] {message}")
        else:
            print(f"[ERROR] {message}")
    
    # Configure pipeline
    config = AnalysisConfig(
        max_features=args.max_features,
        use_cache=not args.no_cache,
        output_dir=args.output_dir,
        anthropic_api_key=api_key,
        progress_callback=progress_callback,
        use_scoring=args.use_scoring,
        scorer_type=args.scorer_type,
        biobert_model=args.biobert_model,
        use_mesh_enhancement=args.use_mesh,
        mesh_terms_count=args.mesh_terms_count
    )
    
    # Run analysis
    pipeline = SimplifiedPipeline(config)
    
    try:
        report_path = await pipeline.run_analysis(
            de_file=args.de_file,
            metadata_file=args.metadata,
            output_name=args.output
        )
        print(f"\nSuccess! Report generated: {report_path}")
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())