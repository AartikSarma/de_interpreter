"""Multi-omics interpretation pipeline."""

import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv

from .parsers.omics_data import OmicsType
from .parsers.omics_parser import OmicsParser, OmicsMetadataParser
from .prioritization.omics_prioritizer import OmicsPrioritizer
from .literature import FutureHouseClient, PMCClient, LiteratureCache
from .synthesis import ClaudeSynthesizer
from .reporting import ReportGenerator


class OmicsInterpreter:
    """Main orchestrator for multi-omics interpretation pipeline."""

    def __init__(
        self,
        omics_type: OmicsType = OmicsType.TRANSCRIPTOMICS,
        use_cache: bool = True,
        top_n_features: int = 50,
        max_analysis_features: Optional[int] = None,
        use_pmc: bool = True,
    ):
        self.omics_type = omics_type
        self.use_cache = use_cache
        self.use_pmc = use_pmc
        self.top_n_features = top_n_features
        # Default to top_n_features for analysis, but allow override
        self.max_analysis_features = max_analysis_features or min(top_n_features, 30)

        # Initialize omics-specific components
        self.omics_parser = OmicsParser(omics_type)
        self.metadata_parser = OmicsMetadataParser()
        self.prioritizer = OmicsPrioritizer(top_n=top_n_features)
        self.cache = LiteratureCache() if use_cache else None
        self.report_generator = ReportGenerator()

    async def run(
        self,
        data_file: Path,
        metadata_file: Path,
        output_name: str = "omics_analysis_report",
    ) -> Path:
        """Run the complete multi-omics analysis pipeline."""
        print(f"Starting {self.omics_type.value} interpretation pipeline...")

        # Step 1: Parse inputs
        print("\n1. Parsing input files...")
        omics_features = self.omics_parser.parse(data_file)
        context = self.metadata_parser.parse(metadata_file)
        
        # Update context with detected omics type if not specified
        if context.omics_type != self.omics_type:
            context.omics_type = self.omics_type
        
        feature_summary = self.omics_parser.summary_stats()

        print(f"   - Loaded {len(omics_features)} {feature_summary.get('feature_type', 'features')}")
        print(f"   - Context: {context.get_context_string()}")
        print(f"   - Omics type: {context.omics_type.value}")

        # Step 2: Prioritize features
        print("\n2. Prioritizing features...")
        prioritized = self.prioritizer.prioritize(omics_features, context)
        priority_summary = self.prioritizer.get_summary_stats()
        omics_summary = self.prioritizer.get_omics_specific_summary()

        print(f"   - Selected top {len(prioritized)} {omics_summary.get('feature_type_name', 'features')}")
        print(f"   - Upregulated: {priority_summary['upregulated']}")
        print(f"   - Downregulated: {priority_summary['downregulated']}")

        # Step 3: Select top features for analysis
        print(f"\n3. Selecting top {self.max_analysis_features} features for detailed analysis...")
        analysis_features = prioritized[:self.max_analysis_features]
        
        print(f"   - Selected {len(analysis_features)} {omics_summary.get('feature_type_name', 'features')} for literature mining and synthesis")

        # Step 4: Literature mining
        print("\n4. Mining literature...")
        feature_papers = await self._fetch_literature(analysis_features, context)

        total_papers = sum(len(papers) for papers in feature_papers.values())
        print(f"   - Found {total_papers} relevant papers")

        # Step 5: Synthesize discussions
        print("\n5. Synthesizing feature discussions...")
        async with ClaudeSynthesizer() as synthesizer:
            # Individual feature discussions
            feature_discussions = await synthesizer.batch_synthesize_omics(
                analysis_features, context, feature_papers
            )

            print(f"   - Generated {len(feature_discussions)} feature discussions")

            # Executive summary
            print("\n6. Generating executive summary...")
            executive_summary = await synthesizer.generate_omics_executive_summary(
                feature_discussions, context, feature_summary, omics_summary
            )

        # Step 6: Generate report
        print("\n7. Generating final report...")
        report_path = self.report_generator.generate_omics_report(
            executive_summary=executive_summary,
            feature_discussions=feature_discussions,
            context=context,
            feature_summary=feature_summary,
            omics_summary=omics_summary,
            analysis_features=analysis_features,
            output_name=output_name,
        )

        print(f"\n✅ Analysis complete! Report saved to: {report_path}")

        return report_path

    async def _fetch_literature(
        self, prioritized_features, context
    ) -> Dict[str, List]:
        """Fetch literature for prioritized features."""
        feature_papers = {}

        # Choose literature client based on configuration
        if self.use_pmc:
            print("   - Using PMC for literature retrieval...")
            async with PMCClient() as client:
                # Prepare queries
                queries = []
                feature_keys = []

                for feature in prioritized_features[: self.max_analysis_features]:
                    feature_key = feature.display_name
                    feature_keys.append(feature_key)

                    # Check cache first
                    if self.cache:
                        # Create omics-aware query
                        query = f"{feature_key} {context.disease} {context.get_omics_context()}"
                        cached = self.cache.get(query)
                        if cached:
                            feature_papers[feature_key] = cached.papers
                            continue

                    queries.append(feature_key)

                # Batch fetch uncached features
                if queries:
                    print(f"   - Fetching literature for {len(queries)} {context.get_feature_type_name()}s...")

                    # Create omics-aware search queries
                    search_queries = [
                        f"{feature} {context.disease} {context.get_omics_context()}" 
                        for feature in queries
                    ]

                    # Batch search with smaller limit for PMC
                    results = await client.batch_search(search_queries, limit_per_query=5)

                    # Process results
                    for feature_key, result in zip(queries, results):
                        feature_papers[feature_key] = result.papers

                        # Cache results
                        if self.cache:
                            self.cache.set(result)

        else:
            print("   - Using FutureHouse for literature retrieval...")
            async with FutureHouseClient() as client:
                # Prepare queries
                queries = []
                feature_keys = []

                for feature in prioritized_features[: self.max_analysis_features]:
                    feature_key = feature.display_name
                    feature_keys.append(feature_key)

                    # Check cache first
                    if self.cache:
                        # Create omics-aware query
                        query = f"{feature_key} {context.disease} {context.get_omics_context()}"
                        cached = self.cache.get(query)
                        if cached:
                            feature_papers[feature_key] = cached.papers
                            continue

                    queries.append(feature_key)

                # Batch fetch uncached features
                if queries:
                    print(f"   - Fetching literature for {len(queries)} {context.get_feature_type_name()}s...")

                    # Create omics-aware search queries
                    search_queries = [
                        f"{feature} {context.disease} {context.get_omics_context()}" 
                        for feature in queries
                    ]

                    # Batch search
                    results = await client.batch_search(search_queries, limit_per_query=10)

                    # Process results
                    for feature_key, result in zip(queries, results):
                        feature_papers[feature_key] = result.papers

                        # Cache results
                        if self.cache:
                            self.cache.set(result)

        return feature_papers


def main():
    """Command-line interface for multi-omics analysis."""
    parser = argparse.ArgumentParser(
        description="Interpret differential omics results with literature context"
    )

    parser.add_argument(
        "--data-file",
        type=Path,
        required=True,
        help="Path to differential omics results (CSV/TSV/Excel)",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to experimental metadata (JSON/YAML)",
    )

    parser.add_argument(
        "--omics-type",
        type=str,
        choices=[t.value for t in OmicsType],
        default="transcriptomics",
        help="Type of omics data (default: transcriptomics)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="omics_analysis_report",
        help="Output report name (without extension)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top features to prioritize (default: 50)",
    )

    parser.add_argument(
        "--max-analysis",
        type=int,
        default=None,
        help="Maximum number of features for detailed analysis (default: min(top_n, 30))",
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Disable literature caching"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.data_file.exists():
        print(f"Error: Data file not found: {args.data_file}")
        sys.exit(1)

    if not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Create interpreter
    omics_type = OmicsType(args.omics_type)
    interpreter = OmicsInterpreter(
        omics_type=omics_type,
        use_cache=not args.no_cache, 
        top_n_features=args.top_n, 
        max_analysis_features=args.max_analysis
    )

    # Run analysis
    try:
        report_path = asyncio.run(
            interpreter.run(
                data_file=args.data_file,
                metadata_file=args.metadata,
                output_name=args.output,
            )
        )
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()