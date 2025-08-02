"""Main orchestrator for DE interpretation pipeline."""

import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import sys
from dotenv import load_dotenv

from .parsers import DEParser, MetadataParser
from .prioritization import GenePrioritizer
from .literature import FutureHouseClient, PMCClient, LiteratureCache
from .synthesis import ClaudeSynthesizer
from .reporting import ReportGenerator


class DEInterpreter:
    """Main orchestrator for the DE interpretation pipeline."""

    def __init__(
        self,
        use_cache: bool = True,
        top_n_genes: int = 50,
        max_analysis_genes: Optional[int] = None,
        use_pmc: bool = True,
    ):
        self.use_cache = use_cache
        self.use_pmc = use_pmc
        self.top_n_genes = top_n_genes
        # Default to top_n_genes for analysis, but allow override
        self.max_analysis_genes = max_analysis_genes or min(top_n_genes, 30)

        # Initialize components
        self.de_parser = DEParser()
        self.metadata_parser = MetadataParser()
        self.prioritizer = GenePrioritizer(top_n=top_n_genes)
        self.cache = LiteratureCache() if use_cache else None
        self.report_generator = ReportGenerator()

    async def run(
        self,
        de_file: Path,
        metadata_file: Path,
        output_name: str = "de_analysis_report",
    ) -> Path:
        """Run the complete analysis pipeline."""
        print("Starting DE interpretation pipeline...")

        # Step 1: Parse inputs
        print("\n1. Parsing input files...")
        de_results = self.de_parser.parse(de_file)
        context = self.metadata_parser.parse(metadata_file)
        de_summary = self.de_parser.summary_stats()

        print(f"   - Loaded {len(de_results)} genes")
        print(f"   - Context: {context.get_context_string()}")

        # Step 2: Prioritize genes
        print("\n2. Prioritizing genes...")
        prioritized = self.prioritizer.prioritize(de_results, context)
        priority_summary = self.prioritizer.get_summary_stats()

        print(f"   - Selected top {len(prioritized)} genes")
        print(f"   - Upregulated: {priority_summary['upregulated']}")
        print(f"   - Downregulated: {priority_summary['downregulated']}")

        # Step 3: Select top genes for analysis
        print(f"\n3. Selecting top {self.max_analysis_genes} genes for detailed analysis...")
        analysis_genes = prioritized[:self.max_analysis_genes]
        
        print(f"   - Selected {len(analysis_genes)} genes for literature mining and synthesis")

        # Step 4: Literature mining
        print("\n4. Mining literature...")
        gene_papers = await self._fetch_literature(analysis_genes, context)

        total_papers = sum(len(papers) for papers in gene_papers.values())
        print(f"   - Found {total_papers} relevant papers")

        # Step 5: Synthesize discussions
        print("\n5. Synthesizing gene discussions...")
        async with ClaudeSynthesizer() as synthesizer:
            # Individual gene discussions for selected genes
            gene_discussions = await synthesizer.batch_synthesize(
                analysis_genes, context, gene_papers
            )

            print(f"   - Generated {len(gene_discussions)} gene discussions")

            # Executive summary
            print("\n6. Generating executive summary...")
            executive_summary = await synthesizer.generate_executive_summary(
                gene_discussions, context, de_summary
            )

        # Step 6: Generate report
        print("\n7. Generating final report...")
        report_path = self.report_generator.generate_report(
            executive_summary=executive_summary,
            gene_discussions=gene_discussions,
            context=context,
            de_summary=de_summary,
            analysis_genes=analysis_genes,
            output_name=output_name,
        )

        print(f"\n✅ Analysis complete! Report saved to: {report_path}")

        return report_path

    async def _fetch_literature(
        self, prioritized_genes: List["PrioritizedGene"], context: "ExperimentalContext"
    ) -> Dict[str, List["Paper"]]:
        """Fetch literature for prioritized genes."""
        gene_papers = {}

        # Choose literature client based on configuration
        if self.use_pmc:
            print("   - Using PMC for literature retrieval...")
            async with PMCClient() as client:
                # Prepare queries
                queries = []
                gene_keys = []

                for gene in prioritized_genes[: self.top_n_genes]:
                    gene_key = gene.gene_symbol or gene.gene_id
                    gene_keys.append(gene_key)

                    # Check cache first
                    if self.cache:
                        query = f"{gene_key} {context.disease}"
                        cached = self.cache.get(query)
                        if cached:
                            gene_papers[gene_key] = cached.papers
                            continue

                    queries.append(gene_key)

                # Batch fetch uncached genes
                if queries:
                    print(f"   - Fetching literature for {len(queries)} genes...")

                    # Create search queries
                    search_queries = [f"{gene} {context.disease}" for gene in queries]

                    # Batch search
                    results = await client.batch_search(search_queries, limit_per_query=5)

                    # Process results
                    for gene_key, result in zip(queries, results):
                        gene_papers[gene_key] = result.papers

                        # Cache results
                        if self.cache:
                            self.cache.set(result)

        else:
            print("   - Using FutureHouse for literature retrieval...")
            async with FutureHouseClient() as client:
                # Prepare queries
                queries = []
                gene_keys = []

                for gene in prioritized_genes[: self.top_n_genes]:
                    gene_key = gene.gene_symbol or gene.gene_id
                    gene_keys.append(gene_key)

                    # Check cache first
                    if self.cache:
                        query = f"{gene_key} {context.disease}"
                        cached = self.cache.get(query)
                        if cached:
                            gene_papers[gene_key] = cached.papers
                            continue

                    queries.append(gene_key)

                # Batch fetch uncached genes
                if queries:
                    print(f"   - Fetching literature for {len(queries)} genes...")

                    # Create search queries
                    search_queries = [f"{gene} {context.disease}" for gene in queries]

                    # Batch search
                    results = await client.batch_search(search_queries, limit_per_query=10)

                    # Process results
                    for gene_key, result in zip(queries, results):
                        gene_papers[gene_key] = result.papers

                        # Cache results
                        if self.cache:
                            self.cache.set(result)

        return gene_papers


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Interpret differential expression results with literature context"
    )

    parser.add_argument(
        "--de-file",
        type=Path,
        required=True,
        help="Path to differential expression results (CSV/TSV/Excel)",
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to experimental metadata (JSON/YAML)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="de_analysis_report",
        help="Output report name (without extension)",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top genes to prioritize (default: 50)",
    )

    parser.add_argument(
        "--max-analysis",
        type=int,
        default=None,
        help="Maximum number of genes for detailed analysis (default: min(top_n, 30))",
    )

    parser.add_argument(
        "--no-cache", action="store_true", help="Disable literature caching"
    )

    parser.add_argument(
        "--use-futurehouse", action="store_true", help="Use FutureHouse API instead of PMC (default: PMC)"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.de_file.exists():
        print(f"Error: DE file not found: {args.de_file}")
        sys.exit(1)

    if not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        sys.exit(1)

    # Load environment variables
    load_dotenv()

    # Create interpreter
    interpreter = DEInterpreter(
        use_cache=not args.no_cache, 
        top_n_genes=args.top_n, 
        max_analysis_genes=args.max_analysis,
        use_pmc=not args.use_futurehouse
    )

    # Run analysis
    try:
        report_path = asyncio.run(
            interpreter.run(
                de_file=args.de_file,
                metadata_file=args.metadata,
                output_name=args.output,
            )
        )
    except Exception as e:
        print(f"\nError during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
