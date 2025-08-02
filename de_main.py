#!/usr/bin/env python3
"""
Backward-compatible CLI for traditional DE analysis.

This provides the same interface as the original main.py but uses the unified pipeline.
"""

import asyncio
import argparse
from pathlib import Path
import sys
from dotenv import load_dotenv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.unified_main import LegacyCompatibilityWrapper


def main():
    """Backward-compatible command-line interface for DE analysis."""
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
        "--use-futurehouse", action="store_true", 
        help="Use FutureHouse API instead of PMC (default: PMC)"
    )

    parser.add_argument(
        "--enable-scoring", action="store_true", 
        help="Enable literature relevance scoring"
    )

    parser.add_argument(
        "--scorer", type=str, default="tfidf", 
        choices=["tfidf", "bm25", "biobert"],
        help="Scoring method to use (default: tfidf)"
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

    # Create legacy compatibility wrapper
    interpreter = LegacyCompatibilityWrapper(
        use_cache=not args.no_cache,
        top_n_features=args.top_n,
        max_analysis_features=args.max_analysis,
        use_pmc=not args.use_futurehouse,
        use_scoring=args.enable_scoring,
        scorer_type=args.scorer
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