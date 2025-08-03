"""Example usage patterns for the DE Interpreter.

This module provides example code showing how to use various features
of the DE Interpreter programmatically.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any

from ..main import SimplifiedPipeline, AnalysisConfig
from .benchmarking import ScoringBenchmark


async def basic_analysis_example():
    """Basic example of running analysis programmatically."""
    
    # Configure the analysis
    config = AnalysisConfig(
        max_features=10,
        use_cache=True,
        use_scoring=True,
        scorer_type="tfidf",
        anthropic_api_key="your-api-key-here"
    )
    
    # Create pipeline
    pipeline = SimplifiedPipeline(config)
    
    # Run analysis
    report_path = await pipeline.run_analysis(
        de_file="covid_data/covid_deg_fixed.csv",
        metadata_file="covid_data/covid_metadata.json",
        output_name="example_analysis"
    )
    
    print(f"Analysis complete! Report: {report_path}")
    return report_path


async def advanced_scoring_example():
    """Example of using different scoring methods."""
    
    scoring_methods = ["tfidf", "bm25", "biobert", "gene_query_similarity"]
    
    for method in scoring_methods:
        print(f"\n🧪 Testing {method} scoring...")
        
        config = AnalysisConfig(
            max_features=5,
            use_cache=False,  # Don't cache for testing
            use_scoring=True,
            scorer_type=method,
            anthropic_api_key="your-api-key-here"
        )
        
        pipeline = SimplifiedPipeline(config)
        
        try:
            report_path = await pipeline.run_analysis(
                de_file="covid_data/covid_deg_fixed.csv",
                metadata_file="covid_data/covid_metadata.json",
                output_name=f"scoring_test_{method}"
            )
            print(f"✅ {method}: {report_path}")
            
        except Exception as e:
            print(f"❌ {method}: {e}")


async def mesh_enhancement_example():
    """Example of using MeSH term enhancement."""
    
    config = AnalysisConfig(
        max_features=8,
        use_cache=True,
        use_scoring=True,
        scorer_type="biobert",
        use_mesh_enhancement=True,
        mesh_terms_count=4,
        anthropic_api_key="your-api-key-here"
    )
    
    pipeline = SimplifiedPipeline(config)
    
    report_path = await pipeline.run_analysis(
        de_file="covid_data/covid_deg_fixed.csv",
        metadata_file="covid_data/covid_metadata.json",
        output_name="mesh_enhanced_analysis"
    )
    
    print(f"MeSH-enhanced analysis complete! Report: {report_path}")
    return report_path


async def benchmark_scoring_methods():
    """Example of benchmarking different scoring methods."""
    
    test_queries = [
        "COVID-19 immune response",
        "ARDS respiratory inflammation",
        "viral infection transcriptomics"
    ]
    
    benchmark = ScoringBenchmark()
    
    print("🚀 Running scoring benchmark...")
    results = await benchmark.compare_scoring_methods(
        queries=test_queries,
        scoring_methods=["tfidf", "gene_query_similarity", "biobert"],
        papers_per_query=5
    )
    
    benchmark.print_results(results)
    return results


def custom_progress_callback_example():
    """Example of using a custom progress callback."""
    
    def my_progress_callback(message: str, percent: int):
        """Custom progress callback with fancy formatting."""
        if percent >= 0:
            bar_length = 20
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r|{bar}| {percent:3d}% {message}', end='')
            if percent >= 100:
                print()  # New line when complete
        else:
            print(f"❌ ERROR: {message}")
    
    return my_progress_callback


async def custom_callback_example():
    """Example using a custom progress callback."""
    
    config = AnalysisConfig(
        max_features=5,
        use_cache=True,
        progress_callback=custom_progress_callback_example(),
        anthropic_api_key="your-api-key-here"
    )
    
    pipeline = SimplifiedPipeline(config)
    
    report_path = await pipeline.run_analysis(
        de_file="covid_data/covid_deg_fixed.csv",
        metadata_file="covid_data/covid_metadata.json",
        output_name="custom_callback_analysis"
    )
    
    print(f"\nCustom callback analysis complete! Report: {report_path}")
    return report_path


# Collection of all examples
EXAMPLES = {
    "basic": basic_analysis_example,
    "scoring": advanced_scoring_example,
    "mesh": mesh_enhancement_example,
    "benchmark": benchmark_scoring_methods,
    "callback": custom_callback_example
}


async def run_example(example_name: str):
    """Run a specific example by name."""
    
    if example_name not in EXAMPLES:
        print(f"❌ Unknown example: {example_name}")
        print(f"Available examples: {', '.join(EXAMPLES.keys())}")
        return
    
    print(f"🏃 Running example: {example_name}")
    print("-" * 40)
    
    try:
        result = await EXAMPLES[example_name]()
        print(f"✅ Example '{example_name}' completed successfully")
        return result
        
    except Exception as e:
        print(f"❌ Example '{example_name}' failed: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        asyncio.run(run_example(example_name))
    else:
        print("Available examples:")
        for name, func in EXAMPLES.items():
            print(f"  {name}: {func.__doc__.split('.')[0] if func.__doc__ else 'No description'}")
        print(f"\nUsage: python -m de_interpreter.utils.examples <example_name>")