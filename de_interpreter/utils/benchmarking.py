"""Benchmarking utilities for scoring methods.

This module provides tools for comparing different scoring approaches
and evaluating their effectiveness on literature relevance ranking.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..literature.pmc_client import PMCClient
from ..literature.scoring import create_scorer, ScoringConfig
from ..literature.paper import Paper


class ScoringBenchmark:
    """Benchmark different scoring methods for literature relevance."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {}
    
    async def compare_scoring_methods(
        self, 
        queries: List[str], 
        scoring_methods: List[str] = None,
        papers_per_query: int = 20
    ) -> Dict[str, Any]:
        """Compare different scoring methods on a set of queries."""
        
        if scoring_methods is None:
            scoring_methods = ["tfidf", "bm25", "biobert", "gene_query_similarity"]
        
        results = {
            "queries": queries,
            "methods": {},
            "summary": {}
        }
        
        # Test each scoring method
        for method in scoring_methods:
            print(f"\n🧪 Testing {method.upper()} scoring...")
            method_results = await self._test_scoring_method(
                method, queries, papers_per_query
            )
            results["methods"][method] = method_results
        
        # Generate summary statistics
        results["summary"] = self._generate_summary(results["methods"])
        
        return results
    
    async def _test_scoring_method(
        self, 
        method: str, 
        queries: List[str], 
        papers_per_query: int
    ) -> Dict[str, Any]:
        """Test a specific scoring method."""
        
        method_results = {
            "total_time": 0,
            "queries": {},
            "avg_papers_retrieved": 0,
            "avg_time_per_query": 0
        }
        
        start_time = time.time()
        total_papers = 0
        
        async with PMCClient(
            use_scoring=True,
            scorer_type=method,
            progress_callback=lambda msg, pct: None  # Silence progress
        ) as client:
            
            for query in queries:
                query_start = time.time()
                
                try:
                    result = await client.search(query, limit=papers_per_query)
                    papers = result.papers
                    
                    query_time = time.time() - query_start
                    total_papers += len(papers)
                    
                    # Calculate scoring statistics
                    scores = [p.relevance_score for p in papers if p.relevance_score is not None]
                    
                    method_results["queries"][query] = {
                        "papers_found": len(papers),
                        "papers_scored": len(scores),
                        "time_seconds": query_time,
                        "avg_score": sum(scores) / len(scores) if scores else 0,
                        "max_score": max(scores) if scores else 0,
                        "min_score": min(scores) if scores else 0
                    }
                    
                    print(f"  ✓ {query}: {len(papers)} papers in {query_time:.1f}s")
                    
                except Exception as e:
                    print(f"  ❌ {query}: Error - {e}")
                    method_results["queries"][query] = {
                        "error": str(e),
                        "papers_found": 0,
                        "time_seconds": 0
                    }
        
        total_time = time.time() - start_time
        
        method_results["total_time"] = total_time
        method_results["avg_papers_retrieved"] = total_papers / len(queries)
        method_results["avg_time_per_query"] = total_time / len(queries)
        
        return method_results
    
    def _generate_summary(self, methods_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate summary statistics across all methods."""
        
        summary = {
            "fastest_method": None,
            "most_papers_method": None,
            "highest_avg_score_method": None,
            "method_comparison": {}
        }
        
        # Compare methods
        fastest_time = float('inf')
        most_papers = 0
        highest_score = 0
        
        for method, results in methods_results.items():
            avg_time = results.get("avg_time_per_query", float('inf'))
            avg_papers = results.get("avg_papers_retrieved", 0)
            
            # Calculate average score across all queries
            query_scores = []
            for query_result in results.get("queries", {}).values():
                if "avg_score" in query_result:
                    query_scores.append(query_result["avg_score"])
            
            method_avg_score = sum(query_scores) / len(query_scores) if query_scores else 0
            
            summary["method_comparison"][method] = {
                "avg_time_per_query": avg_time,
                "avg_papers_retrieved": avg_papers,
                "avg_relevance_score": method_avg_score
            }
            
            # Track best performers
            if avg_time < fastest_time:
                fastest_time = avg_time
                summary["fastest_method"] = method
            
            if avg_papers > most_papers:
                most_papers = avg_papers
                summary["most_papers_method"] = method
            
            if method_avg_score > highest_score:
                highest_score = method_avg_score
                summary["highest_avg_score_method"] = method
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print benchmark results in a readable format."""
        
        print("\n" + "="*60)
        print("📊 SCORING METHODS BENCHMARK RESULTS")
        print("="*60)
        
        summary = results["summary"]
        
        print(f"\n🏆 Best Performers:")
        print(f"  ⚡ Fastest: {summary['fastest_method']}")
        print(f"  📚 Most Papers: {summary['most_papers_method']}")
        print(f"  🎯 Highest Relevance: {summary['highest_avg_score_method']}")
        
        print(f"\n📈 Method Comparison:")
        for method, stats in summary["method_comparison"].items():
            print(f"\n  {method.upper()}:")
            print(f"    Time per query: {stats['avg_time_per_query']:.2f}s")
            print(f"    Papers retrieved: {stats['avg_papers_retrieved']:.1f}")
            print(f"    Avg relevance score: {stats['avg_relevance_score']:.3f}")


# Example usage function
async def run_benchmark():
    """Run a sample benchmark with common biomedical queries."""
    
    test_queries = [
        "cancer gene expression",
        "COVID-19 immune response",
        "diabetes metabolic pathways",
        "alzheimer neurodegeneration",
        "cardiovascular disease markers"
    ]
    
    benchmark = ScoringBenchmark()
    
    print("🚀 Starting literature scoring benchmark...")
    print(f"📝 Testing {len(test_queries)} queries across multiple methods")
    
    results = await benchmark.compare_scoring_methods(
        queries=test_queries,
        papers_per_query=10
    )
    
    benchmark.print_results(results)
    
    return results


if __name__ == "__main__":
    # Run the benchmark
    asyncio.run(run_benchmark())