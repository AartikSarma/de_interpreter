#!/usr/bin/env python3
"""
Benchmark scoring performance and quality for literature retrieval.

This script compares PMC-only vs PMC+Scoring approaches for literature mining.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Any
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.literature.pmc_client import PMCClient


class PerformanceBenchmark:
    """Benchmark literature retrieval performance and quality."""
    
    def __init__(self):
        self.results = {}
        
    def progress_callback(self, message: str, progress: int):
        """Progress callback for benchmarking."""
        print(f"  📊 {progress:3d}% - {message}")
    
    async def benchmark_query(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Benchmark a single query with different configurations."""
        print(f"\n🔍 Benchmarking query: '{query[:50]}...'")
        
        configs = [
            {"name": "PMC Only", "use_scoring": False, "scorer_type": "none"},
            {"name": "PMC + TF-IDF", "use_scoring": True, "scorer_type": "tfidf"},
            {"name": "PMC + BM25", "use_scoring": True, "scorer_type": "bm25"},
        ]
        
        # Add BioBERT if available
        try:
            from de_interpreter.scoring import BioBERTScorer
            configs.append({"name": "PMC + BioBERT", "use_scoring": True, "scorer_type": "biobert"})
        except ImportError:
            print("⚠️  BioBERT not available, skipping")
        
        query_results = {}
        
        for config in configs:
            print(f"\n  🧪 Testing: {config['name']}")
            
            start_time = time.time()
            
            try:
                async with PMCClient(
                    use_scoring=config["use_scoring"],
                    scorer_type=config["scorer_type"],
                    progress_callback=self.progress_callback
                ) as client:
                    
                    result = await client.search(query, limit=limit)
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Analyze results
                    analysis = self._analyze_results(result, duration)
                    analysis["config"] = config
                    
                    query_results[config["name"]] = analysis
                    
                    print(f"  ✅ {config['name']}: {duration:.1f}s, {len(result.papers)} papers")
                    
                    if config["use_scoring"] and result.papers:
                        avg_score = sum(p.relevance_score for p in result.papers) / len(result.papers)
                        print(f"     📈 Average relevance score: {avg_score:.3f}")
                    
            except Exception as e:
                print(f"  ❌ {config['name']} failed: {e}")
                query_results[config["name"]] = {
                    "error": str(e),
                    "config": config,
                    "duration": None,
                    "papers_count": 0
                }
        
        return query_results
    
    def _analyze_results(self, result, duration: float) -> Dict[str, Any]:
        """Analyze search results for quality metrics."""
        papers = result.papers
        
        analysis = {
            "duration": duration,
            "papers_count": len(papers),
            "papers_with_abstracts": sum(1 for p in papers if p.abstract),
            "papers_with_full_text": sum(1 for p in papers if p.full_text),
            "avg_abstract_length": 0,
            "avg_full_text_length": 0,
            "relevance_scores": []
        }
        
        if papers:
            # Calculate average lengths
            abstracts = [p.abstract for p in papers if p.abstract]
            if abstracts:
                analysis["avg_abstract_length"] = sum(len(a) for a in abstracts) / len(abstracts)
            
            full_texts = [p.full_text for p in papers if p.full_text]
            if full_texts:
                analysis["avg_full_text_length"] = sum(len(t) for t in full_texts) / len(full_texts)
            
            # Collect relevance scores
            analysis["relevance_scores"] = [p.relevance_score for p in papers if hasattr(p, 'relevance_score')]
        
        return analysis
    
    async def run_benchmark(self):
        """Run comprehensive benchmark with multiple queries."""
        print("🚀 Starting Literature Retrieval Benchmark")
        print("=" * 60)
        
        # Test queries representing different types of searches
        test_queries = [
            "ACE2 COVID-19 expression",
            "p53 cancer tumor suppressor",
            "BRCA1 breast cancer mutation",
            "insulin diabetes metabolism",
            "tau alzheimer neurodegeneration"
        ]
        
        all_results = {}
        
        for i, query in enumerate(test_queries):
            print(f"\n📋 Query {i+1}/{len(test_queries)}")
            query_results = await self.benchmark_query(query, limit=5)  # Smaller limit for faster testing
            all_results[query] = query_results
            
            # Brief pause between queries to be respectful
            await asyncio.sleep(2)
        
        # Generate summary report
        self._generate_report(all_results)
        
        return all_results
    
    def _generate_report(self, all_results: Dict[str, Dict[str, Any]]):
        """Generate a summary report of benchmark results."""
        print("\n" + "=" * 60)
        print("📊 BENCHMARK SUMMARY REPORT")
        print("=" * 60)
        
        # Aggregate statistics by configuration
        config_stats = {}
        
        for query, query_results in all_results.items():
            for config_name, result in query_results.items():
                if "error" in result:
                    continue
                    
                if config_name not in config_stats:
                    config_stats[config_name] = {
                        "total_duration": 0,
                        "total_papers": 0,
                        "query_count": 0,
                        "relevance_scores": []
                    }
                
                stats = config_stats[config_name]
                stats["total_duration"] += result["duration"]
                stats["total_papers"] += result["papers_count"]
                stats["query_count"] += 1
                stats["relevance_scores"].extend(result["relevance_scores"])
        
        # Print summary table
        print(f"\n{'Configuration':<20} {'Avg Time':<10} {'Avg Papers':<12} {'Avg Score':<12}")
        print("-" * 60)
        
        for config_name, stats in config_stats.items():
            if stats["query_count"] == 0:
                continue
                
            avg_duration = stats["total_duration"] / stats["query_count"]
            avg_papers = stats["total_papers"] / stats["query_count"]
            avg_score = sum(stats["relevance_scores"]) / len(stats["relevance_scores"]) if stats["relevance_scores"] else 0
            
            print(f"{config_name:<20} {avg_duration:<10.1f} {avg_papers:<12.1f} {avg_score:<12.3f}")
        
        # Performance insights
        print(f"\n🎯 KEY INSIGHTS:")
        
        # Find fastest configuration
        fastest_config = min(config_stats.items(), 
                           key=lambda x: x[1]["total_duration"] / x[1]["query_count"])
        print(f"• Fastest: {fastest_config[0]} ({fastest_config[1]['total_duration'] / fastest_config[1]['query_count']:.1f}s avg)")
        
        # Find highest scoring configuration (if scoring is available)
        scoring_configs = {k: v for k, v in config_stats.items() if v["relevance_scores"]}
        if scoring_configs:
            best_quality = max(scoring_configs.items(),
                             key=lambda x: sum(x[1]["relevance_scores"]) / len(x[1]["relevance_scores"]))
            avg_best_score = sum(best_quality[1]["relevance_scores"]) / len(best_quality[1]["relevance_scores"])
            print(f"• Best Quality: {best_quality[0]} ({avg_best_score:.3f} avg score)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if any("BioBERT" in name for name in config_stats.keys()):
            print("• Use BioBERT for highest quality semantic matching")
            print("• Use TF-IDF or BM25 for balanced speed/quality")
            print("• Use PMC-only for fastest retrieval when relevance ranking not needed")
        else:
            print("• Install BioBERT dependencies for best quality scoring")
            print("• TF-IDF provides good speed/quality balance")
            print("• PMC-only is fastest but lacks relevance ranking")
        
        # Save detailed results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"benchmark_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n💾 Detailed results saved to: {results_file}")


async def main():
    """Run the benchmark."""
    benchmark = PerformanceBenchmark()
    
    try:
        results = await benchmark.run_benchmark()
        print(f"\n✅ Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Literature Retrieval Performance Benchmark")
    print("This will test PMC retrieval with and without scoring")
    print("Press Ctrl+C to interrupt\n")
    
    asyncio.run(main())