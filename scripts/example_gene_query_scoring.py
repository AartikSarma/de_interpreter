#!/usr/bin/env python3
"""
Example: Using existing de_interpreter components for gene-query similarity scoring
(Refined Pooled Approach)

This script demonstrates how to use the existing PMC client and scoring components
to implement the refined gene + query similarity scoring task:

1. For each gene: Search "query + gene" 
2. Pool all abstracts together
3. Score each abstract against "query + all_genes"
4. Sum similarities
"""

import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from de_interpreter.literature.pmc_client import PMCClient
from de_interpreter.literature.scoring import create_scorer


async def refined_gene_query_similarity_with_existing_components(
    genes: List[str], 
    query: str, 
    max_papers_per_gene: int = 20
) -> float:
    """
    Use existing de_interpreter components for refined gene-query similarity scoring.
    
    Args:
        genes: List of gene symbols
        query: Search query
        max_papers_per_gene: Maximum papers to retrieve per gene
        
    Returns:
        Total similarity score
    """
    
    # Initialize components
    async with PMCClient() as pmc_client:
        # You can choose different scorers:
        scorer = create_scorer("tfidf")  # Options: "tfidf", "bm25", "biobert"
        
        print(f"🧬 Refined Approach: Processing {len(genes)} genes with query: '{query}'")
        
        # Step 1: Collect all abstracts from individual gene searches
        all_papers = []
        
        for i, gene in enumerate(genes, 1):
            print(f"\n[{i}/{len(genes)}] Searching for: '{query} {gene}'")
            
            # Search for papers about query + this specific gene
            search_result = await pmc_client.search(
                query=f"{query} {gene}",
                limit=max_papers_per_gene
            )
            
            if not search_result.papers:
                print(f"No papers found for '{query} {gene}'")
                continue
            
            # Tag each paper with source gene for tracking
            for paper in search_result.papers:
                paper.source_gene = gene  # Add custom attribute
            
            all_papers.extend(search_result.papers)
            print(f"Found {len(search_result.papers)} papers for {gene}")
        
        print(f"\n📚 Total papers collected: {len(all_papers)}")
        
        if not all_papers:
            print("❌ No papers found for any genes")
            return 0.0
        
        # Step 2: Create combined query with all genes
        genes_string = " ".join(genes)
        combined_query = f"{query} {genes_string}"
        print(f"🎯 Combined scoring query: '{combined_query}'")
        
        # Step 3: Score all papers against the combined query
        print(f"Computing similarities for {len(all_papers)} abstracts...")
        
        if scorer:
            scored_papers = await scorer.score_papers(combined_query, all_papers)
            scores = [paper.relevance_score or 0.0 for paper in scored_papers]
        else:
            # Fallback to simple keyword matching if scoring unavailable
            scores = []
            query_words = set(combined_query.lower().split())
            for paper in all_papers:
                text = (paper.title + " " + (paper.abstract or "")).lower()
                text_words = set(text.split())
                score = len(query_words.intersection(text_words)) / len(query_words)
                scores.append(score)
        
        # Step 4: Calculate total score
        total_score = sum(scores)
        
        print(f"\n🎯 Results:")
        print(f"Total similarity score: {total_score:.3f}")
        print(f"Average per abstract: {total_score/len(all_papers):.3f}")
        print(f"Average per gene: {total_score/len(genes):.3f}")
        
        # Show top scoring papers
        if scores:
            # Create list of (paper, score) tuples and sort by score
            paper_scores = list(zip(documents, scores))
            paper_scores.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 Top 5 Most Similar Papers:")
            for rank, (doc, score) in enumerate(paper_scores[:5], 1):
                title = doc['title'][:80] + "..." if len(doc['title']) > 80 else doc['title']
                source_gene = doc.get('source_gene', 'unknown')
                print(f"  {rank}. [{score:.3f}] (Gene: {source_gene}) {title}")
    
    return total_score


async def main():
    """Example usage of refined approach"""
    # Example gene list and query
    genes = ["TP53", "BRCA1", "MYC"]
    query = "cancer progression"
    
    print("Gene-Query Similarity Scoring: Refined Pooled Approach")
    print("=" * 60)
    print(f"Genes: {', '.join(genes)}")
    print(f"Query: '{query}'")
    print(f"Max papers per gene: 20")
    print("\nApproach:")
    print("1. Search 'cancer progression TP53', 'cancer progression BRCA1', etc.")
    print("2. Pool all abstracts together")
    print("3. Score each vs 'cancer progression TP53 BRCA1 MYC'")
    print("4. Sum all similarity scores")
    
    try:
        total_score = await refined_gene_query_similarity_with_existing_components(
            genes=genes,
            query=query,
            max_papers_per_gene=20
        )
        
        print(f"\n✅ Analysis Complete!")
        print(f"Total similarity score: {total_score:.3f}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
