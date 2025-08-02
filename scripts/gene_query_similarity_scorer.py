#!/usr/bin/env python3
"""
Gene + Query Literature Similarity Scorer (Refined Pooled Approach)

This script implements a refined approach for scoring gene-query literature similarity:

1. For each gene G: Search PubMed for "query + G" 
2. Pool all abstracts from all gene searches into one collection
3. Create combined query: "original_query + all_genes_concatenated"
4. Score each abstract against the combined query using semantic similarity
5. Sum all similarities for a total relevance score

This approach avoids double-counting genes in similarity scoring and provides
a more holistic view of literature relevance across the entire gene set.

Usage:
    python gene_query_similarity_scorer.py --genes TP53 BRCA1 MYC --query "cancer progression" --top-papers 20
"""

import argparse
import asyncio
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import requests
    from xml.etree import ElementTree as ET
    import numpy as np
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install sentence-transformers scikit-learn requests lxml beautifulsoup4")
    sys.exit(1)

# Add the src directory to the path so we can import from de_interpreter
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from de_interpreter.literature.pmc_retriever import PubMedRetriever
    from de_interpreter.scoring.biobert.scorer import BioBERTScorer
except ImportError as e:
    print(f"❌ Could not import de_interpreter modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


class PubMedAbstractFetcher:
    """Fetch abstracts from PubMed using E-utilities API"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        # Be respectful to NCBI servers
        self.session.headers.update({
            'User-Agent': 'GeneQueryScorer/1.0 (literature-research@example.com)'
        })
    
    def search_gene_query(self, gene: str, query: str, max_results: int = 20) -> List[str]:
        """
        Search PubMed for papers matching gene + query and return PMIDs.
        
        Args:
            gene: Gene symbol (e.g., "TP53")
            query: Search query (e.g., "cancer progression")
            max_results: Maximum number of papers to retrieve
            
        Returns:
            List of PMIDs as strings
        """
        # Construct search term
        search_term = f"{gene} AND {query}"
        
        params = {
            'db': 'pubmed',
            'term': search_term,
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance',
            'field': 'title/abstract'  # Search in title and abstract
        }
        
        try:
            print(f"Searching PubMed for: '{search_term}'")
            response = self.session.get(f"{self.base_url}esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            
            print(f"Found {len(pmids)} papers for {gene}")
            return pmids
            
        except Exception as e:
            print(f"❌ Error searching for {gene}: {e}")
            return []
    
    def fetch_abstracts_batch(self, pmids: List[str]) -> List[Dict[str, str]]:
        """
        Fetch abstracts for multiple PMIDs in batch.
        
        Args:
            pmids: List of PubMed IDs
            
        Returns:
            List of dicts with 'pmid', 'title', 'abstract' keys
        """
        if not pmids:
            return []
        
        # Batch fetch to be efficient
        pmid_str = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': pmid_str,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            print(f"Fetching {len(pmids)} abstracts...")
            response = self.session.get(f"{self.base_url}efetch.fcgi", params=params, timeout=60)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            abstracts = []
            for article in root.findall('.//PubmedArticle'):
                try:
                    # Extract PMID
                    pmid_elem = article.find('.//PMID')
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text
                    
                    # Extract title
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ""
                    
                    # Extract abstract
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = ""
                    if abstract_elem is not None:
                        abstract = abstract_elem.text or ""
                    
                    # Skip papers without abstracts
                    if not abstract.strip():
                        continue
                    
                    abstracts.append({
                        'pmid': pmid,
                        'title': title,
                        'abstract': abstract
                    })
                    
                except Exception as e:
                    print(f"⚠️  Error parsing article: {e}")
                    continue
            
            print(f"Successfully extracted {len(abstracts)} abstracts")
            return abstracts
            
        except Exception as e:
            print(f"❌ Error fetching abstracts: {e}")
            return []


class GeneQuerySimilarityScorer:
    """Main scorer that combines gene searches and similarity computation"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the scorer.
        
        Args:
            model_name: Sentence transformer model for similarity scoring
        """
        self.fetcher = PubMedAbstractFetcher()
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading similarity model: {self.model_name}")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.1f}s")
    
    def compute_similarities(self, query: str, abstracts: List[Dict[str, str]]) -> List[float]:
        """
        Compute similarity scores between query and abstracts using the SAME method as score_query_vs_pmid.py:
        - SentenceTransformer embeddings
        - Cosine similarity
        
        Args:
            query: Search query
            abstracts: List of abstract dicts
            
        Returns:
            List of similarity scores
        """
        if not abstracts:
            return []
        
        if self.model is None:
            self.load_model()
        
        # Combine title and abstract for each paper (same as score_query_vs_pmid.py approach)
        texts = [f"{abs_dict['title']} {abs_dict['abstract']}" for abs_dict in abstracts]
        
        print(f"Computing similarities for {len(texts)} abstracts using {self.model_name}...")
        
        # Encode query and all abstracts using SentenceTransformer
        print("Encoding query...")
        query_embedding = self.model.encode([query])
        print("Encoding abstracts...")
        abstract_embeddings = self.model.encode(texts)
        
        # Compute cosine similarities (same as score_query_vs_pmid.py)
        similarities = cosine_similarity(query_embedding, abstract_embeddings)[0]
        
        return similarities.tolist()
    
    def score_gene_list(self, 
                       genes: List[str], 
                       query: str, 
                       top_papers: int = 20,
                       output_file: Optional[str] = None) -> float:
        """
        Score a list of genes against a query using the refined approach:
        1. For each gene, search for "query + gene" 
        2. Pool all abstracts together
        3. Score each abstract against "query + all_genes_combined"
        4. Sum all similarities
        
        Args:
            genes: List of gene symbols
            query: Search query
            top_papers: Number of papers to fetch per gene
            output_file: Optional file to save detailed results
            
        Returns:
            Total similarity score (sum of all individual similarities)
        """
        print(f"\n🧬 Starting gene-query similarity scoring (Refined Approach)")
        print(f"Genes: {', '.join(genes)}")
        print(f"Query: '{query}'")
        print(f"Papers per gene: {top_papers}")
        print("="*80)
        
        # Step 1: Collect all abstracts from all gene searches
        all_abstracts = []
        gene_search_results = []
        
        for i, gene in enumerate(genes, 1):
            print(f"\n[{i}/{len(genes)}] Searching for gene: {gene}")
            
            # Search PubMed for "query + gene"
            pmids = self.fetcher.search_gene_query(gene, query, max_results=top_papers)
            
            if not pmids:
                print(f"No papers found for {gene}")
                gene_search_results.append({
                    'gene': gene,
                    'pmids_found': 0,
                    'abstracts_retrieved': 0
                })
                continue
            
            # Fetch abstracts for this gene
            abstracts = self.fetcher.fetch_abstracts_batch(pmids)
            
            print(f"Gene {gene}: {len(pmids)} PMIDs found, {len(abstracts)} abstracts retrieved")
            
            # Add source gene info to each abstract for tracking
            for abstract in abstracts:
                abstract['source_gene'] = gene
            
            # Add to the pooled collection
            all_abstracts.extend(abstracts)
            
            gene_search_results.append({
                'gene': gene,
                'pmids_found': len(pmids),
                'abstracts_retrieved': len(abstracts)
            })
            
            # Rate limiting to be respectful to NCBI
            if i < len(genes):
                time.sleep(1)
        
        print(f"\n📚 Pooled Results:")
        print(f"Total abstracts collected: {len(all_abstracts)}")
        
        if not all_abstracts:
            print("❌ No abstracts found for any genes")
            return 0.0
        
        # Step 2: Create combined query (query + all genes)
        genes_string = " ".join(genes)
        combined_query = f"{query} {genes_string}"
        print(f"Combined scoring query: '{combined_query}'")
        
        # Step 3: Score all abstracts against the combined query
        print(f"\n🎯 Computing similarities for {len(all_abstracts)} abstracts...")
        similarities = self.compute_similarities(combined_query, all_abstracts)
        
        # Step 4: Calculate total score
        total_score = sum(similarities)
        
        print(f"\n🎯 Final Results:")
        print(f"Total similarity score: {total_score:.3f}")
        print(f"Average per abstract: {total_score/len(all_abstracts):.3f}")
        print(f"Average per gene: {total_score/len(genes):.3f}")
        
        # Create detailed results for saving
        detailed_results = []
        for i, (abstract, similarity) in enumerate(zip(all_abstracts, similarities)):
            detailed_results.append({
                'rank': i + 1,
                'pmid': abstract['pmid'],
                'source_gene': abstract['source_gene'],
                'title': abstract['title'][:100] + "..." if len(abstract['title']) > 100 else abstract['title'],
                'similarity': similarity
            })
        
        # Sort by similarity (highest first) for the detailed results
        detailed_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Show top scoring abstracts
        print(f"\n� Top 5 Most Similar Abstracts:")
        for i, result in enumerate(detailed_results[:5], 1):
            print(f"  {i}. [{result['similarity']:.3f}] (Gene: {result['source_gene']}) {result['title']}")
        
        # Save detailed results if requested
        if output_file:
            results_data = {
                'approach': 'refined_pooled_scoring',
                'query': query,
                'genes': genes,
                'combined_query': combined_query,
                'top_papers_per_gene': top_papers,
                'total_abstracts': len(all_abstracts),
                'total_score': total_score,
                'average_per_abstract': total_score/len(all_abstracts),
                'average_per_gene': total_score/len(genes),
                'model_used': self.model_name,
                'gene_search_summary': gene_search_results,
                'abstract_scores': detailed_results,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Detailed results saved to: {output_file}")
        
        return total_score


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description="Score similarity between gene list and query using PubMed literature (Refined Pooled Approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Refined Approach:
1. For each gene G: Search PubMed for "query + G"
2. Pool all abstracts from all searches
3. Score each abstract against "query + all_genes_combined"
4. Sum all similarities for total score

Examples:
  python gene_query_similarity_scorer.py --genes TP53 BRCA1 MYC --query "cancer progression" --top-papers 20
  python gene_query_similarity_scorer.py --genes ARG1 IL6 TNF --query "inflammatory response" --top-papers 15 --output results.json
  python gene_query_similarity_scorer.py --genes-file genes.txt --query "metabolic dysfunction" --top-papers 10

Example workflow:
  - Search: "cancer progression TP53", "cancer progression BRCA1", "cancer progression MYC"
  - Pool: All abstracts from the 3 searches combined
  - Score: Each abstract vs "cancer progression TP53 BRCA1 MYC" 
  - Result: Sum of all similarity scores
        """
    )
    
    # Gene input options
    gene_group = parser.add_mutually_exclusive_group(required=True)
    gene_group.add_argument(
        "--genes", "-g",
        nargs='+',
        help="Space-separated list of gene symbols (e.g., TP53 BRCA1 MYC)"
    )
    gene_group.add_argument(
        "--genes-file",
        help="File containing gene symbols, one per line"
    )
    
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Search query to combine with each gene"
    )
    
    parser.add_argument(
        "--top-papers", "-n",
        type=int,
        default=20,
        help="Number of top papers to retrieve per gene (default: 20)"
    )
    
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model for similarity scoring"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Output file to save detailed results (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    args = parser.parse_args()
    
    # Get gene list
    if args.genes:
        genes = args.genes
    else:
        try:
            with open(args.genes_file, 'r') as f:
                genes = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"❌ Gene file not found: {args.genes_file}")
            sys.exit(1)
    
    if not genes:
        print("❌ No genes provided")
        sys.exit(1)
    
    # Validate genes (basic check)
    genes = [gene.upper().strip() for gene in genes]
    genes = [gene for gene in genes if gene]  # Remove empty strings
    
    if len(genes) > 50:
        print(f"⚠️  Large gene list ({len(genes)} genes). This may take a while.")
        confirm = input("Continue? (y/N): ")
        if confirm.lower() != 'y':
            sys.exit(0)
    
    # Initialize scorer and run
    try:
        scorer = GeneQuerySimilarityScorer(model_name=args.model)
        total_score = scorer.score_gene_list(
            genes=genes,
            query=args.query,
            top_papers=args.top_papers,
            output_file=args.output
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"Total similarity score: {total_score:.3f}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
