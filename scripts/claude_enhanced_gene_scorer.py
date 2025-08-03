#!/usr/bin/env python3
"""
Claude-Enhanced Gene-Query Similarity Scorer

This script extends minimal_gene_scorer.py by using Claude Haiku to generate
relevant MeSH terms for better PubMed searches:

1. Use Claude Haiku API to get N MeSH terms for the query
2. Concatenate MeSH terms with OR for PubMed search
3. Search each gene + expanded query
4. Pool all abstracts and score against original query
5. Sum similarities for final score

Dependencies: sentence-transformers, scikit-learn, requests, anthropic
"""

import requests
import json
import time
import argparse
import os
from typing import List, Dict
from xml.etree import ElementTree as ET

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from parent directory (main project folder)
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    load_dotenv(env_path)
    print(f"🔧 Loaded environment from: {env_path}")
except ImportError:
    print("ℹ️ python-dotenv not installed, using system environment variables")
except Exception as e:
    print(f"⚠️ Could not load .env file: {e}")

# Turn off future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import anthropic
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install sentence-transformers scikit-learn requests anthropic")
    exit(1)


class ClaudeEnhancedPubMedScorer:
    """Enhanced PubMed scorer using Claude Haiku for MeSH term generation"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 claude_api_key: str = None):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ClaudeEnhancedGeneSimilarityScorer/1.0'
        })
        self.model_name = model_name
        self.model = None
        
        # Store Claude API key for later initialization
        self.claude_api_key = claude_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.claude_client = None
    
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.1f}s")
    
    def _init_claude_client(self):
        """Initialize Claude client only when needed"""
        if self.claude_client is None:
            if not self.claude_api_key:
                raise ValueError("Claude API key not found. Set ANTHROPIC_API_KEY environment variable or pass claude_api_key parameter.")
            self.claude_client = anthropic.Anthropic(api_key=self.claude_api_key)

    def generate_mesh_terms(self, query: str, n_terms: int = 3) -> List[str]:
        """
        Use Claude Haiku to generate relevant MeSH terms for the query
        
        Args:
            query: User query
            n_terms: Number of MeSH terms to generate
            
        Returns:
            List of MeSH terms
        """
        prompt = f"""Given the following biomedical query, please provide {n_terms} relevant MeSH (Medical Subject Headings) terms that would be most effective for a PubMed literature search.

        Query: "{query}"

        Please respond with exactly {n_terms} MeSH terms, one per line, without any additional explanation or formatting. Use the exact MeSH terminology as it appears in PubMed.

        Example format:
        Neoplasms
        Cell Death
        Gene Expression Regulation"""

        try:
            # Initialize Claude client if needed
            self._init_claude_client()
            
            print(f"🤖 Asking Claude Haiku for {n_terms} MeSH terms for: '{query}'")
            
            message = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent terminology
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse the response to extract MeSH terms
            response_text = message.content[0].text.strip()
            mesh_terms = [term.strip() for term in response_text.split('\n') if term.strip()]
            
            # Take only the requested number of terms
            mesh_terms = mesh_terms[:n_terms]
            
            print(f"🔬 Generated MeSH terms: {', '.join(mesh_terms)}")
            return mesh_terms
            
        except Exception as e:
            print(f"❌ Error generating MeSH terms: {e}")
            print(f"🔄 Falling back to keyword search")
            return []  # Return empty list instead of original query
    
    def create_expanded_query(self, query: str, mesh_terms: List[str]) -> str:
        """
        Create expanded PubMed query using MeSH terms with OR concatenation
        
        Args:
            query: Original query
            mesh_terms: List of MeSH terms (real MeSH headings)
            
        Returns:
            Expanded query string for PubMed search
        """
        if not mesh_terms:
            # No MeSH terms available, use original query as keywords
            print(f"📝 Using keyword search: {query}")
            return query
        
        # Create OR-concatenated MeSH query for real MeSH terms
        mesh_query = " OR ".join(f'"{term}"[MeSH Terms]' for term in mesh_terms)
        
        # Also include the original query as keywords for broader coverage
        expanded_query = f"({mesh_query}) OR ({query})"
        
        print(f"📝 Expanded PubMed query: {expanded_query}")
        return expanded_query
    
    def search_gene_query(self, gene: str, expanded_query: str, max_results: int = 10) -> List[str]:
        """Search PubMed for gene + expanded query, return PMIDs"""
        search_term = f"{gene} AND {expanded_query}"
        
        params = {
            'db': 'pubmed',
            'term': search_term,
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance'
        }
        
        try:
            print(f"🔍 Searching: '{gene} AND {expanded_query}'")
            response = self.session.get(f"{self.base_url}esearch.fcgi", params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get('esearchresult', {}).get('idlist', [])
            print(f"Found {len(pmids)} papers for {gene}")
            return pmids
            
        except Exception as e:
            print(f"Error searching {gene}: {e}")
            return []
    
    def fetch_abstracts(self, pmids: List[str]) -> List[Dict[str, str]]:
        """Fetch abstracts for PMIDs"""
        if not pmids:
            return []
        
        pmid_str = ','.join(pmids)
        params = {
            'db': 'pubmed',
            'id': pmid_str,
            'retmode': 'xml',
            'rettype': 'abstract'
        }
        
        try:
            response = self.session.get(f"{self.base_url}efetch.fcgi", params=params, timeout=60)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            abstracts = []
            
            for article in root.findall('.//PubmedArticle'):
                try:
                    pmid_elem = article.find('.//PMID')
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text
                    
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else ""
                    
                    abstract_elem = article.find('.//Abstract/AbstractText')
                    abstract = abstract_elem.text if abstract_elem is not None else ""
                    
                    if abstract and abstract.strip():  # Only include papers with abstracts
                        abstracts.append({
                            'pmid': pmid,
                            'title': title,
                            'abstract': abstract
                        })
                
                except Exception:
                    continue
            
            print(f"Retrieved {len(abstracts)} abstracts")
            return abstracts
            
        except Exception as e:
            print(f"Error fetching abstracts: {e}")
            return []
    
    def compute_similarity(self, query: str, abstract: str) -> float:
        """
        Compute cosine similarity between query and abstract using SentenceTransformer
        Note: We score against the ORIGINAL query, not the expanded MeSH query
        
        Args:
            query: Original user query string
            abstract: Abstract text (title + abstract combined)
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.model is None:
            self.load_model()
        
        # Encode both texts
        embeddings = self.model.encode([query, abstract])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]
        
        return float(similarity_score)
    
    def score_gene_list(self, 
                       genes: List[str], 
                       query: str, 
                       top_papers: int = 10,
                       n_mesh_terms: int = 3) -> float:
        """
        Score gene list using Claude-enhanced approach:
        1. Generate MeSH terms using Claude Haiku
        2. Create expanded PubMed query
        3. Search each gene + expanded query
        4. Pool all abstracts
        5. Score each against ORIGINAL query (not expanded)
        6. Sum similarities
        """
        print(f"\n🧬 Claude-Enhanced Gene-Query Similarity Scoring")
        print(f"Genes: {', '.join(genes)}")
        print(f"Original Query: '{query}'")
        print(f"MeSH terms to generate: {n_mesh_terms}")
        print(f"Method: Claude Haiku + MeSH + SentenceTransformer")
        print("="*80)
        
        # Step 1: Generate MeSH terms using Claude
        mesh_terms = self.generate_mesh_terms(query, n_mesh_terms)
        
        # Step 2: Create expanded query for PubMed search
        expanded_query = self.create_expanded_query(query, mesh_terms)
        
        # Step 3: Collect abstracts from all gene searches using expanded query
        all_abstracts = []
        
        for i, gene in enumerate(genes, 1):
            print(f"\n[{i}/{len(genes)}] Processing {gene}...")
            
            pmids = self.search_gene_query(gene, expanded_query, max_results=top_papers)
            if pmids:
                abstracts = self.fetch_abstracts(pmids)
                for abstract in abstracts:
                    abstract['source_gene'] = gene
                all_abstracts.extend(abstracts)
            
            # Rate limiting
            if i < len(genes):
                time.sleep(1)
        
        if not all_abstracts:
            print("❌ No abstracts found")
            return 0.0
        
        print(f"\n📚 Total abstracts collected: {len(all_abstracts)}")
        
        # Step 4: Score each abstract against the ORIGINAL query (not expanded)
        print(f"🎯 Scoring against ORIGINAL query: '{query}'")
        
        similarities = []
        for abstract in all_abstracts:
            text = f"{abstract['title']} {abstract['abstract']}"
            similarity = self.compute_similarity(query, text)  # Score against original query
            similarities.append(similarity)
        
        # Step 5: Calculate results
        total_score = sum(similarities)
        
        print(f"\n🎯 Results:")
        print(f"MeSH terms used: {', '.join(mesh_terms)}")
        print(f"Total similarity score: {total_score:.3f}")
        print(f"Average per abstract: {total_score/len(all_abstracts):.3f}")
        print(f"Average per gene: {total_score/len(genes):.3f}")
        
        # Show top abstracts
        abstract_scores = list(zip(all_abstracts, similarities))
        abstract_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Top 10 Most Similar Abstracts:")
        for i, (abstract, score) in enumerate(abstract_scores[:10], 1):
            title = abstract['title'][:60] + "..." if len(abstract['title']) > 60 else abstract['title']
            print(f"  {i}. [{score:.3f}] (Gene: {abstract['source_gene']}) {title}")
        
        return total_score

    def score_gene_list_with_manual_mesh(self, 
                                       genes: List[str], 
                                       query: str, 
                                       mesh_terms: List[str],
                                       top_papers: int = 10) -> float:
        """
        Score gene list using manual MeSH terms (no Claude required):
        1. Use provided MeSH terms
        2. Create expanded PubMed query
        3. Search each gene + expanded query
        4. Pool all abstracts
        5. Score each against ORIGINAL query (not expanded)
        6. Sum similarities
        """
        print(f"\n🧬 Manual MeSH Gene-Query Similarity Scoring")
        print(f"Genes: {', '.join(genes)}")
        print(f"Original Query: '{query}'")
        print(f"Manual MeSH terms: {', '.join(mesh_terms)}")
        print(f"Method: Manual MeSH + SentenceTransformer")
        print("="*80)
        
        # Step 1: Create expanded query for PubMed search using provided MeSH terms
        expanded_query = self.create_expanded_query(query, mesh_terms)
        
        # Step 2: Collect abstracts from all gene searches using expanded query
        all_abstracts = []
        for i, gene in enumerate(genes):
            print(f"\n🔍 [{i+1}/{len(genes)}] Searching {gene}...")
            pmids = self.search_gene_query(gene, expanded_query, top_papers)
            if pmids:
                abstracts = self.fetch_abstracts(pmids)
                # Tag abstracts with source gene
                for abstract in abstracts:
                    abstract['source_gene'] = gene
                all_abstracts.extend(abstracts)
            
            # Rate limiting
            if i < len(genes):
                time.sleep(1)
        
        if not all_abstracts:
            print("❌ No abstracts found")
            return 0.0
        
        print(f"\n📚 Total abstracts collected: {len(all_abstracts)}")
        
        # Step 3: Score each abstract against the ORIGINAL query (not expanded)
        print(f"🎯 Scoring against ORIGINAL query: '{query}'")
        
        similarities = []
        for abstract in all_abstracts:
            text = f"{abstract['title']} {abstract['abstract']}"
            similarity = self.compute_similarity(query, text)  # Score against original query
            similarities.append(similarity)
        
        # Step 4: Calculate results
        total_score = sum(similarities)
        
        print(f"\n🎯 Results:")
        print(f"MeSH terms used: {', '.join(mesh_terms)}")
        print(f"Total similarity score: {total_score:.3f}")
        print(f"Average per abstract: {total_score/len(all_abstracts):.3f}")
        print(f"Average per gene: {total_score/len(genes):.3f}")
        
        # Show top abstracts
        abstract_scores = list(zip(all_abstracts, similarities))
        abstract_scores.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 Top 10 Most Similar Abstracts:")
        for i, (abstract, score) in enumerate(abstract_scores[:10], 1):
            title = abstract['title'][:60] + "..." if len(abstract['title']) > 60 else abstract['title']
            print(f"  {i}. [{score:.3f}] (Gene: {abstract['source_gene']}) {title}")
        
        return total_score


def main():
    parser = argparse.ArgumentParser(
        description="Claude-enhanced gene-query similarity scorer with MeSH term generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with 3 MeSH terms (default)
  python claude_enhanced_gene_scorer.py --genes TP53 BRCA1 --query "cancer progression" --top-papers 10
  
  # Custom number of MeSH terms
  python claude_enhanced_gene_scorer.py --genes IL6 TNF NFKB1 --query "inflammatory response" --mesh-terms 5 --top-papers 15
  
  # With custom model
  python claude_enhanced_gene_scorer.py --genes APOE PSEN1 --query "Alzheimer disease" --model sentence-transformers/all-mpnet-base-v2

Environment Variables:
  ANTHROPIC_API_KEY - Required for Claude Haiku API access
        """
    )
    
    parser.add_argument("--genes", "-g", nargs='+', required=True, help="Gene symbols")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--top-papers", "-n", type=int, default=10, help="Papers per gene (default: 10)")
    parser.add_argument("--mesh-terms", "-m", nargs='*', help="Manual MeSH terms to use (if provided, skips Claude generation)")
    parser.add_argument("--n-mesh-terms", type=int, default=3, help="Number of MeSH terms to generate with Claude (default: 3)")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    parser.add_argument("--claude-key", help="Claude API key (overrides ANTHROPIC_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.mesh_terms is None and (args.n_mesh_terms < 1 or args.n_mesh_terms > 10):
        print("❌ Number of MeSH terms to generate must be between 1 and 10")
        exit(1)
    
    try:
        scorer = ClaudeEnhancedPubMedScorer(
            model_name=args.model,
            claude_api_key=args.claude_key
        )
        
        # Use manual MeSH terms if provided, otherwise generate with Claude
        if args.mesh_terms:
            print(f"ℹ️ Using manual MeSH terms: {args.mesh_terms}")
            total_score = scorer.score_gene_list_with_manual_mesh(
                genes=args.genes, 
                query=args.query, 
                mesh_terms=args.mesh_terms,
                top_papers=args.top_papers
            )
        else:
            total_score = scorer.score_gene_list(
                genes=args.genes, 
                query=args.query, 
                top_papers=args.top_papers,
                n_mesh_terms=args.n_mesh_terms
            )
        
        print(f"\n✅ Final Score: {total_score:.3f}")
        
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("Make sure to set ANTHROPIC_API_KEY environment variable or use --claude-key")
        exit(1)
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        exit(1)


if __name__ == "__main__":
    main()
