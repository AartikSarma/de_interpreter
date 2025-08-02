#!/usr/bin/env python3
"""
Score similarity between a query and a PubMed abstract by PMID.

This script fetches an abstract from PubMed using its PMID and computes
semantic similarity with a user query using sentence-transformers.
"""

import argparse
import sys
import time
from typing import Optional, Tuple

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import requests
    from xml.etree import ElementTree as ET
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: conda activate de_interpreter")
    sys.exit(1)


class PubMedFetcher:
    """Fetch abstracts from PubMed using the E-utilities API"""
    
    def __init__(self):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        
    def fetch_abstract(self, pmid: str) -> Optional[Tuple[str, str]]:
        """
        Fetch abstract and title for a given PMID.
        
        Args:
            pmid: PubMed ID as string
            
        Returns:
            Tuple of (title, abstract) or None if not found
        """
        try:
            # Construct the efetch URL
            url = f"{self.base_url}efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': pmid,
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            print(f"Fetching PMID {pmid} from PubMed...")
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find the article
            article = root.find('.//PubmedArticle')
            if article is None:
                print(f"❌ No article found for PMID {pmid}")
                return None
            
            # Extract title
            title_elem = article.find('.//ArticleTitle')
            title = title_elem.text if title_elem is not None else "No title"
            
            # Extract abstract
            abstract_elem = article.find('.//Abstract/AbstractText')
            if abstract_elem is not None:
                abstract = abstract_elem.text or ""
            else:
                print(f"⚠️  No abstract found for PMID {pmid}")
                abstract = ""
            
            if not abstract.strip():
                print(f"⚠️  Empty abstract for PMID {pmid}")
                return title, ""
            
            print(f"✅ Successfully fetched abstract ({len(abstract)} characters)")
            return title, abstract
            
        except requests.RequestException as e:
            print(f"❌ Network error fetching PMID {pmid}: {e}")
            return None
        except ET.ParseError as e:
            print(f"❌ XML parsing error for PMID {pmid}: {e}")
            return None
        except Exception as e:
            print(f"❌ Unexpected error fetching PMID {pmid}: {e}")
            return None


class QueryAbstractScorer:
    """Score similarity between queries and abstracts using BioBERT"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the scorer with a specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        self.model_name = model_name
        self.model = None
        
    def load_model(self):
        """Load the sentence transformer model"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.1f}s")
    
    def score_similarity(self, query: str, abstract: str) -> float:
        """
        Compute cosine similarity between query and abstract.
        
        Args:
            query: User query string
            abstract: PubMed abstract text
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.model is None:
            self.load_model()
        
        # Encode both texts
        print("Encoding query and abstract...")
        embeddings = self.model.encode([query, abstract])
        
        # Compute cosine similarity
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]
        
        return float(similarity_score)


def main():
    """Main function to handle command line arguments and run scoring"""
    parser = argparse.ArgumentParser(
        description="Score similarity between a query and PubMed abstract",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python score_query_vs_pmid.py "ARG1 pancreatic cancer" 12345678
  python score_query_vs_pmid.py "What is the role of p53 in cancer?" 87654321
        """
    )
    
    parser.add_argument(
        "query",
        help="Query string to compare against the abstract"
    )
    
    parser.add_argument(
        "pmid",
        help="PubMed ID (PMID) of the article to fetch"
    )
    
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence transformer model to use (default: all-MiniLM-L6-v2)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output including title and abstract excerpt"
    )
    
    args = parser.parse_args()
    
    # Initialize components
    fetcher = PubMedFetcher()
    scorer = QueryAbstractScorer(model_name=args.model)
    
    print("="*80)
    print(f"Query: '{args.query}'")
    print(f"PMID: {args.pmid}")
    print(f"Model: {args.model}")
    print("="*80)
    
    # Fetch abstract
    result = fetcher.fetch_abstract(args.pmid)
    if result is None:
        print("❌ Failed to fetch abstract")
        sys.exit(1)
    
    title, abstract = result
    
    if not abstract.strip():
        print("❌ No abstract available for scoring")
        sys.exit(1)
    
    # Show details if verbose
    if args.verbose:
        print(f"\nTitle: {title}")
        print(f"\nAbstract preview: {abstract[:200]}...")
        print(f"Abstract length: {len(abstract)} characters")
    
    # Score similarity
    try:
        similarity_score = scorer.score_similarity(args.query, abstract)
        
        print(f"\n📊 Similarity Score: {similarity_score:.3f}")
        
        # Interpret the score
        if similarity_score >= 0.7:
            interpretation = "🟢 Highly relevant"
        elif similarity_score >= 0.5:
            interpretation = "🟡 Moderately relevant"
        elif similarity_score >= 0.3:
            interpretation = "🟠 Somewhat relevant"
        else:
            interpretation = "🔴 Low relevance"
        
        print(f"Interpretation: {interpretation}")
        
        if args.verbose:
            print(f"\nModel used: {args.model}")
            print(f"Embedding dimension: {scorer.model.get_sentence_embedding_dimension()}")
        
    except Exception as e:
        print(f"❌ Error computing similarity: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
