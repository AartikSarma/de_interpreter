#!/usr/bin/env python3
"""
Minimal Gene-Query Similarity Scorer (Same similarity as score_query_vs_pmid.py)

This is a simplified version that uses:
1. PubMed E-utilities API (free)
2. SentenceTransformer + cosine similarity (same as score_query_vs_pmid.py)

Dependencies: sentence-transformers, scikit-learn, requests
"""

import requests
import json
import time
import argparse
from typing import List, Dict
from xml.etree import ElementTree as ET

#turn off future warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Run: pip install sentence-transformers scikit-learn requests")
    exit(1)


class SimplePubMedScorer:
    """Minimal implementation using PubMed API and SentenceTransformer similarity (same as score_query_vs_pmid.py)"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GeneSimilarityScorer/1.0'
        })
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """Load the sentence transformer model (same as score_query_vs_pmid.py)"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            start_time = time.time()
            self.model = SentenceTransformer(self.model_name)
            load_time = time.time() - start_time
            print(f"Model loaded in {load_time:.1f}s")
    
    def search_gene_query(self, gene: str, query: str, max_results: int = 10) -> List[str]:
        """Search PubMed for gene + query, return PMIDs"""
        search_term = f"{query} {gene}"
        
        params = {
            'db': 'pubmed',
            'term': search_term,
            'retmode': 'json',
            'retmax': max_results,
            'sort': 'relevance'
        }
        
        try:
            print(f"Searching: '{search_term}'")
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
        (EXACT same method as score_query_vs_pmid.py)
        
        Args:
            query: User query string
            abstract: Abstract text (title + abstract combined)
            
        Returns:
            Cosine similarity score (0-1)
        """
        if self.model is None:
            self.load_model()
        
        # Encode both texts
        embeddings = self.model.encode([query, abstract])
        
        # Compute cosine similarity (same as score_query_vs_pmid.py)
        similarity_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
        similarity_score = similarity_matrix[0][0]
        
        return float(similarity_score)
    
    def score_gene_list(self, genes: List[str], query: str, top_papers: int = 10) -> float:
        """
        Score gene list using simplified approach:
        1. Search query + each gene
        2. Pool all abstracts  
        3. Score each against query + all genes
        4. Sum similarities
        """
        print(f"\n🧬 Minimal Gene-Query Similarity Scoring")
        print(f"Genes: {', '.join(genes)}")
        print(f"Query: '{query}'")
        print(f"Method: SentenceTransformer + Cosine similarity (same as score_query_vs_pmid.py)")
        print("="*60)
        
        # Step 1: Collect abstracts from all gene searches
        all_abstracts = []
        
        for i, gene in enumerate(genes, 1):
            print(f"\n[{i}/{len(genes)}] Processing {gene}...")
            
            pmids = self.search_gene_query(gene, query, max_results=top_papers)
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
        
        # Step 2: Create combined query
        combined_query = f"{query} {' '.join(genes)}"
        print(f"🎯 Scoring against: '{combined_query}'")
        
        # Step 3: Score each abstract using SentenceTransformer (same as score_query_vs_pmid.py)
        similarities = []
        for abstract in all_abstracts:
            text = f"{abstract['title']} {abstract['abstract']}"
            similarity = self.compute_similarity(combined_query, text)
            similarities.append(similarity)
        
        # Step 4: Calculate results
        total_score = sum(similarities)
        
        print(f"\n🎯 Results:")
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
    parser = argparse.ArgumentParser(description="Minimal gene-query similarity scorer (same similarity as score_query_vs_pmid.py)")
    
    parser.add_argument("--genes", "-g", nargs='+', required=True, help="Gene symbols")
    parser.add_argument("--query", "-q", required=True, help="Search query")
    parser.add_argument("--top-papers", "-n", type=int, default=10, help="Papers per gene")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformer model")
    
    args = parser.parse_args()
    
    scorer = SimplePubMedScorer(model_name=args.model)
    total_score = scorer.score_gene_list(args.genes, args.query, args.top_papers)
    
    print(f"\n✅ Final Score: {total_score:.3f}")


if __name__ == "__main__":
    main()
