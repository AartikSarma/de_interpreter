"""
BM25 scoring for literature relevance.
"""

from typing import List, Dict, Any, Optional
import re
import math
from collections import defaultdict

from ..base import BaseScorer


class BM25Scorer(BaseScorer):
    """
    BM25 (Best Matching 25) scorer for information retrieval.
    
    BM25 is a probabilistic ranking function that improves upon TF-IDF
    by incorporating document length normalization and term saturation.
    """
    
    def __init__(self,
                 k1: float = 1.5,
                 b: float = 0.75,
                 stop_words: Optional[List[str]] = None,
                 lowercase: bool = True):
        """
        Initialize BM25 scorer.
        
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Document length normalization parameter (0-1)
            stop_words: List of stop words to exclude
            lowercase: Whether to convert to lowercase
        """
        super().__init__(name="BM25")
        
        self.k1 = k1
        self.b = b
        self.stop_words = set(stop_words or self._default_stop_words())
        self.lowercase = lowercase
        
        # BM25 model state
        self.document_frequencies = {}
        self.document_lengths = []
        self.average_document_length = 0.0
        self.total_documents = 0
        self.fitted = False
        
    def _default_stop_words(self) -> List[str]:
        """Default English stop words."""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their'
        ]
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
            
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        # Simple tokenization using regex
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter stop words and short tokens
        tokens = [
            token for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def fit(self, documents: List[Dict[str, Any]]):
        """
        Fit the BM25 model on a corpus of documents.
        
        Args:
            documents: List of documents with 'title' and 'abstract' fields
        """
        self.total_documents = len(documents)
        self.document_frequencies = defaultdict(int)
        self.document_lengths = []
        
        # Process each document
        for doc in documents:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            text = f"{title} {abstract}".strip()
            
            tokens = self._tokenize(text)
            self.document_lengths.append(len(tokens))
            
            # Count unique terms in this document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.document_frequencies[term] += 1
        
        # Compute average document length
        if self.document_lengths:
            self.average_document_length = sum(self.document_lengths) / len(self.document_lengths)
        else:
            self.average_document_length = 0.0
            
        self.fitted = True
    
    def _compute_idf(self, term: str) -> float:
        """
        Compute IDF score for a term.
        
        Args:
            term: Term to compute IDF for
            
        Returns:
            IDF score
        """
        if term not in self.document_frequencies:
            return 0.0
            
        df = self.document_frequencies[term]
        return math.log((self.total_documents - df + 0.5) / (df + 0.5))
    
    def _compute_bm25_score(self, query_terms: List[str], document_text: str) -> float:
        """
        Compute BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            document_text: Document text
            
        Returns:
            BM25 score
        """
        doc_tokens = self._tokenize(document_text)
        doc_length = len(doc_tokens)
        
        if doc_length == 0:
            return 0.0
            
        # Count term frequencies in document
        term_freqs = defaultdict(int)
        for token in doc_tokens:
            term_freqs[token] += 1
        
        score = 0.0
        
        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = self._compute_idf(term)
                
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.average_document_length)
                )
                
                score += idf * (numerator / denominator)
                
        return score
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document against a query using BM25.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            BM25 score
        """
        if not self.fitted:
            # If not fitted, fit on just this document (not ideal but functional)
            self.fit([document])
            
        # Prepare document text
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        doc_text = f"{title} {abstract}".strip()
        
        if not doc_text:
            return 0.0
            
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return 0.0
            
        return self._compute_bm25_score(query_terms, doc_text)
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple documents against a query using BM25.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of BM25 scores
        """
        if not documents:
            return []
            
        # Fit model if not already fitted
        if not self.fitted:
            self.fit(documents)
            
        # Tokenize query
        query_terms = self._tokenize(query)
        
        if not query_terms:
            return [0.0] * len(documents)
            
        scores = []
        for doc in documents:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            doc_text = f"{title} {abstract}".strip()
            
            if not doc_text:
                scores.append(0.0)
                continue
                
            score = self._compute_bm25_score(query_terms, doc_text)
            scores.append(score)
            
        return scores
    
    def get_term_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the fitted model.
        
        Returns:
            Dictionary with model statistics
        """
        if not self.fitted:
            return {}
            
        return {
            'total_documents': self.total_documents,
            'unique_terms': len(self.document_frequencies),
            'average_document_length': self.average_document_length,
            'min_document_length': min(self.document_lengths) if self.document_lengths else 0,
            'max_document_length': max(self.document_lengths) if self.document_lengths else 0,
        }
    
    def get_top_terms_by_frequency(self, n: int = 10) -> List[tuple]:
        """
        Get top terms by document frequency.
        
        Args:
            n: Number of top terms to return
            
        Returns:
            List of (term, frequency) tuples
        """
        if not self.fitted:
            return []
            
        sorted_terms = sorted(
            self.document_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_terms[:n]
