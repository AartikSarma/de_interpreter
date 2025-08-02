"""
Jaccard similarity scoring for literature relevance.
"""

from typing import List, Dict, Any, Set, Optional
import re

from ..base import BaseScorer


class JaccardScorer(BaseScorer):
    """
    Jaccard similarity scorer for simple overlap-based relevance.
    
    Computes Jaccard similarity (intersection over union) between
    query and document term sets.
    """
    
    def __init__(self,
                 stop_words: Optional[List[str]] = None,
                 lowercase: bool = True,
                 min_term_length: int = 2):
        """
        Initialize Jaccard scorer.
        
        Args:
            stop_words: List of stop words to exclude
            lowercase: Whether to convert to lowercase
            min_term_length: Minimum term length to consider
        """
        super().__init__(name="Jaccard")
        
        self.stop_words = set(stop_words or self._default_stop_words())
        self.lowercase = lowercase
        self.min_term_length = min_term_length
        
    def _default_stop_words(self) -> List[str]:
        """Default English stop words."""
        return [
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'do', 'how', 'their'
        ]
    
    def _tokenize(self, text: str) -> Set[str]:
        """
        Tokenize text into a set of terms.
        
        Args:
            text: Input text
            
        Returns:
            Set of tokens
        """
        if not text:
            return set()
            
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
            
        # Simple tokenization using regex
        tokens = re.findall(r'\b\w+\b', text)
        
        # Filter stop words and short tokens
        filtered_tokens = {
            token for token in tokens 
            if (token not in self.stop_words and 
                len(token) >= self.min_term_length)
        }
        
        return filtered_tokens
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Compute Jaccard similarity between two sets.
        
        Args:
            set1: First set of terms
            set2: Second set of terms
            
        Returns:
            Jaccard similarity coefficient (0-1)
        """
        if not set1 and not set2:
            return 1.0  # Both empty sets are identical
            
        if not set1 or not set2:
            return 0.0  # One empty, one non-empty
            
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document against a query using Jaccard similarity.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Jaccard similarity score (0-1)
        """
        # Prepare document text
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        doc_text = f"{title} {abstract}".strip()
        
        if not doc_text:
            return 0.0
            
        # Tokenize query and document
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(doc_text)
        
        # Compute Jaccard similarity
        return self._jaccard_similarity(query_terms, doc_terms)
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple documents against a query using Jaccard similarity.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of Jaccard similarity scores
        """
        if not documents:
            return []
            
        # Tokenize query once
        query_terms = self._tokenize(query)
        
        scores = []
        for doc in documents:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            doc_text = f"{title} {abstract}".strip()
            
            if not doc_text:
                scores.append(0.0)
                continue
                
            doc_terms = self._tokenize(doc_text)
            similarity = self._jaccard_similarity(query_terms, doc_terms)
            scores.append(similarity)
            
        return scores
    
    def compute_overlap_stats(self, 
                             query: str, 
                             document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute detailed overlap statistics between query and document.
        
        Args:
            query: Search query
            document: Document to analyze
            
        Returns:
            Dictionary with overlap statistics
        """
        # Prepare document text
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        doc_text = f"{title} {abstract}".strip()
        
        # Tokenize
        query_terms = self._tokenize(query)
        doc_terms = self._tokenize(doc_text)
        
        # Compute statistics
        intersection = query_terms & doc_terms
        union = query_terms | doc_terms
        
        return {
            'query_terms': list(query_terms),
            'document_terms': list(doc_terms),
            'intersection': list(intersection),
            'union': list(union),
            'query_term_count': len(query_terms),
            'document_term_count': len(doc_terms),
            'intersection_size': len(intersection),
            'union_size': len(union),
            'jaccard_similarity': len(intersection) / len(union) if union else 0.0,
            'query_coverage': len(intersection) / len(query_terms) if query_terms else 0.0,
            'document_coverage': len(intersection) / len(doc_terms) if doc_terms else 0.0,
        }
    
    def find_matching_terms(self, 
                           query: str, 
                           documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find terms that match between query and each document.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            List of match information for each document
        """
        query_terms = self._tokenize(query)
        
        results = []
        for i, doc in enumerate(documents):
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            doc_text = f"{title} {abstract}".strip()
            
            doc_terms = self._tokenize(doc_text)
            matching_terms = query_terms & doc_terms
            
            results.append({
                'document_index': i,
                'matching_terms': list(matching_terms),
                'match_count': len(matching_terms),
                'jaccard_score': self._jaccard_similarity(query_terms, doc_terms)
            })
            
        return results
