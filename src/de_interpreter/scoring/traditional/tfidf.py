"""
TF-IDF based scoring for literature relevance.
"""

from typing import List, Dict, Any, Optional
import re
import math
from collections import Counter, defaultdict

from ..base import BaseScorer


class TFIDFScorer(BaseScorer):
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) scorer.
    
    This scorer computes TF-IDF vectors for queries and documents,
    then calculates cosine similarity for relevance scoring.
    """
    
    def __init__(self, 
                 min_df: int = 1,
                 max_df_ratio: float = 0.8,
                 stop_words: Optional[List[str]] = None,
                 lowercase: bool = True):
        """
        Initialize TF-IDF scorer.
        
        Args:
            min_df: Minimum document frequency for terms
            max_df_ratio: Maximum document frequency ratio (0-1)
            stop_words: List of stop words to exclude
            lowercase: Whether to convert to lowercase
        """
        super().__init__(name="TF-IDF")
        
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.stop_words = set(stop_words or self._default_stop_words())
        self.lowercase = lowercase
        
        # TF-IDF model state
        self.vocabulary = {}
        self.idf_scores = {}
        self.document_count = 0
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
    
    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary of term frequencies
        """
        if not tokens:
            return {}
            
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        return {
            token: count / total_tokens 
            for token, count in token_counts.items()
        }
    
    def _build_vocabulary(self, documents: List[str]):
        """
        Build vocabulary and compute IDF scores.
        
        Args:
            documents: List of document texts
        """
        # Count document frequencies
        doc_freq = defaultdict(int)
        self.document_count = len(documents)
        
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1
        
        # Filter by min/max document frequency
        max_df = int(self.max_df_ratio * self.document_count)
        
        self.vocabulary = {
            token: idx for idx, (token, freq) in enumerate(doc_freq.items())
            if self.min_df <= freq <= max_df
        }
        
        # Compute IDF scores
        self.idf_scores = {
            token: math.log(self.document_count / doc_freq[token])
            for token in self.vocabulary
        }
        
        self.fitted = True
    
    def _compute_tfidf_vector(self, text: str) -> Dict[str, float]:
        """
        Compute TF-IDF vector for text.
        
        Args:
            text: Input text
            
        Returns:
            TF-IDF vector as dictionary
        """
        if not self.fitted:
            raise ValueError("Scorer must be fitted before computing TF-IDF vectors")
            
        tokens = self._tokenize(text)
        tf_scores = self._compute_tf(tokens)
        
        tfidf_vector = {}
        for token, tf in tf_scores.items():
            if token in self.vocabulary:
                tfidf_vector[token] = tf * self.idf_scores[token]
                
        return tfidf_vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two TF-IDF vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Get common terms
        common_terms = set(vec1.keys()) & set(vec2.keys())
        
        if not common_terms:
            return 0.0
            
        # Compute dot product
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Compute magnitudes
        mag1 = math.sqrt(sum(val**2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val**2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
            
        return dot_product / (mag1 * mag2)
    
    def fit(self, documents: List[Dict[str, Any]]):
        """
        Fit the TF-IDF model on a corpus of documents.
        
        Args:
            documents: List of documents with 'title' and 'abstract' fields
        """
        # Prepare document texts
        doc_texts = []
        for doc in documents:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            text = f"{title} {abstract}".strip()
            doc_texts.append(text)
            
        self._build_vocabulary(doc_texts)
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document against a query using TF-IDF.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            TF-IDF cosine similarity score
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
            
        # Compute TF-IDF vectors
        query_vector = self._compute_tfidf_vector(query)
        doc_vector = self._compute_tfidf_vector(doc_text)
        
        # Compute similarity
        return self._cosine_similarity(query_vector, doc_vector)
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple documents against a query.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of TF-IDF similarity scores
        """
        if not documents:
            return []
            
        # Fit model if not already fitted
        if not self.fitted:
            self.fit(documents)
            
        # Compute query vector
        query_vector = self._compute_tfidf_vector(query)
        
        scores = []
        for doc in documents:
            title = doc.get('title', '')
            abstract = doc.get('abstract', '')
            doc_text = f"{title} {abstract}".strip()
            
            if not doc_text:
                scores.append(0.0)
                continue
                
            doc_vector = self._compute_tfidf_vector(doc_text)
            similarity = self._cosine_similarity(query_vector, doc_vector)
            scores.append(similarity)
            
        return scores
    
    def get_vocabulary_size(self) -> int:
        """Get the size of the fitted vocabulary."""
        return len(self.vocabulary)
    
    def get_top_terms(self, n: int = 10) -> List[str]:
        """
        Get top terms by IDF score.
        
        Args:
            n: Number of top terms to return
            
        Returns:
            List of top terms
        """
        if not self.fitted:
            return []
            
        sorted_terms = sorted(
            self.idf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [term for term, score in sorted_terms[:n]]
