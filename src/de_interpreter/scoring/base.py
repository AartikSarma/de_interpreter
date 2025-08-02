"""
Abstract base class for all scorers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd


class BaseScorer(ABC):
    """
    Abstract base class for literature relevance scorers.
    
    All scoring implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, name: str = None):
        """
        Initialize the scorer.
        
        Args:
            name: Human-readable name for this scorer
        """
        self.name = name or self.__class__.__name__
        
    @abstractmethod
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score a list of documents against a query.
        
        Args:
            query: The search query (e.g., DE analysis context)
            documents: List of documents to score, each containing
                      at least 'title' and 'abstract' fields
                      
        Returns:
            List of relevance scores, one per document
        """
        pass
    
    @abstractmethod
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document against a query.
        
        Args:
            query: The search query
            document: Document to score with 'title' and 'abstract' fields
            
        Returns:
            Relevance score for the document
        """
        pass
    
    def rank_documents(self, 
                      query: str, 
                      documents: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """
        Rank documents by relevance score.
        
        Args:
            query: The search query
            documents: List of documents to rank
            
        Returns:
            List of (document_index, score) tuples sorted by score (descending)
        """
        scores = self.score_documents(query, documents)
        ranked = [(i, score) for i, score in enumerate(scores)]
        return sorted(ranked, key=lambda x: x[1], reverse=True)
    
    def prepare_query(self, de_results: pd.DataFrame, **kwargs) -> str:
        """
        Prepare a query from differential expression results.
        
        Args:
            de_results: DataFrame with DE analysis results
            **kwargs: Additional parameters for query preparation
            
        Returns:
            Formatted query string
        """
        # Default implementation - can be overridden by subclasses
        top_genes = de_results.nlargest(10, 'padj_neg_log10')['gene_symbol'].tolist()
        return " ".join(top_genes)
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self) -> str:
        return str(self)
