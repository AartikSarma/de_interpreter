"""
BioBERT-based scorer for literature relevance.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import cosine_similarity

from ..base import BaseScorer
from .models import BioBERTModel
from .utils import preprocess_biomedical_text


class BioBERTScorer(BaseScorer):
    """
    Scorer using BioBERT embeddings for semantic similarity.
    
    This scorer uses pre-trained BioBERT models to generate embeddings
    for queries and documents, then computes cosine similarity scores.
    """
    
    def __init__(self, 
                 model_name: str = "dmis-lab/biobert-base-cased-v1.1",
                 device: Optional[str] = None,
                 batch_size: int = 8,
                 max_length: int = 512,
                 cache_embeddings: bool = True):
        """
        Initialize BioBERT scorer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
            batch_size: Batch size for processing documents
            max_length: Maximum sequence length for tokenization
            cache_embeddings: Whether to cache computed embeddings
        """
        super().__init__(name="BioBERT")
        
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_embeddings = cache_embeddings
        
        # Initialize model
        self.model = BioBERTModel(model_name, device=self.device)
        self.embedding_cache = {} if cache_embeddings else None
        
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for a text string.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor
        """
        if self.cache_embeddings and text in self.embedding_cache:
            return self.embedding_cache[text]
            
        # Preprocess text
        processed_text = preprocess_biomedical_text(text)
        
        # Get embedding
        embedding = self.model.encode(processed_text, max_length=self.max_length)
        
        if self.cache_embeddings:
            self.embedding_cache[text] = embedding
            
        return embedding
    
    def _prepare_document_text(self, document: Dict[str, Any]) -> str:
        """
        Prepare document text for embedding.
        
        Args:
            document: Document with 'title' and 'abstract' fields
            
        Returns:
            Combined text string
        """
        title = document.get('title', '')
        abstract = document.get('abstract', '')
        
        # Combine title and abstract with proper formatting
        if title and abstract:
            return f"{title}. {abstract}"
        elif title:
            return title
        elif abstract:
            return abstract
        else:
            return ""
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document against a query using BioBERT embeddings.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Cosine similarity score between query and document
        """
        doc_text = self._prepare_document_text(document)
        
        if not doc_text.strip():
            return 0.0
            
        # Get embeddings
        query_embedding = self._get_text_embedding(query)
        doc_embedding = self._get_text_embedding(doc_text)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            query_embedding.unsqueeze(0),
            doc_embedding.unsqueeze(0)
        ).item()
        
        return float(similarity)
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple documents against a query.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of similarity scores
        """
        if not documents:
            return []
            
        # Get query embedding once
        query_embedding = self._get_text_embedding(query)
        
        scores = []
        for doc in documents:
            doc_text = self._prepare_document_text(doc)
            
            if not doc_text.strip():
                scores.append(0.0)
                continue
                
            doc_embedding = self._get_text_embedding(doc_text)
            
            similarity = cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_embedding.unsqueeze(0)
            ).item()
            
            scores.append(float(similarity))
            
        return scores
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.embedding_cache is not None:
            self.embedding_cache.clear()
    
    def get_cache_size(self) -> int:
        """Get the number of cached embeddings."""
        return len(self.embedding_cache) if self.embedding_cache else 0
