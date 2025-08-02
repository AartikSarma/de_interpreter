"""
BioBERT model loading and caching logic.
"""

from typing import Optional, Dict, Any
import os
import torch
from transformers import AutoTokenizer, AutoModel
import hashlib


class BioBERTModel:
    """
    Wrapper for BioBERT model with caching and utilities.
    """
    
    def __init__(self, 
                 model_name: str = "dmis-lab/biobert-base-cased-v1.1",
                 device: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize BioBERT model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = cache_dir
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        ).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
    
    def encode(self, text: str, max_length: int = 512) -> torch.Tensor:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Embedding tensor
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings.squeeze()
    
    def encode_batch(self, texts: list, max_length: int = 512) -> torch.Tensor:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Batch of embedding tensors
        """
        if not texts:
            return torch.empty(0, self.model.config.hidden_size)
            
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        return embeddings


class ModelCache:
    """
    Cache for storing and retrieving model instances.
    """
    
    _cache: Dict[str, BioBERTModel] = {}
    
    @classmethod
    def get_model(cls, 
                  model_name: str,
                  device: Optional[str] = None,
                  cache_dir: Optional[str] = None) -> BioBERTModel:
        """
        Get or create a cached model instance.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on
            cache_dir: Model cache directory
            
        Returns:
            BioBERT model instance
        """
        # Create cache key
        key_parts = [model_name]
        if device:
            key_parts.append(f"device:{device}")
        if cache_dir:
            key_parts.append(f"cache:{cache_dir}")
        
        cache_key = "|".join(key_parts)
        
        if cache_key not in cls._cache:
            cls._cache[cache_key] = BioBERTModel(
                model_name=model_name,
                device=device,
                cache_dir=cache_dir
            )
            
        return cls._cache[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models."""
        cls._cache.clear()
    
    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            'cached_models': list(cls._cache.keys()),
            'cache_size': len(cls._cache)
        }
