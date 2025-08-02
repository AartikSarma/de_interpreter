"""
BioBERT-specific utilities for text processing.
"""

import re
from typing import List, Set, Dict, Any
import string


def preprocess_biomedical_text(text: str) -> str:
    """
    Preprocess biomedical text for better embedding quality.
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    if not text:
        return ""
        
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep hyphens and underscores for gene names
    text = re.sub(r'[^\w\s\-_]', ' ', text)
    
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_entities(text: str, entity_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Extract biological entities from text using simple pattern matching.
    
    Args:
        text: Input text
        entity_types: Types of entities to extract (gene, protein, etc.)
        
    Returns:
        Dictionary mapping entity types to lists of found entities
    """
    if entity_types is None:
        entity_types = ['gene', 'protein']
        
    entities = {etype: [] for etype in entity_types}
    
    if not text:
        return entities
        
    # Simple gene/protein pattern (uppercase letters, numbers, hyphens)
    gene_pattern = r'\b[A-Z][A-Z0-9\-_]*[0-9]*\b'
    
    if 'gene' in entity_types or 'protein' in entity_types:
        matches = re.findall(gene_pattern, text)
        # Filter out common false positives
        filtered_matches = [
            m for m in matches 
            if len(m) > 1 and m not in {'AND', 'OR', 'NOT', 'THE', 'FOR'}
        ]
        
        if 'gene' in entity_types:
            entities['gene'] = filtered_matches
        if 'protein' in entity_types:
            entities['protein'] = filtered_matches
            
    return entities


def normalize_gene_names(gene_list: List[str]) -> List[str]:
    """
    Normalize gene names for consistent matching.
    
    Args:
        gene_list: List of gene names
        
    Returns:
        Normalized gene names
    """
    normalized = []
    
    for gene in gene_list:
        if not gene:
            continue
            
        # Convert to uppercase
        gene = gene.upper()
        
        # Remove common prefixes/suffixes
        gene = re.sub(r'^(HUMAN|MOUSE|RAT)[-_]?', '', gene)
        gene = re.sub(r'[-_]?(HUMAN|MOUSE|RAT)$', '', gene)
        
        # Standardize separators
        gene = re.sub(r'[-_]', '-', gene)
        
        normalized.append(gene)
        
    return list(set(normalized))  # Remove duplicates


def create_biomedical_query(gene_symbols: List[str], 
                           condition: str = "",
                           additional_terms: List[str] = None) -> str:
    """
    Create a biomedical query from gene symbols and conditions.
    
    Args:
        gene_symbols: List of gene symbols
        condition: Disease/condition context
        additional_terms: Additional search terms
        
    Returns:
        Formatted query string
    """
    query_parts = []
    
    # Add gene symbols
    if gene_symbols:
        normalized_genes = normalize_gene_names(gene_symbols)
        query_parts.extend(normalized_genes)
    
    # Add condition
    if condition:
        query_parts.append(condition.lower())
        
    # Add additional terms
    if additional_terms:
        query_parts.extend([term.lower() for term in additional_terms])
    
    return " ".join(query_parts)


def extract_mesh_terms(text: str) -> List[str]:
    """
    Extract potential MeSH terms from text using simple heuristics.
    
    Args:
        text: Input text
        
    Returns:
        List of potential MeSH terms
    """
    if not text:
        return []
        
    # Simple patterns for common biomedical terms
    mesh_patterns = [
        r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Two-word capitalized terms
        r'\b\w+osis\b',  # Disease terms ending in -osis
        r'\b\w+itis\b',  # Inflammation terms ending in -itis
        r'\b\w+emia\b',  # Blood condition terms ending in -emia
    ]
    
    potential_terms = set()
    
    for pattern in mesh_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        potential_terms.update(matches)
    
    return list(potential_terms)
