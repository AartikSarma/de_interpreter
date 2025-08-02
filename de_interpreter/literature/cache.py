"""Literature caching functionality."""

import json
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .paper import Paper


@dataclass 
class SearchResult:
    """Container for search results."""
    query: str
    papers: List[Paper]
    search_date: datetime
    total_found: int


class LiteratureCache:
    """Cache for literature search results."""
    
    def __init__(self, cache_dir: str = "cache/literature"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[SearchResult]:
        """Get cached search result."""
        cache_key = self._get_cache_key(query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct SearchResult
            papers = []
            for paper_data in data['papers']:
                paper = Paper(
                    pmid=paper_data['pmid'],
                    title=paper_data['title'],
                    abstract=paper_data.get('abstract'),
                    authors=paper_data['authors'],
                    journal=paper_data['journal'],
                    publication_date=datetime.fromisoformat(paper_data['publication_date']) if paper_data.get('publication_date') else None,
                    doi=paper_data.get('doi'),
                    full_text=paper_data.get('full_text'),
                    keywords=paper_data.get('keywords', []),
                    mesh_terms=paper_data.get('mesh_terms', []),
                    relevance_score=paper_data.get('relevance_score')
                )
                papers.append(paper)
            
            return SearchResult(
                query=data['query'],
                papers=papers,
                search_date=datetime.fromisoformat(data['search_date']),
                total_found=data['total_found']
            )
            
        except Exception:
            return None
    
    def set(self, result: SearchResult) -> None:
        """Cache search result."""
        cache_key = self._get_cache_key(result.query)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Convert to serializable format
        papers_data = []
        for paper in result.papers:
            paper_data = {
                'pmid': paper.pmid,
                'title': paper.title,
                'abstract': paper.abstract,
                'authors': paper.authors,
                'journal': paper.journal,
                'publication_date': paper.publication_date.isoformat() if paper.publication_date else None,
                'doi': paper.doi,
                'full_text': paper.full_text,
                'keywords': paper.keywords,
                'mesh_terms': paper.mesh_terms,
                'relevance_score': paper.relevance_score
            }
            papers_data.append(paper_data)
        
        data = {
            'query': result.query,
            'papers': papers_data,
            'search_date': result.search_date.isoformat(),
            'total_found': result.total_found
        }
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()