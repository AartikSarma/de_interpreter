"""Paper data structure."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Paper:
    """Container for research paper data."""
    
    pmid: str
    title: str
    abstract: Optional[str]
    authors: List[str]
    journal: str
    publication_date: Optional[datetime]
    doi: Optional[str] = None
    full_text: Optional[str] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.mesh_terms is None:
            self.mesh_terms = []
    
    @property
    def citation(self) -> str:
        """Generate a basic citation for this paper."""
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        
        year = self.publication_date.year if self.publication_date else "Unknown"
        return f"{author_str}. {self.title}. {self.journal}. {year}."
    
    @property
    def text_content(self) -> str:
        """Get the full text content or abstract as fallback."""
        if self.full_text:
            return self.full_text
        return self.abstract or ""