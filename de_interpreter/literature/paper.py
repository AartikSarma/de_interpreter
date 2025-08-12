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
    def pmc_url(self) -> str:
        """Generate PMC URL for this paper."""
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"
    
    def to_claude_source(self) -> Dict[str, Any]:
        """Convert paper to Claude API source format for citations."""
        return {
            "type": "document",
            "id": f"pmid_{self.pmid}",
            "content": self.text_content[:8000],  # Limit content size for API
            "metadata": {
                "title": self.title,
                "authors": self.authors,
                "journal": self.journal,
                "pmid": self.pmid,
                "url": self.pmc_url,
                "year": self.publication_date.year if self.publication_date else None,
                "doi": self.doi,
                "citation": self.citation
            }
        }
    
    def to_formatted_citation(self) -> str:
        """Generate formatted citation with link for reports."""
        citation_text = self.citation
        if self.pmid:
            return f"[{citation_text}]({self.pmc_url})"
        return citation_text
    
    @property
    def text_content(self) -> str:
        """Get the full text content or abstract as fallback."""
        if self.full_text:
            return self.full_text
        return self.abstract or ""