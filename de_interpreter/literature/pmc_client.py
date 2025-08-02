"""Simplified PMC client for literature retrieval."""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any
from datetime import datetime
import xml.etree.ElementTree as ET
import re

from .paper import Paper
from .cache import SearchResult

# Optional scoring import
try:
    from .scoring import LiteratureScorer, ScoringConfig, create_scorer
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False


class PMCClient:
    """Simplified PMC client for literature search."""
    
    def __init__(
        self, 
        use_scoring: bool = False,
        scorer_type: str = "tfidf",
        biobert_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        progress_callback: Optional[callable] = None
    ):
        self.use_scoring = use_scoring and SCORING_AVAILABLE
        self.scorer_type = scorer_type
        self.biobert_model = biobert_model
        self.progress_callback = progress_callback
        self.session = None
        
        # Initialize scorer if requested and available
        self.scorer = None
        if self.use_scoring:
            try:
                self.scorer = create_scorer(scorer_type, biobert_model)
                if self.scorer and not self.scorer.is_available():
                    self.scorer = None
                    self.use_scoring = False
                    if progress_callback:
                        progress_callback("Scoring dependencies not available - continuing without scoring", 0)
            except Exception as e:
                self.scorer = None
                self.use_scoring = False
                if progress_callback:
                    progress_callback(f"Failed to initialize scorer: {e}", 0)
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search PubMed for papers."""
        if self.progress_callback:
            self.progress_callback(f"Searching PMC for: {query[:50]}...", 10)
        
        # Search PubMed
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json"
        }
        
        try:
            async with self.session.get(search_url, params=search_params) as response:
                data = await response.json()
                pmids = data.get("esearchresult", {}).get("idlist", [])
                
            if not pmids:
                return SearchResult(
                    query=query,
                    papers=[],
                    search_date=datetime.now(),
                    total_found=0
                )
            
            if self.progress_callback:
                self.progress_callback(f"Found {len(pmids)} papers, fetching details...", 50)
            
            # Fetch paper details
            papers = await self._fetch_paper_details(pmids)
            
            if self.progress_callback:
                self.progress_callback(f"Retrieved {len(papers)} papers", 70)
            
            # Apply scoring if enabled
            if self.use_scoring and self.scorer and papers:
                if self.progress_callback:
                    self.progress_callback("Scoring papers for relevance...", 80)
                
                try:
                    papers = await self.scorer.score_papers(query, papers, self.progress_callback)
                    if self.progress_callback:
                        self.progress_callback(f"Scored and ranked {len(papers)} papers", 95)
                except Exception as e:
                    if self.progress_callback:
                        self.progress_callback(f"Scoring failed: {e}", 90)
            
            if self.progress_callback:
                self.progress_callback(f"Search complete: {len(papers)} papers", 100)
            
            return SearchResult(
                query=query,
                papers=papers,
                search_date=datetime.now(),
                total_found=len(papers)
            )
            
        except Exception as e:
            print(f"Error searching PMC: {e}")
            return SearchResult(
                query=query,
                papers=[],
                search_date=datetime.now(),
                total_found=0
            )
    
    async def batch_search(self, queries: List[str], limit_per_query: int = 5) -> List[SearchResult]:
        """Perform batch search for multiple queries."""
        results = []
        
        for i, query in enumerate(queries):
            if self.progress_callback:
                progress = int((i / len(queries)) * 90)
                self.progress_callback(f"Searching for query {i+1}/{len(queries)}", progress)
            
            result = await self.search(query, limit_per_query)
            results.append(result)
            
            # Small delay to be respectful to API
            await asyncio.sleep(0.5)
        
        return results
    
    async def _fetch_paper_details(self, pmids: List[str]) -> List[Paper]:
        """Fetch detailed information for papers."""
        if not pmids:
            return []
        
        # Fetch paper summaries
        summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        summary_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        
        papers = []
        
        try:
            async with self.session.get(summary_url, params=summary_params) as response:
                data = await response.json()
                
            for pmid in pmids:
                if pmid in data.get("result", {}):
                    paper_data = data["result"][pmid]
                    
                    # Parse publication date
                    pub_date = None
                    if "pubdate" in paper_data:
                        try:
                            pub_date = datetime.strptime(paper_data["pubdate"], "%Y %b %d")
                        except:
                            try:
                                pub_date = datetime.strptime(paper_data["pubdate"], "%Y %b")
                            except:
                                pass
                    
                    # Parse authors
                    authors = []
                    if "authors" in paper_data:
                        for author in paper_data["authors"]:
                            if "name" in author:
                                authors.append(author["name"])
                    
                    paper = Paper(
                        pmid=pmid,
                        title=paper_data.get("title", ""),
                        abstract=None,  # Will fetch separately if needed
                        authors=authors,
                        journal=paper_data.get("fulljournalname", ""),
                        publication_date=pub_date,
                        doi=None  # Could extract from paper_data if needed
                    )
                    papers.append(paper)
            
            # Fetch abstracts
            await self._fetch_abstracts(papers)
            
        except Exception as e:
            print(f"Error fetching paper details: {e}")
        
        return papers
    
    async def _fetch_abstracts(self, papers: List[Paper]) -> None:
        """Fetch abstracts for papers."""
        if not papers:
            return
        
        pmids = [paper.pmid for paper in papers]
        
        # Fetch abstracts
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml"
        }
        
        try:
            async with self.session.get(fetch_url, params=fetch_params) as response:
                xml_content = await response.text()
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
            for article in root.findall(".//PubmedArticle"):
                pmid_elem = article.find(".//PMID")
                if pmid_elem is not None:
                    pmid = pmid_elem.text
                    
                    # Find corresponding paper
                    paper = next((p for p in papers if p.pmid == pmid), None)
                    if paper:
                        # Extract abstract
                        abstract_elem = article.find(".//Abstract/AbstractText")
                        if abstract_elem is not None:
                            paper.abstract = abstract_elem.text or ""
                        
                        # Extract DOI if available
                        doi_elem = article.find(".//ArticleId[@IdType='doi']")
                        if doi_elem is not None:
                            paper.doi = doi_elem.text
        
        except Exception as e:
            print(f"Error fetching abstracts: {e}")