"""PMC-based literature mining client using the PMC to PQA2 retriever."""

import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import tempfile
import shutil
from bs4 import BeautifulSoup
import re

from .pmc_retriever import PubMedRetriever


@dataclass
class Paper:
    """Container for paper metadata extracted from PMC."""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    pmc_id: Optional[str] = None
    relevance_score: Optional[float] = None
    full_text: Optional[str] = None
    file_path: Optional[str] = None

    def get_citation(self) -> str:
        """Generate a citation string."""
        author_str = self.authors[0] if self.authors else "Unknown"
        if len(self.authors) > 1:
            author_str += " et al."

        return f"{author_str} ({self.year}). {self.title}. {self.journal or 'Unknown Journal'}"


@dataclass
class SearchResult:
    """Container for search results."""

    query: str
    papers: List[Paper]
    total_results: int
    search_time: datetime
    raw_answer: str  # Store any processed text


class PMCClient:
    """PMC-based literature mining client."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize PMC client with optional output directory."""
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="pmc_papers_")
        self.retriever = PubMedRetriever(output_dir=self.output_dir)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up temporary files if using temp directory
        if "tmp" in self.output_dir and Path(self.output_dir).exists():
            shutil.rmtree(self.output_dir, ignore_errors=True)

    def convert_xml_to_text(self, xml_file_path: str) -> str:
        """Convert PMC XML file to plain text."""
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        soup = BeautifulSoup(xml_content, 'lxml-xml')
        
        text_parts = []
        
        # Title
        title = soup.find('article-title')
        if title:
            text_parts.append(f"Title: {title.get_text()}\n")
        
        # Authors
        authors = soup.find_all('contrib', {'contrib-type': 'author'})
        if authors:
            author_names = []
            for author in authors:
                surname = author.find('surname')
                given_names = author.find('given-names')
                if surname:
                    name = surname.get_text()
                    if given_names:
                        name = f"{given_names.get_text()} {name}"
                    author_names.append(name)
            if author_names:
                text_parts.append(f"Authors: {', '.join(author_names)}\n")
        
        # Abstract
        abstract = soup.find('abstract')
        if abstract:
            abstract_text = ' '.join(p.get_text() for p in abstract.find_all('p'))
            if not abstract_text:
                abstract_text = abstract.get_text()
            text_parts.append(f"\nAbstract:\n{abstract_text}\n")
        
        # Body sections
        body = soup.find('body')
        if body:
            text_parts.append("\nMain Text:\n")
            
            for section in body.find_all('sec'):
                title = section.find('title')
                if title:
                    text_parts.append(f"\n{title.get_text()}\n")
                
                for p in section.find_all('p', recursive=False):
                    text_parts.append(p.get_text() + "\n")
        
        # Join and clean
        full_text = '\n'.join(text_parts)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        full_text = re.sub(r' {2,}', ' ', full_text)
        
        return full_text

    def extract_paper_metadata(self, xml_file_path: str, pmc_id: str) -> Optional[Paper]:
        """Extract paper metadata from XML file."""
        try:
            with open(xml_file_path, 'r', encoding='utf-8') as f:
                xml_content = f.read()
            
            soup = BeautifulSoup(xml_content, 'lxml-xml')
            
            # Extract title
            title_elem = soup.find('article-title')
            title = title_elem.get_text() if title_elem else "Unknown Title"
            
            # Extract authors
            authors = []
            author_elems = soup.find_all('contrib', {'contrib-type': 'author'})
            for author in author_elems:
                surname = author.find('surname')
                given_names = author.find('given-names')
                if surname:
                    name = surname.get_text()
                    if given_names:
                        name = f"{given_names.get_text()} {name}"
                    authors.append(name)
            
            # Extract abstract
            abstract_elem = soup.find('abstract')
            abstract = ""
            if abstract_elem:
                abstract = ' '.join(p.get_text() for p in abstract_elem.find_all('p'))
                if not abstract:
                    abstract = abstract_elem.get_text()
            
            # Extract publication year
            year = datetime.now().year  # Default to current year
            pub_date = soup.find('pub-date', {'pub-type': 'epub'}) or soup.find('pub-date')
            if pub_date:
                year_elem = pub_date.find('year')
                if year_elem:
                    try:
                        year = int(year_elem.get_text())
                    except ValueError:
                        pass
            
            # Extract journal
            journal_elem = soup.find('journal-title')
            journal = journal_elem.get_text() if journal_elem else None
            
            # Extract DOI
            doi_elem = soup.find('article-id', {'pub-id-type': 'doi'})
            doi = doi_elem.get_text() if doi_elem else None
            
            # Extract PMID
            pmid_elem = soup.find('article-id', {'pub-id-type': 'pmid'})
            pmid = pmid_elem.get_text() if pmid_elem else None
            
            # Get full text
            full_text = self.convert_xml_to_text(xml_file_path)
            
            return Paper(
                paper_id=f"pmc_{pmc_id}",
                title=title,
                abstract=abstract,
                authors=authors,
                year=year,
                journal=journal,
                doi=doi,
                pmid=pmid,
                pmc_id=pmc_id,
                relevance_score=1.0,
                full_text=full_text,
                file_path=xml_file_path
            )
            
        except Exception as e:
            print(f"Error extracting metadata from {xml_file_path}: {e}")
            return None

    async def search(self, query: str, limit: int = 10) -> SearchResult:
        """Search for papers using PMC."""
        try:
            # Search PMC
            pmc_ids = self.retriever.search_pubmed(query, max_results=limit)
            
            if not pmc_ids:
                return SearchResult(
                    query=query,
                    papers=[],
                    total_results=0,
                    search_time=datetime.now(),
                    raw_answer=f"No papers found for query: {query}"
                )
            
            papers = []
            downloaded_count = 0
            
            for pmc_id in pmc_ids:
                # Get paper info
                info = self.retriever.get_pmc_info(pmc_id)
                if not info:
                    continue
                
                # Download full text
                file_path = self.retriever.download_fulltext(pmc_id, info['title'])
                if not file_path:
                    continue
                
                # Extract metadata if it's an XML file
                if file_path.endswith('.xml'):
                    paper = self.extract_paper_metadata(file_path, pmc_id)
                    if paper:
                        paper.relevance_score = 1.0 - (downloaded_count * 0.1)
                        papers.append(paper)
                        downloaded_count += 1
                else:
                    # For non-XML files, create basic paper object
                    paper = Paper(
                        paper_id=f"pmc_{pmc_id}",
                        title=info['title'],
                        abstract="",
                        authors=[],
                        year=datetime.now().year,
                        pmc_id=pmc_id,
                        relevance_score=1.0 - (downloaded_count * 0.1),
                        file_path=file_path
                    )
                    papers.append(paper)
                    downloaded_count += 1
            
            return SearchResult(
                query=query,
                papers=papers,
                total_results=len(papers),
                search_time=datetime.now(),
                raw_answer=f"Retrieved {len(papers)} papers from PMC for query: {query}"
            )
            
        except Exception as e:
            return SearchResult(
                query=query,
                papers=[],
                total_results=0,
                search_time=datetime.now(),
                raw_answer=f"Search failed: {e}"
            )

    async def search_gene_disease(
        self,
        gene: str,
        disease: str,
        additional_terms: Optional[List[str]] = None,
        limit: int = 10,
        use_deep_search: bool = False
    ) -> SearchResult:
        """Search for papers about a gene in disease context."""
        # Build query for PMC search
        query_parts = [gene, disease]
        
        if additional_terms:
            query_parts.extend(additional_terms)
        
        # Add common biological terms to improve search
        query_parts.extend(["expression", "function", "mechanism"])
        
        query = " ".join(query_parts)
        
        return await self.search(query, limit=limit)

    async def batch_search(
        self, queries: List[str], limit_per_query: int = 5
    ) -> List[SearchResult]:
        """Perform multiple searches sequentially."""
        results = []
        
        for query in queries:
            try:
                result = await self.search(query, limit=limit_per_query)
                results.append(result)
                
                # Add delay between searches to be respectful to PMC
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                results.append(SearchResult(
                    query=query,
                    papers=[],
                    total_results=0,
                    search_time=datetime.now(),
                    raw_answer=f"Search failed: {e}"
                ))
        
        return results

    def get_paper_text(self, paper: Paper) -> str:
        """Get the full text of a paper."""
        if paper.full_text:
            return paper.full_text
        
        if paper.file_path and Path(paper.file_path).exists():
            if paper.file_path.endswith('.xml'):
                return self.convert_xml_to_text(paper.file_path)
            else:
                # For text files, read directly
                try:
                    with open(paper.file_path, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    print(f"Error reading {paper.file_path}: {e}")
                    return ""
        
        return paper.abstract or ""


async def create_client(output_dir: Optional[str] = None) -> PMCClient:
    """Factory function to create PMC client."""
    return PMCClient(output_dir=output_dir)