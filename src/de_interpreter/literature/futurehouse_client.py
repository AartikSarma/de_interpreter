"""FutureHouse Paper Search API client."""

import os
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio
from pathlib import Path
import json

# Use the official FutureHouse client
from futurehouse_client import FutureHouseClient as OfficialFutureHouseClient
from futurehouse_client import JobNames as OfficialJobNames


@dataclass
class Paper:
    """Container for paper metadata extracted from FutureHouse responses."""

    paper_id: str
    title: str
    abstract: str
    authors: List[str]
    year: int
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    relevance_score: Optional[float] = None

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
    raw_answer: str  # Store the full FutureHouse response


class JobNames:
    """FutureHouse job types."""
    CROW = OfficialJobNames.CROW  # Fast Search
    FALCON = OfficialJobNames.FALCON  # Deep Search


class FutureHouseClient:
    """Wrapper around the official FutureHouse client for paper search."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FUTUREHOUSE_API_KEY")
        if not self.api_key:
            raise ValueError("FutureHouse API key not provided")

        self.client = OfficialFutureHouseClient(api_key=self.api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Official client handles cleanup automatically
        pass

    async def search(
        self, query: str, job_type: str = None, limit: int = 20
    ) -> SearchResult:
        """Search for papers using FutureHouse API."""
        job_type = job_type or JobNames.CROW  # Default to fast search
        
        try:
            # Use the official client to run the task
            task_response = self.client.run_tasks_until_done([{
                "name": job_type,
                "query": query,
            }])
            
            # Extract the response
            if not task_response or not task_response[0]:
                raise Exception("No response from FutureHouse")
            
            response = task_response[0]
            
            # Get the answer
            raw_answer = getattr(response, 'answer', '')
            formatted_answer = getattr(response, 'formatted_answer', raw_answer)
            
            if not raw_answer:
                raise Exception("Empty response from FutureHouse")
            
            # Parse citations and create Paper objects from the response
            papers = self._extract_papers_from_response(formatted_answer, query)

            return SearchResult(
                query=query,
                papers=papers,
                total_results=len(papers),
                search_time=datetime.now(),
                raw_answer=raw_answer
            )

        except Exception as e:
            raise Exception(f"Search failed: {e}")

    async def search_gene_disease(
        self,
        gene: str,
        disease: str,
        additional_terms: Optional[List[str]] = None,
        limit: int = 20,
        use_deep_search: bool = False
    ) -> SearchResult:
        """Search for papers about a gene in disease context."""
        # Build a natural language query for FutureHouse
        query_parts = [
            f"What is the role of {gene} in {disease}?",
            f"How is {gene} expression regulated in {disease}?",
            f"What are the mechanisms of {gene} in {disease} pathogenesis?"
        ]
        
        if additional_terms:
            query_parts.append(f"Consider {', '.join(additional_terms)} in the context.")
        
        # Choose one query (FutureHouse works better with focused questions)
        query = query_parts[0]
        
        # Use deep search for more comprehensive results if requested
        job_type = JobNames.FALCON if use_deep_search else JobNames.CROW
        
        return await self.search(query, job_type=job_type, limit=limit)

    async def batch_search(
        self, queries: List[str], limit_per_query: int = 10
    ) -> List[SearchResult]:
        """Perform multiple searches using FutureHouse batch processing."""
        try:
            # Prepare batch tasks
            tasks = []
            for query in queries:
                tasks.append({
                    "name": JobNames.CROW,  # Use fast search for batch
                    "query": query,
                })
            
            # Run batch tasks
            task_responses = self.client.run_tasks_until_done(tasks)
            
            # Process responses
            results = []
            for i, response in enumerate(task_responses):
                query = queries[i] if i < len(queries) else f"Query {i}"
                
                try:
                    if response and hasattr(response, 'answer'):
                        raw_answer = response.answer
                        formatted_answer = getattr(response, 'formatted_answer', raw_answer)
                        papers = self._extract_papers_from_response(formatted_answer, query)
                        
                        results.append(SearchResult(
                            query=query,
                            papers=papers,
                            total_results=len(papers),
                            search_time=datetime.now(),
                            raw_answer=raw_answer
                        ))
                    else:
                        # Empty response
                        results.append(SearchResult(
                            query=query,
                            papers=[],
                            total_results=0,
                            search_time=datetime.now(),
                            raw_answer="No response from FutureHouse"
                        ))
                        
                except Exception as e:
                    print(f"Error processing response for '{query}': {e}")
                    results.append(SearchResult(
                        query=query,
                        papers=[],
                        total_results=0,
                        search_time=datetime.now(),
                        raw_answer=f"Processing failed: {e}"
                    ))
            
            return results
            
        except Exception as e:
            print(f"Batch search failed: {e}")
            # Return empty results for all queries
            return [
                SearchResult(
                    query=query,
                    papers=[],
                    total_results=0,
                    search_time=datetime.now(),
                    raw_answer=f"Batch search failed: {e}"
                )
                for query in queries
            ]

    def _extract_papers_from_response(self, formatted_answer: str, query: str) -> List[Paper]:
        """Extract paper information from FutureHouse formatted response."""
        papers = []
        
        # FutureHouse responses include citations in various formats
        # This is a simplified parser - in practice, would need more sophisticated parsing
        import re
        
        # Look for citation patterns in the response
        citation_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+et\s+al\.?\s*\((\d{4})\)\.?\s*([^.]+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+\((\d{4})\)\.?\s*([^.]+)',
            r'\[(\d+)\]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)\s+et\s+al\.?\s*\((\d{4})\)\.?\s*([^.]+)',
        ]
        
        paper_count = 0
        for pattern in citation_patterns:
            matches = re.finditer(pattern, formatted_answer)
            
            for match in matches:
                if paper_count >= 10:  # Limit to reasonable number of papers
                    break
                    
                try:
                    if len(match.groups()) == 3:  # author, year, title format
                        authors = [match.group(1)]
                        year = int(match.group(2))
                        title = match.group(3).strip()
                    elif len(match.groups()) == 4:  # numbered citation format
                        authors = [match.group(2)]
                        year = int(match.group(3))
                        title = match.group(4).strip()
                    else:
                        continue
                    
                    # Clean up title
                    title = title.rstrip('.,;:')
                    if len(title) > 200:  # Truncate very long titles
                        title = title[:200] + "..."
                    
                    paper = Paper(
                        paper_id=f"fh_{paper_count}_{hash(title) % 10000}",
                        title=title,
                        abstract="",  # FutureHouse doesn't provide abstracts directly
                        authors=authors,
                        year=year,
                        relevance_score=1.0 - (paper_count * 0.1)  # Decreasing relevance
                    )
                    papers.append(paper)
                    paper_count += 1
                    
                except (ValueError, IndexError) as e:
                    continue
        
        # If no citations found, create a synthetic paper from the response
        if not papers and formatted_answer:
            papers.append(Paper(
                paper_id=f"fh_summary_{hash(query) % 10000}",
                title=f"Literature Summary: {query[:50]}...",
                abstract=formatted_answer[:500] + "..." if len(formatted_answer) > 500 else formatted_answer,
                authors=["FutureHouse AI"],
                year=datetime.now().year,
                relevance_score=1.0
            ))
        
        return papers


async def create_client() -> FutureHouseClient:
    """Factory function to create FutureHouse client."""
    return FutureHouseClient()
