"""Caching for literature search results."""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import hashlib

from .futurehouse_client import SearchResult, Paper


class LiteratureCache:
    """Simple file-based cache for literature search results."""

    def __init__(self, cache_dir: Path = Path("cache/literature"), ttl_days: int = 7):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(days=ttl_days)

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"

    def get(self, query: str) -> Optional[SearchResult]:
        """Retrieve cached search result."""
        cache_key = self._get_cache_key(query)
        cache_path = self._get_cache_path(cache_key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                data = json.load(f)

            # Check if cache is expired
            cached_time = datetime.fromisoformat(data["search_time"])
            if datetime.now() - cached_time > self.ttl:
                cache_path.unlink()  # Delete expired cache
                return None

            # Reconstruct SearchResult
            papers = [
                Paper(
                    paper_id=p["paper_id"],
                    title=p["title"],
                    abstract=p["abstract"],
                    authors=p["authors"],
                    year=p["year"],
                    journal=p.get("journal"),
                    doi=p.get("doi"),
                    pmid=p.get("pmid"),
                    relevance_score=p.get("relevance_score"),
                )
                for p in data["papers"]
            ]

            return SearchResult(
                query=data["query"],
                papers=papers,
                total_results=data["total_results"],
                search_time=cached_time,
                raw_answer=data.get("raw_answer", "")
            )

        except Exception as e:
            print(f"Error reading cache: {e}")
            return None

    def set(self, result: SearchResult) -> None:
        """Cache search result."""
        cache_key = self._get_cache_key(result.query)
        cache_path = self._get_cache_path(cache_key)

        # Convert to serializable format
        data = {
            "query": result.query,
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract,
                    "authors": p.authors,
                    "year": p.year,
                    "journal": p.journal,
                    "doi": p.doi,
                    "pmid": p.pmid,
                    "relevance_score": p.relevance_score,
                }
                for p in result.papers
            ],
            "total_results": result.total_results,
            "search_time": result.search_time.isoformat(),
            "raw_answer": result.raw_answer
        }

        try:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error writing cache: {e}")

    def clear(self) -> None:
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        cache_files = list(self.cache_dir.glob("*.json"))
        valid_count = 0
        expired_count = 0

        for cache_file in cache_files:
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                cached_time = datetime.fromisoformat(data["search_time"])
                if datetime.now() - cached_time <= self.ttl:
                    valid_count += 1
                else:
                    expired_count += 1
            except:
                pass

        return {
            "total_entries": len(cache_files),
            "valid_entries": valid_count,
            "expired_entries": expired_count,
        }
