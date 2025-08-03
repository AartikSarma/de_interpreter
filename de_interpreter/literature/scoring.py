"""Optional literature scoring functionality for enhanced relevance ranking."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

# Optional dependencies - graceful fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    SCORING_AVAILABLE = True
except ImportError:
    SCORING_AVAILABLE = False

from .paper import Paper


@dataclass
class ScoringConfig:
    """Configuration for literature scoring."""
    scorer_type: str = "tfidf"  # "tfidf", "bm25", "biobert", "gene_query_similarity"
    biobert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_embeddings: bool = True
    cache_dir: str = "cache/embeddings"
    similarity_threshold: float = 0.1  # Minimum similarity to keep papers
    max_papers_per_query: int = 20
    # Gene-query similarity specific options
    gene_query_mode: bool = False  # Enable gene+query pooled approach
    papers_per_gene: int = 10  # Papers to fetch per gene in gene-query mode


class LiteratureScorer:
    """Optional literature scoring for enhanced relevance ranking."""
    
    def __init__(self, config: ScoringConfig):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not SCORING_AVAILABLE:
            logging.warning("Scoring dependencies not available. Install with: pip install sentence-transformers scikit-learn")
    
    def is_available(self) -> bool:
        """Check if scoring functionality is available."""
        return SCORING_AVAILABLE
    
    async def score_papers(
        self, 
        query: str, 
        papers: List[Paper],
        progress_callback: Optional[callable] = None
    ) -> List[Paper]:
        """Score and rank papers by relevance to query."""
        
        if not self.is_available():
            logging.warning("Scoring not available - returning papers unranked")
            return papers
        
        if not papers:
            return papers
        
        if progress_callback:
            progress_callback(f"Scoring {len(papers)} papers with {self.config.scorer_type}", 10)
        
        try:
            if self.config.scorer_type == "biobert":
                scored_papers = await self._score_with_biobert(query, papers, progress_callback)
            elif self.config.scorer_type == "bm25":
                scored_papers = await self._score_with_bm25(query, papers, progress_callback)
            elif self.config.scorer_type == "gene_query_similarity":
                scored_papers = await self._score_with_gene_query_similarity(query, papers, progress_callback)
            else:  # tfidf
                scored_papers = await self._score_with_tfidf(query, papers, progress_callback)
            
            # Filter by threshold and sort by score
            filtered_papers = [
                paper for paper in scored_papers 
                if paper.relevance_score and paper.relevance_score >= self.config.similarity_threshold
            ]
            
            # Sort by relevance score (descending)
            ranked_papers = sorted(
                filtered_papers, 
                key=lambda p: p.relevance_score or 0, 
                reverse=True
            )
            
            # Limit number of papers
            final_papers = ranked_papers[:self.config.max_papers_per_query]
            
            if progress_callback:
                progress_callback(f"Ranked papers: {len(final_papers)}/{len(papers)} above threshold", 100)
            
            return final_papers
            
        except Exception as e:
            logging.error(f"Error scoring papers: {e}")
            if progress_callback:
                progress_callback(f"Scoring failed: {e}", -1)
            return papers  # Return original papers if scoring fails
    
    async def _score_with_biobert(
        self, 
        query: str, 
        papers: List[Paper],
        progress_callback: Optional[callable] = None
    ) -> List[Paper]:
        """Score papers using BioBERT embeddings."""
        
        # Load model if needed
        if self.model is None:
            if progress_callback:
                progress_callback(f"Loading BioBERT model: {self.config.biobert_model}", 20)
            
            try:
                self.model = SentenceTransformer(self.config.biobert_model)
            except Exception as e:
                logging.error(f"Failed to load BioBERT model: {e}")
                if progress_callback:
                    progress_callback("BioBERT loading failed, using TF-IDF", 30)
                return await self._score_with_tfidf(query, papers, progress_callback)
        
        # Prepare texts for encoding
        texts = []
        paper_indices = []
        
        for i, paper in enumerate(papers):
            text = paper.text_content
            if text and text.strip():
                texts.append(text[:2000])  # Truncate for efficiency
                paper_indices.append(i)
        
        if not texts:
            return papers
        
        # Add query to texts
        texts = [query] + texts
        
        if progress_callback:
            progress_callback(f"Encoding {len(texts)} texts with BioBERT", 50)
        
        try:
            # Encode all texts
            embeddings = self.model.encode(texts, show_progress_bar=False)
            
            # Query embedding is first
            query_embedding = embeddings[0:1]
            paper_embeddings = embeddings[1:]
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
            
            # Assign scores to papers
            for i, similarity in enumerate(similarities):
                paper_idx = paper_indices[i]
                papers[paper_idx].relevance_score = float(similarity)
            
            if progress_callback:
                progress_callback("BioBERT scoring complete", 90)
            
            return papers
            
        except Exception as e:
            logging.error(f"BioBERT encoding error: {e}")
            if progress_callback:
                progress_callback("BioBERT failed, using TF-IDF", 60)
            return await self._score_with_tfidf(query, papers, progress_callback)
    
    async def _score_with_tfidf(
        self, 
        query: str, 
        papers: List[Paper],
        progress_callback: Optional[callable] = None
    ) -> List[Paper]:
        """Score papers using TF-IDF similarity."""
        
        if progress_callback:
            progress_callback("Computing TF-IDF scores", 30)
        
        # Prepare texts
        texts = []
        paper_indices = []
        
        for i, paper in enumerate(papers):
            text = paper.text_content
            if text and text.strip():
                texts.append(text)
                paper_indices.append(i)
        
        if not texts:
            return papers
        
        try:
            # Create TF-IDF vectorizer
            # Adjust parameters for small datasets
            max_df = min(0.8, max(2, len(texts) * 0.8)) if len(texts) > 2 else len(texts)
            
            vectorizer = TfidfVectorizer(
                max_features=min(1000, len(texts) * 50),
                stop_words='english',
                ngram_range=(1, min(2, len(texts))),
                max_df=max_df,
                min_df=1
            )
            
            # Fit vectorizer on paper texts
            paper_vectors = vectorizer.fit_transform(texts)
            
            # Transform query
            query_vector = vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, paper_vectors)[0]
            
            # Assign scores to papers
            for i, similarity in enumerate(similarities):
                paper_idx = paper_indices[i]
                papers[paper_idx].relevance_score = float(similarity)
            
            if progress_callback:
                progress_callback("TF-IDF scoring complete", 90)
            
            return papers
            
        except Exception as e:
            logging.error(f"TF-IDF scoring error: {e}")
            # Assign default scores
            for paper in papers:
                paper.relevance_score = 0.5
            return papers
    
    async def _score_with_bm25(
        self, 
        query: str, 
        papers: List[Paper],
        progress_callback: Optional[callable] = None
    ) -> List[Paper]:
        """Score papers using BM25 similarity (approximated with TF-IDF)."""
        
        if progress_callback:
            progress_callback("Computing BM25 scores", 30)
        
        # For now, use TF-IDF as BM25 approximation
        # Could implement proper BM25 with rank-bm25 library if needed
        papers = await self._score_with_tfidf(query, papers, progress_callback)
        
        # Adjust scores slightly for BM25-like behavior (boost shorter documents)
        for paper in papers:
            if paper.relevance_score is not None and paper.text_content:
                doc_length = len(paper.text_content.split())
                # Slight boost for shorter, more focused documents
                length_penalty = min(1.0, 500 / max(doc_length, 100))
                paper.relevance_score = paper.relevance_score * (0.8 + 0.2 * length_penalty)
        
        return papers
    
    async def _score_with_gene_query_similarity(
        self, 
        query: str, 
        papers: List[Paper],
        progress_callback: Optional[callable] = None
    ) -> List[Paper]:
        """Score papers using gene-query similarity approach (enhanced for gene sets)."""
        
        if progress_callback:
            progress_callback("Computing gene-query similarity scores", 30)
        
        # This is a placeholder implementation - in a real gene-query similarity scorer,
        # we would need additional gene information. For now, use enhanced TF-IDF
        # that considers gene mentions in the text more heavily.
        
        # Prepare texts with gene-aware weighting
        texts = []
        paper_indices = []
        
        for i, paper in enumerate(papers):
            text = paper.text_content
            if text and text.strip():
                # Weight text by potential gene mentions (simple heuristic)
                # In practice, this would use actual gene lists from the analysis
                weighted_text = text
                
                # Simple gene mention detection (this could be enhanced)
                gene_patterns = ['gene', 'protein', 'expression', 'regulation']
                gene_score = sum(1 for pattern in gene_patterns if pattern.lower() in text.lower())
                
                # Repeat gene-relevant portions to increase their weight
                if gene_score > 0:
                    gene_context = ' '.join([sent for sent in text.split('.') 
                                           if any(pattern in sent.lower() for pattern in gene_patterns)])
                    weighted_text = text + ' ' + gene_context
                
                texts.append(weighted_text)
                paper_indices.append(i)
        
        if not texts:
            return papers
        
        try:
            # Create TF-IDF vectorizer with gene-aware parameters
            # Adjust parameters based on number of documents
            max_df = min(0.8, max(2, len(texts) * 0.8)) if len(texts) > 2 else len(texts)
            
            vectorizer = TfidfVectorizer(
                max_features=min(1500, len(texts) * 100),  # Adjust for small datasets
                stop_words='english',
                ngram_range=(1, min(3, len(texts))),  # Adjust n-grams for small datasets
                max_df=max_df,
                min_df=1
            )
            
            # Fit vectorizer on paper texts
            paper_vectors = vectorizer.fit_transform(texts)
            
            # Create enhanced query with gene context hints
            enhanced_query = f"{query} gene expression regulation protein"
            query_vector = vectorizer.transform([enhanced_query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, paper_vectors)[0]
            
            # Apply gene-specific boost (papers with more gene mentions get higher scores)
            for i, similarity in enumerate(similarities):
                paper_idx = paper_indices[i]
                original_text = papers[paper_idx].text_content or ""
                
                # Boost based on gene-related content
                gene_mentions = sum(1 for pattern in ['gene', 'protein', 'expression', 'regulation']
                                  if pattern.lower() in original_text.lower())
                gene_boost = 1.0 + (gene_mentions * 0.1)  # 10% boost per gene term
                
                boosted_similarity = similarity * gene_boost
                papers[paper_idx].relevance_score = float(boosted_similarity)
            
            if progress_callback:
                progress_callback("Gene-query similarity scoring complete", 90)
            
            return papers
            
        except Exception as e:
            logging.error(f"Gene-query similarity scoring error: {e}")
            # Fallback to basic TF-IDF
            return await self._score_with_tfidf(query, papers, progress_callback)
    
    def get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache an embedding to disk."""
        if not self.config.cache_embeddings:
            return
        
        try:
            cache_key = self.get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            np.save(cache_file, embedding)
        except Exception as e:
            logging.warning(f"Failed to cache embedding: {e}")
    
    def load_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Load cached embedding from disk."""
        if not self.config.cache_embeddings:
            return None
        
        try:
            cache_key = self.get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                return np.load(cache_file)
        except Exception as e:
            logging.warning(f"Failed to load cached embedding: {e}")
        
        return None


def create_scorer(
    scorer_type: str = "tfidf", 
    biobert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Optional[LiteratureScorer]:
    """Factory function to create a literature scorer."""
    
    if not SCORING_AVAILABLE:
        logging.warning("Scoring dependencies not available")
        return None
    
    config = ScoringConfig(
        scorer_type=scorer_type,
        biobert_model=biobert_model
    )
    
    return LiteratureScorer(config)


# Utility function for testing
async def test_scoring():
    """Test scoring functionality with sample data."""
    if not SCORING_AVAILABLE:
        print("❌ Scoring dependencies not available")
        return
    
    # Create test papers
    test_papers = [
        Paper(
            pmid="1",
            title="Gene expression in cancer",
            abstract="This study examines differential gene expression in cancer cells",
            authors=["Smith, J."],
            journal="Cancer Research",
            publication_date=None
        ),
        Paper(
            pmid="2", 
            title="Metabolic pathways in diabetes",
            abstract="Analysis of metabolic changes in diabetic patients",
            authors=["Jones, A."],
            journal="Diabetes Journal",
            publication_date=None
        )
    ]
    
    # Test different scoring methods
    for scorer_type in ["tfidf", "bm25", "biobert"]:
        print(f"\n--- Testing {scorer_type} ---")
        
        scorer = create_scorer(scorer_type)
        if scorer:
            query = "cancer gene expression"
            scored_papers = await scorer.score_papers(query, test_papers.copy())
            
            for paper in scored_papers:
                print(f"PMID {paper.pmid}: {paper.relevance_score:.3f} - {paper.title}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_scoring())