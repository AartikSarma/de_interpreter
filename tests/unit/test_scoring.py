"""Unit tests for the scoring module."""

import pytest
from unittest.mock import Mock, patch
import asyncio

from de_interpreter.literature.scoring import LiteratureScorer, ScoringConfig, create_scorer
from de_interpreter.literature.paper import Paper


class TestScoringConfig:
    """Test ScoringConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ScoringConfig()
        
        assert config.scorer_type == "tfidf"
        assert config.biobert_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert config.cache_embeddings is True
        assert config.similarity_threshold == 0.1
        assert config.max_papers_per_query == 20
        assert config.gene_query_mode is False
        assert config.papers_per_gene == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ScoringConfig(
            scorer_type="biobert",
            similarity_threshold=0.2,
            gene_query_mode=True
        )
        
        assert config.scorer_type == "biobert"
        assert config.similarity_threshold == 0.2
        assert config.gene_query_mode is True


class TestLiteratureScorer:
    """Test LiteratureScorer class."""
    
    def test_scorer_initialization(self):
        """Test scorer initialization."""
        config = ScoringConfig()
        scorer = LiteratureScorer(config)
        
        assert scorer.config == config
        assert scorer.model is None
        assert scorer.vectorizer is None
    
    def test_is_available_no_deps(self):
        """Test is_available when dependencies are missing."""
        config = ScoringConfig()
        scorer = LiteratureScorer(config)
        
        # Mock SCORING_AVAILABLE to False
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', False):
            assert scorer.is_available() is False
    
    @pytest.mark.asyncio
    async def test_score_papers_empty_list(self):
        """Test scoring with empty paper list."""
        config = ScoringConfig()
        scorer = LiteratureScorer(config)
        
        result = await scorer.score_papers("test query", [])
        assert result == []
    
    @pytest.mark.asyncio
    async def test_score_papers_no_deps(self):
        """Test scoring when dependencies are not available."""
        config = ScoringConfig()
        scorer = LiteratureScorer(config)
        
        papers = [
            Paper(pmid="1", title="Test", abstract="Test abstract", authors=[], journal="Test Journal")
        ]
        
        with patch.object(scorer, 'is_available', return_value=False):
            result = await scorer.score_papers("test query", papers)
            assert result == papers
    
    def test_get_cache_key(self):
        """Test cache key generation."""
        config = ScoringConfig()
        scorer = LiteratureScorer(config)
        
        key1 = scorer.get_cache_key("test text")
        key2 = scorer.get_cache_key("test text")
        key3 = scorer.get_cache_key("different text")
        
        assert key1 == key2  # Same text should produce same key
        assert key1 != key3  # Different text should produce different key
        assert len(key1) == 32  # MD5 hash length


class TestCreateScorer:
    """Test create_scorer factory function."""
    
    def test_create_scorer_default(self):
        """Test creating scorer with default parameters."""
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', True):
            scorer = create_scorer()
            
            assert isinstance(scorer, LiteratureScorer)
            assert scorer.config.scorer_type == "tfidf"
    
    def test_create_scorer_custom(self):
        """Test creating scorer with custom parameters."""
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', True):
            scorer = create_scorer(scorer_type="biobert", biobert_model="custom-model")
            
            assert isinstance(scorer, LiteratureScorer)
            assert scorer.config.scorer_type == "biobert"
            assert scorer.config.biobert_model == "custom-model"
    
    def test_create_scorer_no_deps(self):
        """Test creating scorer when dependencies are not available."""
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', False):
            scorer = create_scorer()
            assert scorer is None


@pytest.mark.asyncio
class TestScoringMethods:
    """Test different scoring methods."""
    
    @pytest.fixture
    def sample_papers(self):
        """Sample papers for testing."""
        return [
            Paper(
                pmid="1",
                title="Gene expression in cancer",
                abstract="This study examines gene expression patterns in cancer cells",
                authors=["Smith, J."],
                journal="Cancer Research"
            ),
            Paper(
                pmid="2",
                title="Protein function analysis",
                abstract="Analysis of protein function and regulation mechanisms",
                authors=["Jones, A."],
                journal="Protein Science"
            )
        ]
    
    async def test_tfidf_scoring_small_dataset(self, sample_papers):
        """Test TF-IDF scoring with small dataset."""
        config = ScoringConfig(scorer_type="tfidf")
        scorer = LiteratureScorer(config)
        
        # Mock the TF-IDF dependencies
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', True), \
             patch('de_interpreter.literature.scoring.TfidfVectorizer') as mock_vectorizer, \
             patch('de_interpreter.literature.scoring.cosine_similarity') as mock_cosine:
            
            # Setup mocks
            mock_vectorizer_instance = Mock()
            mock_vectorizer.return_value = mock_vectorizer_instance
            mock_vectorizer_instance.fit_transform.return_value = Mock()
            mock_vectorizer_instance.transform.return_value = Mock()
            mock_cosine.return_value = [[0.5, 0.3]]  # Mock similarities
            
            result = await scorer.score_papers("cancer gene expression", sample_papers)
            
            # Verify papers were processed
            assert len(result) == 2
            assert all(hasattr(paper, 'relevance_score') for paper in result if paper.text_content)
    
    async def test_gene_query_similarity_scoring(self, sample_papers):
        """Test gene-query similarity scoring."""
        config = ScoringConfig(scorer_type="gene_query_similarity")
        scorer = LiteratureScorer(config)
        
        with patch('de_interpreter.literature.scoring.SCORING_AVAILABLE', True), \
             patch('de_interpreter.literature.scoring.TfidfVectorizer') as mock_vectorizer, \
             patch('de_interpreter.literature.scoring.cosine_similarity') as mock_cosine:
            
            # Setup mocks
            mock_vectorizer_instance = Mock()
            mock_vectorizer.return_value = mock_vectorizer_instance
            mock_vectorizer_instance.fit_transform.return_value = Mock()
            mock_vectorizer_instance.transform.return_value = Mock()
            mock_cosine.return_value = [[0.6, 0.4]]  # Mock similarities
            
            result = await scorer.score_papers("cancer gene", sample_papers)
            
            # Verify papers were processed
            assert len(result) == 2