"""Literature mining modules."""

from .paper import Paper
from .cache import SearchResult, LiteratureCache
from .pmc_client import PMCClient

# Optional scoring functionality
try:
    from .scoring import LiteratureScorer, ScoringConfig, create_scorer
    __all__ = ["Paper", "SearchResult", "LiteratureCache", "PMCClient", "LiteratureScorer", "ScoringConfig", "create_scorer"]
except ImportError:
    __all__ = ["Paper", "SearchResult", "LiteratureCache", "PMCClient"]