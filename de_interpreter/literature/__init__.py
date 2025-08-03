"""Literature mining modules."""

from .paper import Paper
from .cache import SearchResult, LiteratureCache
from .pmc_client import PMCClient

# Optional scoring functionality
try:
    from .scoring import LiteratureScorer, ScoringConfig, create_scorer
    scoring_available = True
except ImportError:
    scoring_available = False

# Optional MeSH enhancement functionality
try:
    from .mesh_enhancer import MeshQueryEnhancer, MeshEnhancedQuery, create_mesh_enhancer
    mesh_available = True
except ImportError:
    mesh_available = False

# Build __all__ dynamically
__all__ = ["Paper", "SearchResult", "LiteratureCache", "PMCClient"]

if scoring_available:
    __all__.extend(["LiteratureScorer", "ScoringConfig", "create_scorer"])

if mesh_available:
    __all__.extend(["MeshQueryEnhancer", "MeshEnhancedQuery", "create_mesh_enhancer"])