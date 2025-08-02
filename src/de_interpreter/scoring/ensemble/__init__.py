"""
Ensemble methods for combining multiple scorers.
"""

from .weighted import WeightedEnsemble
from .rank_fusion import RankFusion

__all__ = [
    'WeightedEnsemble',
    'RankFusion',
]
