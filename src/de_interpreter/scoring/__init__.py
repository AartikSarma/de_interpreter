"""
Scoring module for literature relevance scoring.

This module provides various scoring mechanisms for determining the relevance
of literature to differential expression analysis results.
"""

from .base import BaseScorer
from .biobert.scorer import BioBERTScorer
from .traditional.tfidf import TFIDFScorer
from .traditional.bm25 import BM25Scorer
from .traditional.jaccard import JaccardScorer
from .ensemble.weighted import WeightedEnsemble
from .ensemble.rank_fusion import RankFusion

__all__ = [
    'BaseScorer',
    'BioBERTScorer',
    'TFIDFScorer',
    'BM25Scorer',
    'JaccardScorer',
    'WeightedEnsemble',
    'RankFusion',
]
