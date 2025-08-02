"""
Weighted ensemble scorer that combines multiple scorers.
"""

from typing import List, Dict, Any, Optional
import numpy as np

from ..base import BaseScorer


class WeightedEnsemble(BaseScorer):
    """
    Weighted ensemble that combines scores from multiple scorers.
    
    This scorer combines the outputs of multiple base scorers using
    weighted averaging to produce final relevance scores.
    """
    
    def __init__(self, 
                 scorers: List[BaseScorer],
                 weights: Optional[List[float]] = None,
                 normalize_scores: bool = True):
        """
        Initialize weighted ensemble.
        
        Args:
            scorers: List of base scorers to combine
            weights: List of weights for each scorer (if None, equal weights)
            normalize_scores: Whether to normalize scores to [0,1] before combining
        """
        super().__init__(name="WeightedEnsemble")
        
        if not scorers:
            raise ValueError("At least one scorer must be provided")
            
        self.scorers = scorers
        self.normalize_scores = normalize_scores
        
        # Set weights
        if weights is None:
            self.weights = [1.0 / len(scorers)] * len(scorers)
        else:
            if len(weights) != len(scorers):
                raise ValueError("Number of weights must match number of scorers")
            # Normalize weights to sum to 1
            weight_sum = sum(weights)
            self.weights = [w / weight_sum for w in weights]
            
        # Update name to include component scorers
        scorer_names = [scorer.name for scorer in self.scorers]
        self.name = f"WeightedEnsemble({', '.join(scorer_names)})"
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of scores to normalize
            
        Returns:
            Normalized scores
        """
        if not scores:
            return scores
            
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            # All scores are the same
            return [0.5] * len(scores)
            
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document using weighted ensemble.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Weighted ensemble score
        """
        scores = []
        
        # Get scores from each scorer
        for scorer in self.scorers:
            try:
                score = scorer.score_single_document(query, document)
                scores.append(score)
            except Exception as e:
                # If a scorer fails, assign score of 0
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                scores.append(0.0)
        
        # Normalize if requested
        if self.normalize_scores:
            # For single document, we can't normalize across documents
            # So we'll just use the raw scores
            pass
            
        # Compute weighted average
        weighted_score = sum(
            score * weight 
            for score, weight in zip(scores, self.weights)
        )
        
        return weighted_score
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score multiple documents using weighted ensemble.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of weighted ensemble scores
        """
        if not documents:
            return []
            
        # Get scores from each scorer
        all_scorer_scores = []
        
        for scorer in self.scorers:
            try:
                scores = scorer.score_documents(query, documents)
                all_scorer_scores.append(scores)
            except Exception as e:
                # If a scorer fails, assign scores of 0
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                all_scorer_scores.append([0.0] * len(documents))
        
        # Normalize scores if requested
        if self.normalize_scores:
            normalized_scores = []
            for scores in all_scorer_scores:
                normalized_scores.append(self._normalize_scores(scores))
            all_scorer_scores = normalized_scores
        
        # Compute weighted averages
        ensemble_scores = []
        for doc_idx in range(len(documents)):
            doc_scores = [scorer_scores[doc_idx] for scorer_scores in all_scorer_scores]
            weighted_score = sum(
                score * weight 
                for score, weight in zip(doc_scores, self.weights)
            )
            ensemble_scores.append(weighted_score)
            
        return ensemble_scores
    
    def get_scorer_contributions(self, 
                               query: str, 
                               documents: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """
        Get individual scorer contributions to the ensemble.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            Dictionary mapping scorer names to their scores
        """
        contributions = {}
        
        for scorer in self.scorers:
            try:
                scores = scorer.score_documents(query, documents)
                if self.normalize_scores:
                    scores = self._normalize_scores(scores)
                contributions[scorer.name] = scores
            except Exception as e:
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                contributions[scorer.name] = [0.0] * len(documents)
                
        return contributions
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get the weights assigned to each scorer.
        
        Returns:
            Dictionary mapping scorer names to weights
        """
        return {
            scorer.name: weight 
            for scorer, weight in zip(self.scorers, self.weights)
        }
    
    def update_weights(self, new_weights: List[float]):
        """
        Update the weights for the ensemble.
        
        Args:
            new_weights: New weights for each scorer
        """
        if len(new_weights) != len(self.scorers):
            raise ValueError("Number of weights must match number of scorers")
            
        # Normalize weights to sum to 1
        weight_sum = sum(new_weights)
        if weight_sum == 0:
            raise ValueError("Sum of weights cannot be zero")
            
        self.weights = [w / weight_sum for w in new_weights]
    
    def add_scorer(self, scorer: BaseScorer, weight: float = 1.0):
        """
        Add a new scorer to the ensemble.
        
        Args:
            scorer: New scorer to add
            weight: Weight for the new scorer
        """
        self.scorers.append(scorer)
        
        # Recalculate weights
        current_total_weight = sum(self.weights)
        new_total_weight = current_total_weight + weight
        
        # Normalize all weights
        self.weights = [w * current_total_weight / new_total_weight for w in self.weights]
        self.weights.append(weight / new_total_weight)
        
        # Update name
        scorer_names = [scorer.name for scorer in self.scorers]
        self.name = f"WeightedEnsemble({', '.join(scorer_names)})"
    
    def remove_scorer(self, scorer_name: str):
        """
        Remove a scorer from the ensemble.
        
        Args:
            scorer_name: Name of the scorer to remove
        """
        # Find scorer index
        scorer_idx = None
        for i, scorer in enumerate(self.scorers):
            if scorer.name == scorer_name:
                scorer_idx = i
                break
                
        if scorer_idx is None:
            raise ValueError(f"Scorer '{scorer_name}' not found in ensemble")
            
        if len(self.scorers) <= 1:
            raise ValueError("Cannot remove scorer - ensemble must have at least one scorer")
            
        # Remove scorer and weight
        self.scorers.pop(scorer_idx)
        removed_weight = self.weights.pop(scorer_idx)
        
        # Renormalize remaining weights
        remaining_weight = sum(self.weights)
        if remaining_weight > 0:
            self.weights = [w / remaining_weight for w in self.weights]
        
        # Update name
        scorer_names = [scorer.name for scorer in self.scorers]
        self.name = f"WeightedEnsemble({', '.join(scorer_names)})"
