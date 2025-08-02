"""
Rank fusion methods for combining multiple scorer rankings.
"""

from typing import List, Dict, Any, Tuple, Optional
import math

from ..base import BaseScorer


class RankFusion(BaseScorer):
    """
    Rank fusion scorer that combines rankings from multiple scorers.
    
    This scorer uses rank-based fusion methods to combine the rankings
    from multiple base scorers into a single ranking.
    """
    
    def __init__(self, 
                 scorers: List[BaseScorer],
                 method: str = "rrf",
                 k: float = 60.0):
        """
        Initialize rank fusion scorer.
        
        Args:
            scorers: List of base scorers to combine
            method: Fusion method ('rrf' for Reciprocal Rank Fusion)
            k: Parameter for RRF (typically 60)
        """
        super().__init__(name="RankFusion")
        
        if not scorers:
            raise ValueError("At least one scorer must be provided")
            
        self.scorers = scorers
        self.method = method.lower()
        self.k = k
        
        if self.method not in ['rrf', 'borda']:
            raise ValueError(f"Unsupported fusion method: {method}")
            
        # Update name to include component scorers
        scorer_names = [scorer.name for scorer in self.scorers]
        self.name = f"RankFusion-{method.upper()}({', '.join(scorer_names)})"
    
    def _get_rankings(self, 
                     query: str, 
                     documents: List[Dict[str, Any]]) -> List[List[Tuple[int, float]]]:
        """
        Get rankings from all scorers.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            List of rankings, one per scorer
        """
        all_rankings = []
        
        for scorer in self.scorers:
            try:
                ranking = scorer.rank_documents(query, documents)
                all_rankings.append(ranking)
            except Exception as e:
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                # Create empty ranking
                all_rankings.append([])
                
        return all_rankings
    
    def _reciprocal_rank_fusion(self, 
                               rankings: List[List[Tuple[int, float]]]) -> List[Tuple[int, float]]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).
        
        Args:
            rankings: List of rankings from different scorers
            
        Returns:
            Combined ranking
        """
        # Collect all document indices
        all_doc_indices = set()
        for ranking in rankings:
            for doc_idx, _ in ranking:
                all_doc_indices.add(doc_idx)
        
        # Compute RRF scores
        rrf_scores = {}
        
        for doc_idx in all_doc_indices:
            rrf_score = 0.0
            
            for ranking in rankings:
                # Find position of document in this ranking
                position = None
                for rank, (d_idx, _) in enumerate(ranking):
                    if d_idx == doc_idx:
                        position = rank + 1  # 1-based ranking
                        break
                
                if position is not None:
                    rrf_score += 1.0 / (self.k + position)
            
            rrf_scores[doc_idx] = rrf_score
        
        # Sort by RRF score (descending)
        sorted_docs = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(doc_idx, score) for doc_idx, score in sorted_docs]
    
    def _borda_count_fusion(self, 
                           rankings: List[List[Tuple[int, float]]], 
                           num_documents: int) -> List[Tuple[int, float]]:
        """
        Combine rankings using Borda count.
        
        Args:
            rankings: List of rankings from different scorers
            num_documents: Total number of documents
            
        Returns:
            Combined ranking
        """
        # Collect all document indices
        all_doc_indices = set()
        for ranking in rankings:
            for doc_idx, _ in ranking:
                all_doc_indices.add(doc_idx)
        
        # Compute Borda scores
        borda_scores = {}
        
        for doc_idx in all_doc_indices:
            borda_score = 0.0
            
            for ranking in rankings:
                # Find position of document in this ranking
                position = None
                for rank, (d_idx, _) in enumerate(ranking):
                    if d_idx == doc_idx:
                        position = rank
                        break
                
                if position is not None:
                    # Borda count: higher rank = higher score
                    borda_score += (num_documents - position - 1)
                # If document not found in ranking, it gets 0 points
            
            borda_scores[doc_idx] = borda_score
        
        # Sort by Borda score (descending)
        sorted_docs = sorted(
            borda_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [(doc_idx, score) for doc_idx, score in sorted_docs]
    
    def rank_documents(self, 
                      query: str, 
                      documents: List[Dict[str, Any]]) -> List[Tuple[int, float]]:
        """
        Rank documents using rank fusion.
        
        Args:
            query: Search query
            documents: List of documents to rank
            
        Returns:
            Combined ranking
        """
        if not documents:
            return []
            
        # Get rankings from all scorers
        rankings = self._get_rankings(query, documents)
        
        # Apply fusion method
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(rankings)
        elif self.method == "borda":
            return self._borda_count_fusion(rankings, len(documents))
        else:
            raise ValueError(f"Unsupported fusion method: {self.method}")
    
    def score_documents(self, 
                       query: str, 
                       documents: List[Dict[str, Any]]) -> List[float]:
        """
        Score documents using rank fusion.
        
        Args:
            query: Search query
            documents: List of documents to score
            
        Returns:
            List of fusion scores
        """
        if not documents:
            return []
            
        # Get combined ranking
        ranking = self.rank_documents(query, documents)
        
        # Convert ranking to scores
        scores = [0.0] * len(documents)
        for doc_idx, score in ranking:
            if 0 <= doc_idx < len(documents):
                scores[doc_idx] = score
                
        return scores
    
    def score_single_document(self, 
                             query: str, 
                             document: Dict[str, Any]) -> float:
        """
        Score a single document using rank fusion.
        
        Args:
            query: Search query
            document: Document to score
            
        Returns:
            Fusion score
        """
        # For single document, just average the scores from all scorers
        scores = []
        
        for scorer in self.scorers:
            try:
                score = scorer.score_single_document(query, document)
                scores.append(score)
            except Exception as e:
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def get_individual_rankings(self, 
                               query: str, 
                               documents: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get individual rankings from each scorer.
        
        Args:
            query: Search query
            documents: List of documents
            
        Returns:
            Dictionary mapping scorer names to their rankings
        """
        individual_rankings = {}
        
        for scorer in self.scorers:
            try:
                ranking = scorer.rank_documents(query, documents)
                individual_rankings[scorer.name] = ranking
            except Exception as e:
                print(f"Warning: Scorer {scorer.name} failed with error: {e}")
                individual_rankings[scorer.name] = []
                
        return individual_rankings
    
    def analyze_ranking_agreement(self, 
                                 query: str, 
                                 documents: List[Dict[str, Any]],
                                 top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze agreement between different scorer rankings.
        
        Args:
            query: Search query
            documents: List of documents
            top_k: Number of top documents to analyze
            
        Returns:
            Dictionary with agreement statistics
        """
        rankings = self._get_rankings(query, documents)
        
        if not rankings:
            return {}
        
        # Get top-k documents from each ranking
        top_k_sets = []
        for ranking in rankings:
            top_docs = set(doc_idx for doc_idx, _ in ranking[:top_k])
            top_k_sets.append(top_docs)
        
        # Compute intersection and union
        if top_k_sets:
            intersection = set.intersection(*top_k_sets) if len(top_k_sets) > 1 else top_k_sets[0]
            union = set.union(*top_k_sets)
            
            jaccard_similarity = len(intersection) / len(union) if union else 0.0
            overlap_ratio = len(intersection) / top_k if top_k > 0 else 0.0
        else:
            jaccard_similarity = 0.0
            overlap_ratio = 0.0
        
        return {
            'num_scorers': len(rankings),
            'top_k': top_k,
            'intersection_size': len(intersection) if top_k_sets else 0,
            'union_size': len(union) if top_k_sets else 0,
            'jaccard_similarity': jaccard_similarity,
            'overlap_ratio': overlap_ratio,
            'individual_ranking_sizes': [len(ranking) for ranking in rankings]
        }
