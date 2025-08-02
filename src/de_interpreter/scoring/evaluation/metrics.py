"""
Evaluation metrics for scoring methods.
"""

from typing import List, Dict, Any, Tuple, Optional
import math
from collections import defaultdict


class ScoringMetrics:
    """
    Metrics for evaluating scoring method performance.
    
    Provides various metrics to assess how well a scoring method
    ranks relevant documents.
    """
    
    @staticmethod
    def precision_at_k(relevant_docs: List[int], 
                      ranked_docs: List[int], 
                      k: int) -> float:
        """
        Compute precision at k.
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            k: Cutoff rank
            
        Returns:
            Precision at k
        """
        if k <= 0 or not ranked_docs:
            return 0.0
            
        top_k = ranked_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))
        
        return relevant_in_top_k / min(k, len(top_k))
    
    @staticmethod
    def recall_at_k(relevant_docs: List[int], 
                   ranked_docs: List[int], 
                   k: int) -> float:
        """
        Compute recall at k.
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            k: Cutoff rank
            
        Returns:
            Recall at k
        """
        if not relevant_docs or k <= 0:
            return 0.0
            
        top_k = ranked_docs[:k]
        relevant_in_top_k = len(set(top_k) & set(relevant_docs))
        
        return relevant_in_top_k / len(relevant_docs)
    
    @staticmethod
    def f1_at_k(relevant_docs: List[int], 
               ranked_docs: List[int], 
               k: int) -> float:
        """
        Compute F1 score at k.
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            k: Cutoff rank
            
        Returns:
            F1 score at k
        """
        precision = ScoringMetrics.precision_at_k(relevant_docs, ranked_docs, k)
        recall = ScoringMetrics.recall_at_k(relevant_docs, ranked_docs, k)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def average_precision(relevant_docs: List[int], 
                         ranked_docs: List[int]) -> float:
        """
        Compute average precision (AP).
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            
        Returns:
            Average precision
        """
        if not relevant_docs or not ranked_docs:
            return 0.0
            
        relevant_set = set(relevant_docs)
        precision_sum = 0.0
        relevant_count = 0
        
        for i, doc_id in enumerate(ranked_docs):
            if doc_id in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
                
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    @staticmethod
    def mean_average_precision(queries_results: List[Tuple[List[int], List[int]]]) -> float:
        """
        Compute mean average precision (MAP) across multiple queries.
        
        Args:
            queries_results: List of (relevant_docs, ranked_docs) tuples
            
        Returns:
            Mean average precision
        """
        if not queries_results:
            return 0.0
            
        ap_scores = [
            ScoringMetrics.average_precision(relevant, ranked)
            for relevant, ranked in queries_results
        ]
        
        return sum(ap_scores) / len(ap_scores)
    
    @staticmethod
    def dcg_at_k(relevant_docs: List[int], 
                ranked_docs: List[int], 
                k: int,
                relevance_scores: Optional[Dict[int, float]] = None) -> float:
        """
        Compute Discounted Cumulative Gain at k.
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            k: Cutoff rank
            relevance_scores: Optional dict mapping doc_id to relevance score
            
        Returns:
            DCG at k
        """
        if k <= 0 or not ranked_docs:
            return 0.0
            
        relevant_set = set(relevant_docs)
        dcg = 0.0
        
        for i, doc_id in enumerate(ranked_docs[:k]):
            if doc_id in relevant_set:
                # Use provided relevance score or default to 1
                relevance = relevance_scores.get(doc_id, 1.0) if relevance_scores else 1.0
                discount = math.log2(i + 2)  # i+2 because i is 0-indexed
                dcg += relevance / discount
                
        return dcg
    
    @staticmethod
    def ndcg_at_k(relevant_docs: List[int], 
                 ranked_docs: List[int], 
                 k: int,
                 relevance_scores: Optional[Dict[int, float]] = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            relevant_docs: List of relevant document indices
            ranked_docs: List of documents ranked by score (descending)
            k: Cutoff rank
            relevance_scores: Optional dict mapping doc_id to relevance score
            
        Returns:
            NDCG at k
        """
        dcg = ScoringMetrics.dcg_at_k(relevant_docs, ranked_docs, k, relevance_scores)
        
        # Compute ideal DCG (IDCG)
        if relevance_scores:
            # Sort relevant docs by relevance score (descending)
            sorted_relevant = sorted(
                relevant_docs,
                key=lambda doc_id: relevance_scores.get(doc_id, 1.0),
                reverse=True
            )
        else:
            sorted_relevant = relevant_docs
            
        idcg = ScoringMetrics.dcg_at_k(relevant_docs, sorted_relevant, k, relevance_scores)
        
        return dcg / idcg if idcg > 0 else 0.0


class RankingMetrics:
    """
    Metrics for comparing rankings between different methods.
    """
    
    @staticmethod
    def kendall_tau(ranking1: List[int], ranking2: List[int]) -> float:
        """
        Compute Kendall's tau correlation between two rankings.
        
        Args:
            ranking1: First ranking (list of document indices)
            ranking2: Second ranking (list of document indices)
            
        Returns:
            Kendall's tau correlation (-1 to 1)
        """
        # Find common documents
        common_docs = set(ranking1) & set(ranking2)
        
        if len(common_docs) < 2:
            return 0.0
            
        # Create position mappings
        pos1 = {doc: i for i, doc in enumerate(ranking1) if doc in common_docs}
        pos2 = {doc: i for i, doc in enumerate(ranking2) if doc in common_docs}
        
        # Count concordant and discordant pairs
        concordant = 0
        discordant = 0
        
        docs = list(common_docs)
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                doc_i, doc_j = docs[i], docs[j]
                
                # Check if pair order is same in both rankings
                order1 = pos1[doc_i] < pos1[doc_j]
                order2 = pos2[doc_i] < pos2[doc_j]
                
                if order1 == order2:
                    concordant += 1
                else:
                    discordant += 1
        
        total_pairs = concordant + discordant
        if total_pairs == 0:
            return 0.0
            
        return (concordant - discordant) / total_pairs
    
    @staticmethod
    def spearman_correlation(ranking1: List[int], ranking2: List[int]) -> float:
        """
        Compute Spearman's rank correlation between two rankings.
        
        Args:
            ranking1: First ranking (list of document indices)
            ranking2: Second ranking (list of document indices)
            
        Returns:
            Spearman's correlation (-1 to 1)
        """
        # Find common documents
        common_docs = list(set(ranking1) & set(ranking2))
        
        if len(common_docs) < 2:
            return 0.0
            
        # Create position mappings
        pos1 = {doc: i for i, doc in enumerate(ranking1) if doc in common_docs}
        pos2 = {doc: i for i, doc in enumerate(ranking2) if doc in common_docs}
        
        # Calculate rank differences
        rank_diffs = [pos1[doc] - pos2[doc] for doc in common_docs]
        
        n = len(common_docs)
        sum_diff_squared = sum(d * d for d in rank_diffs)
        
        # Spearman's formula
        correlation = 1 - (6 * sum_diff_squared) / (n * (n * n - 1))
        
        return correlation
    
    @staticmethod
    def rank_overlap_at_k(ranking1: List[int], 
                         ranking2: List[int], 
                         k: int) -> float:
        """
        Compute overlap between top-k items in two rankings.
        
        Args:
            ranking1: First ranking
            ranking2: Second ranking
            k: Cutoff rank
            
        Returns:
            Overlap ratio (0 to 1)
        """
        if k <= 0:
            return 0.0
            
        top_k1 = set(ranking1[:k])
        top_k2 = set(ranking2[:k])
        
        intersection = len(top_k1 & top_k2)
        union = len(top_k1 | top_k2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def average_overlap_at_k(rankings: List[List[int]], k: int) -> float:
        """
        Compute average pairwise overlap at k across multiple rankings.
        
        Args:
            rankings: List of rankings
            k: Cutoff rank
            
        Returns:
            Average overlap ratio
        """
        if len(rankings) < 2:
            return 0.0
            
        total_overlap = 0.0
        pair_count = 0
        
        for i in range(len(rankings)):
            for j in range(i + 1, len(rankings)):
                overlap = RankingMetrics.rank_overlap_at_k(rankings[i], rankings[j], k)
                total_overlap += overlap
                pair_count += 1
                
        return total_overlap / pair_count if pair_count > 0 else 0.0


def evaluate_scorer_performance(scorer, 
                               queries_and_docs: List[Tuple[str, List[Dict[str, Any]], List[int]]],
                               k_values: List[int] = None) -> Dict[str, Any]:
    """
    Evaluate a scorer's performance across multiple queries.
    
    Args:
        scorer: Scorer instance to evaluate
        queries_and_docs: List of (query, documents, relevant_indices) tuples
        k_values: List of k values for precision@k, recall@k, etc.
        
    Returns:
        Dictionary with evaluation results
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]
        
    results = {
        'num_queries': len(queries_and_docs),
        'k_values': k_values,
        'precision_at_k': {k: [] for k in k_values},
        'recall_at_k': {k: [] for k in k_values},
        'f1_at_k': {k: [] for k in k_values},
        'ndcg_at_k': {k: [] for k in k_values},
        'average_precision': [],
        'total_relevant_docs': 0,
        'total_documents': 0
    }
    
    for query, documents, relevant_indices in queries_and_docs:
        # Get ranking from scorer
        ranking = scorer.rank_documents(query, documents)
        ranked_doc_indices = [doc_idx for doc_idx, _ in ranking]
        
        # Update totals
        results['total_relevant_docs'] += len(relevant_indices)
        results['total_documents'] += len(documents)
        
        # Compute metrics for each k
        for k in k_values:
            precision = ScoringMetrics.precision_at_k(relevant_indices, ranked_doc_indices, k)
            recall = ScoringMetrics.recall_at_k(relevant_indices, ranked_doc_indices, k)
            f1 = ScoringMetrics.f1_at_k(relevant_indices, ranked_doc_indices, k)
            ndcg = ScoringMetrics.ndcg_at_k(relevant_indices, ranked_doc_indices, k)
            
            results['precision_at_k'][k].append(precision)
            results['recall_at_k'][k].append(recall)
            results['f1_at_k'][k].append(f1)
            results['ndcg_at_k'][k].append(ndcg)
        
        # Compute average precision
        ap = ScoringMetrics.average_precision(relevant_indices, ranked_doc_indices)
        results['average_precision'].append(ap)
    
    # Compute averages
    results['mean_average_precision'] = sum(results['average_precision']) / len(results['average_precision'])
    
    for k in k_values:
        results[f'mean_precision_at_{k}'] = sum(results['precision_at_k'][k]) / len(results['precision_at_k'][k])
        results[f'mean_recall_at_{k}'] = sum(results['recall_at_k'][k]) / len(results['recall_at_k'][k])
        results[f'mean_f1_at_{k}'] = sum(results['f1_at_k'][k]) / len(results['f1_at_k'][k])
        results[f'mean_ndcg_at_{k}'] = sum(results['ndcg_at_k'][k]) / len(results['ndcg_at_k'][k])
    
    return results
