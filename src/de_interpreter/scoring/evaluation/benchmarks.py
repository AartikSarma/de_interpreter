"""
Benchmark datasets for evaluating scoring methods.
"""

from typing import List, Dict, Any, Tuple, Optional
import json
import os
import random
from abc import ABC, abstractmethod


class BenchmarkDataset(ABC):
    """
    Abstract base class for benchmark datasets.
    
    Provides a common interface for loading and using benchmark
    datasets to evaluate scoring methods.
    """
    
    def __init__(self, name: str):
        """
        Initialize benchmark dataset.
        
        Args:
            name: Name of the benchmark dataset
        """
        self.name = name
        self.queries = []
        self.documents = []
        self.relevance_judgments = {}
        self.loaded = False
    
    @abstractmethod
    def load(self, data_path: str):
        """
        Load the benchmark dataset from files.
        
        Args:
            data_path: Path to the dataset files
        """
        pass
    
    def get_queries(self) -> List[Dict[str, Any]]:
        """
        Get all queries in the dataset.
        
        Returns:
            List of query dictionaries
        """
        return self.queries
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents in the dataset.
        
        Returns:
            List of document dictionaries
        """
        return self.documents
    
    def get_relevant_documents(self, query_id: str) -> List[int]:
        """
        Get relevant document indices for a query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            List of relevant document indices
        """
        return self.relevance_judgments.get(query_id, [])
    
    def get_query_document_pairs(self) -> List[Tuple[str, List[Dict[str, Any]], List[int]]]:
        """
        Get all query-document pairs with relevance judgments.
        
        Returns:
            List of (query_text, documents, relevant_indices) tuples
        """
        pairs = []
        
        for query in self.queries:
            query_id = query.get('id', '')
            query_text = query.get('text', '')
            relevant_indices = self.get_relevant_documents(query_id)
            
            if query_text and relevant_indices:
                pairs.append((query_text, self.documents, relevant_indices))
                
        return pairs
    
    def create_train_test_split(self, 
                               test_ratio: float = 0.2,
                               random_seed: Optional[int] = None) -> Tuple[List, List]:
        """
        Create train/test split of query-document pairs.
        
        Args:
            test_ratio: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_pairs, test_pairs)
        """
        if random_seed is not None:
            random.seed(random_seed)
            
        pairs = self.get_query_document_pairs()
        random.shuffle(pairs)
        
        split_idx = int(len(pairs) * (1 - test_ratio))
        train_pairs = pairs[:split_idx]
        test_pairs = pairs[split_idx:]
        
        return train_pairs, test_pairs


class PubMedBenchmark(BenchmarkDataset):
    """
    PubMed-based benchmark dataset for biomedical literature scoring.
    
    This creates a synthetic benchmark using PubMed abstracts
    and gene-disease associations for relevance judgments.
    """
    
    def __init__(self):
        super().__init__("PubMed-Biomedical")
        
    def load(self, data_path: str):
        """
        Load PubMed benchmark data.
        
        Args:
            data_path: Path to the benchmark data directory
        """
        # Load queries (gene sets + conditions)
        queries_file = os.path.join(data_path, "queries.json")
        if os.path.exists(queries_file):
            with open(queries_file, 'r') as f:
                self.queries = json.load(f)
        
        # Load documents (PubMed abstracts)
        documents_file = os.path.join(data_path, "documents.json")
        if os.path.exists(documents_file):
            with open(documents_file, 'r') as f:
                self.documents = json.load(f)
        
        # Load relevance judgments
        relevance_file = os.path.join(data_path, "relevance.json")
        if os.path.exists(relevance_file):
            with open(relevance_file, 'r') as f:
                self.relevance_judgments = json.load(f)
        
        self.loaded = True
    
    def create_synthetic_benchmark(self,
                                  gene_lists: List[List[str]],
                                  conditions: List[str],
                                  pubmed_abstracts: List[Dict[str, Any]],
                                  num_queries: int = 50) -> None:
        """
        Create a synthetic benchmark from gene lists and PubMed abstracts.
        
        Args:
            gene_lists: List of gene symbol lists (from DE analyses)
            conditions: List of conditions/diseases
            pubmed_abstracts: List of PubMed abstracts
            num_queries: Number of queries to create
        """
        # Create queries from gene lists and conditions
        self.queries = []
        for i in range(min(num_queries, len(gene_lists))):
            gene_list = gene_lists[i]
            condition = conditions[i % len(conditions)] if conditions else ""
            
            query = {
                'id': f"query_{i}",
                'text': f"{' '.join(gene_list[:10])} {condition}".strip(),
                'genes': gene_list,
                'condition': condition
            }
            self.queries.append(query)
        
        # Use provided abstracts as documents
        self.documents = pubmed_abstracts
        
        # Create synthetic relevance judgments
        self.relevance_judgments = {}
        for query in self.queries:
            relevant_docs = self._find_relevant_documents(query)
            self.relevance_judgments[query['id']] = relevant_docs
        
        self.loaded = True
    
    def _find_relevant_documents(self, query: Dict[str, Any]) -> List[int]:
        """
        Find relevant documents for a query using simple keyword matching.
        
        Args:
            query: Query dictionary
            
        Returns:
            List of relevant document indices
        """
        query_genes = set(gene.lower() for gene in query.get('genes', []))
        query_condition = query.get('condition', '').lower()
        
        relevant_indices = []
        
        for i, doc in enumerate(self.documents):
            title = doc.get('title', '').lower()
            abstract = doc.get('abstract', '').lower()
            doc_text = f"{title} {abstract}"
            
            # Count gene matches
            gene_matches = sum(1 for gene in query_genes if gene in doc_text)
            
            # Check condition match
            condition_match = query_condition in doc_text if query_condition else False
            
            # Consider relevant if has gene matches or condition match
            if gene_matches >= 2 or condition_match:
                relevant_indices.append(i)
        
        return relevant_indices
    
    def save_benchmark(self, output_path: str):
        """
        Save the benchmark dataset to files.
        
        Args:
            output_path: Directory to save the benchmark files
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Save queries
        with open(os.path.join(output_path, "queries.json"), 'w') as f:
            json.dump(self.queries, f, indent=2)
        
        # Save documents
        with open(os.path.join(output_path, "documents.json"), 'w') as f:
            json.dump(self.documents, f, indent=2)
        
        # Save relevance judgments
        with open(os.path.join(output_path, "relevance.json"), 'w') as f:
            json.dump(self.relevance_judgments, f, indent=2)
        
        # Save metadata
        metadata = {
            'name': self.name,
            'num_queries': len(self.queries),
            'num_documents': len(self.documents),
            'total_relevance_judgments': sum(len(docs) for docs in self.relevance_judgments.values())
        }
        
        with open(os.path.join(output_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)


class TRECBenchmark(BenchmarkDataset):
    """
    TREC-style benchmark dataset adapter.
    
    Loads datasets in TREC format for information retrieval evaluation.
    """
    
    def __init__(self, name: str = "TREC"):
        super().__init__(name)
    
    def load(self, data_path: str):
        """
        Load TREC-format benchmark data.
        
        Args:
            data_path: Path to TREC format files
        """
        # Load queries from topics file
        topics_file = os.path.join(data_path, "topics.txt")
        if os.path.exists(topics_file):
            self.queries = self._load_trec_topics(topics_file)
        
        # Load documents
        docs_file = os.path.join(data_path, "documents.txt")
        if os.path.exists(docs_file):
            self.documents = self._load_trec_documents(docs_file)
        
        # Load relevance judgments (qrels)
        qrels_file = os.path.join(data_path, "qrels.txt")
        if os.path.exists(qrels_file):
            self.relevance_judgments = self._load_trec_qrels(qrels_file)
        
        self.loaded = True
    
    def _load_trec_topics(self, topics_file: str) -> List[Dict[str, Any]]:
        """Load TREC topics file."""
        queries = []
        
        with open(topics_file, 'r') as f:
            content = f.read()
            
        # Simple parser for TREC topics
        # This is a basic implementation - real TREC files may need more sophisticated parsing
        topics = content.split('<top>')
        
        for topic in topics[1:]:  # Skip first empty split
            lines = topic.strip().split('\n')
            query_dict = {'id': '', 'text': '', 'title': '', 'description': ''}
            
            for line in lines:
                line = line.strip()
                if line.startswith('<num>'):
                    query_dict['id'] = line.replace('<num>', '').replace('Number:', '').strip()
                elif line.startswith('<title>'):
                    query_dict['title'] = line.replace('<title>', '').strip()
                    query_dict['text'] = query_dict['title']  # Use title as main text
                elif line.startswith('<desc>'):
                    # Description continues until next tag
                    desc_start = lines.index(line)
                    desc_lines = []
                    for desc_line in lines[desc_start + 1:]:
                        if desc_line.startswith('<'):
                            break
                        desc_lines.append(desc_line.strip())
                    query_dict['description'] = ' '.join(desc_lines)
            
            if query_dict['id'] and query_dict['text']:
                queries.append(query_dict)
        
        return queries
    
    def _load_trec_documents(self, docs_file: str) -> List[Dict[str, Any]]:
        """Load TREC documents file."""
        documents = []
        
        with open(docs_file, 'r') as f:
            lines = f.readlines()
        
        # Simple document format: each line is "doc_id\ttitle\tabstract"
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                doc = {
                    'id': parts[0],
                    'title': parts[1],
                    'abstract': parts[2]
                }
                documents.append(doc)
        
        return documents
    
    def _load_trec_qrels(self, qrels_file: str) -> Dict[str, List[int]]:
        """Load TREC qrels (relevance judgments) file."""
        relevance_judgments = {}
        
        with open(qrels_file, 'r') as f:
            lines = f.readlines()
        
        # TREC qrels format: query_id iteration doc_id relevance
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])
                
                # Only consider relevant documents (relevance > 0)
                if relevance > 0:
                    if query_id not in relevance_judgments:
                        relevance_judgments[query_id] = []
                    
                    # Find document index
                    for i, doc in enumerate(self.documents):
                        if doc.get('id') == doc_id:
                            relevance_judgments[query_id].append(i)
                            break
        
        return relevance_judgments


def create_cross_validation_splits(benchmark: BenchmarkDataset, 
                                  n_folds: int = 5,
                                  random_seed: Optional[int] = None) -> List[Tuple[List, List]]:
    """
    Create cross-validation splits for a benchmark dataset.
    
    Args:
        benchmark: Benchmark dataset
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
        
    Returns:
        List of (train_pairs, test_pairs) tuples for each fold
    """
    if random_seed is not None:
        random.seed(random_seed)
        
    pairs = benchmark.get_query_document_pairs()
    random.shuffle(pairs)
    
    fold_size = len(pairs) // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(pairs)
        
        test_pairs = pairs[start_idx:end_idx]
        train_pairs = pairs[:start_idx] + pairs[end_idx:]
        
        folds.append((train_pairs, test_pairs))
    
    return folds
