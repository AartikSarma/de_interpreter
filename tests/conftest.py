"""Shared test configuration and fixtures for DE Interpreter tests."""

import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

from de_interpreter.main import AnalysisConfig
from de_interpreter.parsers.omics_data import OmicsType


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_de_data():
    """Sample DE results data."""
    return [
        {
            "gene_id": "ENSG00000141510",
            "gene_symbol": "TP53",
            "log2FoldChange": 2.5,
            "pvalue": 0.001,
            "padj": 0.01
        },
        {
            "gene_id": "ENSG00000012048", 
            "gene_symbol": "BRCA1",
            "log2FoldChange": -1.8,
            "pvalue": 0.005,
            "padj": 0.02
        },
        {
            "gene_id": "ENSG00000136997",
            "gene_symbol": "MYC",
            "log2FoldChange": 3.2,
            "pvalue": 0.0001,
            "padj": 0.005
        }
    ]


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "disease": "cancer",
        "tissue": "tumor",
        "cell_type": "epithelial",
        "treatment": "drug_A",
        "control": "vehicle",
        "time_point": "24h",
        "organism": "human",
        "sample_size": {
            "treatment": 6,
            "control": 6
        }
    }


@pytest.fixture
def basic_config():
    """Basic analysis configuration for testing."""
    return AnalysisConfig(
        max_features=3,
        use_cache=False,
        anthropic_api_key=None,  # No API key for testing
        progress_callback=None,
        use_scoring=False
    )


@pytest.fixture
def scoring_config():
    """Configuration with scoring enabled."""
    return AnalysisConfig(
        max_features=3,
        use_cache=False,
        anthropic_api_key=None,
        progress_callback=None,
        use_scoring=True,
        scorer_type="tfidf"
    )


@pytest.fixture
def create_test_files(temp_dir, sample_de_data, sample_metadata):
    """Create test CSV and metadata files."""
    def _create_files(de_data=None, metadata=None):
        if de_data is None:
            de_data = sample_de_data
        if metadata is None:
            metadata = sample_metadata
            
        # Create CSV file
        csv_file = temp_dir / "test_de_results.csv"
        with open(csv_file, 'w') as f:
            f.write("gene_id,gene_symbol,log2FoldChange,pvalue,padj\n")
            for row in de_data:
                f.write(f"{row['gene_id']},{row['gene_symbol']},{row['log2FoldChange']},{row['pvalue']},{row['padj']}\n")
        
        # Create metadata file
        metadata_file = temp_dir / "test_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return csv_file, metadata_file
    
    return _create_files


@pytest.fixture
def mock_progress_callback():
    """Mock progress callback that captures messages."""
    messages = []
    
    def callback(message: str, percent: int):
        messages.append((message, percent))
    
    callback.messages = messages
    return callback