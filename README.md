# Omics interpreter

An LLM-powered pipeline to find and summarize relevant papers to interpret the results of omics analyses. 

## Features
- 🧬 **Multi-Omics Support**: Transcriptomics, proteomics, metabolomics, genomics, metagenomics, epigenomics, lipidomics
- 🔎 **Smart Auto-Detection**: Automatically detects omics type from your data columns
- 📚 **Advanced Literature Mining**: PMC integration with multiple scoring algorithms
- 🏷️ **MeSH Enhancement**: Claude Haiku generates Medical Subject Headings for precise searches
- 🎯 **Gene-Aware Scoring**: Specialized scoring methods optimized for omics data
- 🤖 **AI-Powered Synthesis**: Uses Claude Sonnet to generate coherent, referenced discussions
- 🔬 **Disease-Specific Context**: Prompts tailor interpretations to your experimental conditions
- 📝 **Professional Reports**: Generates structured markdown reports with citations
- 🌐 **Web Interface**: User-friendly Streamlit interface for easy analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/AartikSarma/de_interpreter.git
cd de_interpreter

# Install dependencies (includes all features: scoring, MeSH enhancement, web interface)
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_claude_api_key
```

## Usage

### Command Line (Unified Interface)

```bash
# Basic analysis (auto-detects omics type)
python -m de_interpreter \
    --de-file results.csv \
    --metadata metadata.json \
    --output my_analysis \
    --max-features 25

# Advanced analysis with all features
python -m de_interpreter \
    --de-file results.csv \
    --metadata metadata.json \
    --output my_analysis \
    --max-features 25 \
    --use-scoring \
    --scorer-type gene_query_similarity \
    --use-mesh \
    --mesh-terms-count 3

# Launch web interface
python -m de_interpreter --web
```

### Web Interface (Recommended)

```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py
# OR
python -m de_interpreter --web
```

### Python API

```python
import asyncio
from de_interpreter.main import SimplifiedPipeline, AnalysisConfig

async def analyze():
    # Configure analysis
    config = AnalysisConfig(
        max_features=25,
        use_scoring=True,
        scorer_type="gene_query_similarity",
        use_mesh_enhancement=True,
        anthropic_api_key="your-api-key"
    )
    
    # Create pipeline
    pipeline = SimplifiedPipeline(config)
    
    # Run analysis
    report_path = await pipeline.run_analysis(
        de_file="results.csv",
        metadata_file="metadata.json",
        output_name="my_analysis"
    )
    
    print(f"Report generated: {report_path}")

asyncio.run(analyze())
```

## Input Formats

### Differential Expression Results (CSV/TSV/Excel)

Required columns:
- Gene identifier (gene_id, ensembl_id, or similar)
- log2FoldChange (or log2FC, logFC)
- pvalue (or p_value)
- padj (or p_adj, FDR)


### Metadata File (JSON/YAML)

```json
{
  "disease": "Alzheimer's disease",
  "tissue": "hippocampus",
  "cell_type": "neurons",
  "treatment": "amyloid-beta oligomers",
  "control": "vehicle",
  "time_point": "24 hours",
  "organism": "human",
  "sample_size": {
    "treatment": 6,
    "control": 6
  }
}
```

## Output

The tool generates:

1. **Markdown Report** (`output/my_analysis.md`)
   - Executive summary
   - Gene-by-gene discussions with citations
   - Cluster analyses
   - Methods description

2. **Metadata JSON** (`output/my_analysis_metadata.json`)
   - Analysis parameters
   - Summary statistics

## Example

### COVID-19 ARDS Example Data

The repository includes example COVID-19 ARDS transcriptomics data from:

> **Sarma, A., Christenson, S.A., Byrne, A. et al.** Tracheal aspirate RNA sequencing identifies distinct immunological features of COVID-19 ARDS. *Nat Commun* **12**, 5152 (2021). https://doi.org/10.1038/s41467-021-25040-5

This dataset contains differential gene expression results from tracheal aspirate RNA sequencing comparing COVID-19 ARDS patients to controls.

### Running the Example

```bash
# Using the web interface (recommended)
streamlit run streamlit_app.py
# Then click "🦠 COVID-19 ARDS" example button

# Or via command line
python -m de_interpreter.main \
    --de-file covid_data/covid_deg_fixed.csv \
    --metadata covid_data/covid_metadata.json \
    --output covid_analysis \
    --max-features 25
```

## Advanced Features

### Literature Scoring Methods

Choose from multiple scoring algorithms optimized for different use cases:

- **TF-IDF**: Fast baseline similarity scoring
- **BM25**: Balanced relevance ranking 
- **BioBERT**: Semantic similarity using biomedical language models
- **Gene-Query Similarity**: Enhanced scoring specifically designed for omics data

### MeSH Term Enhancement

Automatically generate Medical Subject Headings using Claude Haiku to improve literature search precision:

```bash
python -m de_interpreter \
    --de-file data.csv \
    --use-mesh \
    --mesh-terms-count 4 \
    --use-scoring \
    --scorer-type biobert
```

### Utility Tools

```bash
# Benchmark different scoring methods
python -m de_interpreter.utils.benchmarking

# Run usage examples
python -m de_interpreter.utils.examples scoring
```

## Development

```bash
# Run tests
pytest tests/

# Run specific test categories
pytest tests/unit/           # Unit tests
pytest tests/integration/    # Integration tests

# Format code
black de_interpreter/
ruff check de_interpreter/ --fix

# Type checking
mypy de_interpreter/
```

## Architecture

```
Input Processing → Gene Prioritization → Literature Mining → AI Synthesis → Report Generation
      ↓                    ↓                    ↓                ↓              ↓
   DE Parser          Statistical &         PMC/Literature   Claude API    Markdown
   Metadata Parser    Biological Scoring    Scoring                        Reports
```

## Additional Tools

### Standalone Scoring Scripts (`scripts/`)

The project includes several standalone utility scripts for gene-query literature analysis:

#### Gene-Query Similarity Scoring
```bash
# Basic gene scoring against a literature query
python scripts/gene_query_similarity_scorer.py \
    --genes TP53 BRCA1 MYC \
    --query "cancer progression" \
    --top-papers 20

# With output file
python scripts/gene_query_similarity_scorer.py \
    --genes FAM71A P2RY14 CAB39L \
    --query "COVID-19 inflammatory response" \
    --output results.json
```

#### Claude-Enhanced Scoring
```bash
# Use Claude Haiku to generate MeSH terms for enhanced searches
python scripts/claude_enhanced_gene_scorer.py \
    --genes IFNG TNF IL6 \
    --query "immune response" \
    --mesh-terms 5
```

#### Integration Examples
```bash
# Example showing how to integrate with de_interpreter components
python scripts/example_gene_query_scoring.py
```

See `scripts/README.md` for detailed documentation and `docs/` for comprehensive guides.

### Benchmarking Tools
```bash
# Benchmark different scoring methods
python scripts/benchmark_scoring.py

# Test individual paper scoring
python scripts/score_query_vs_pmid.py "cancer gene expression" 12345678
```

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Citation

If you use DE Interpreter in your research, please cite:

```
Omics Interpreter: AI-Powered Omics Analysis
https://github.com/AartikSarma/de_interpreter
```
