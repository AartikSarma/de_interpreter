# Omics interpreter

An LLM-powered pipeline to find and summarize relevant papers to interpret the results of omics analyses. 

## Features
- 🔎 **Structured query generation**: Uses Claude Haiku to generate search terms based on a description of experimetnal conditions
- 📚 **Literature Integration**: Automatically retrieves and incorporates recent research
- 🤖 **AI-Powered Synthesis**: Uses Claude Sonnet to generate coherent, referenced discussions
- 🔬 **Disease-Specific Context**: Prompts tailor interpretations to your experimental conditions
- 📝 **Professional Reports**: Generates structured markdown reports with citations

## Installation

```bash
# Clone the repository
git clone https://github.com/AartikSarma/de_interpreter.git
cd de_interpreter

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_claude_api_key
```

## Usage

### Command Line

```bash
python -m de_interpreter.main \
    --de-file results.csv \
    --metadata metadata.json \
    --output my_analysis \
    --top-n 50
```

### Python API

```python
import asyncio
from pathlib import Path
from de_interpreter.main import DEInterpreter

async def analyze():
    interpreter = DEInterpreter(top_n_genes=50)
    
    report_path = await interpreter.run(
        de_file=Path("results.csv"),
        metadata_file=Path("metadata.json"),
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

> **Bost, P., Giladi, A., Liu, Y. et al.** Host-Viral Infection Maps Reveal Signatures of Severe COVID-19 Patients. *Nat Commun* **12**, 4312 (2021). https://doi.org/10.1038/s41467-021-25040-5

This dataset contains differential gene expression results from respiratory samples comparing COVID-19 ARDS patients to controls.

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

## Development

```bash
# Run tests
pytest tests/

# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/
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
