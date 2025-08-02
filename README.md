# DE Interpreter

A sophisticated agent that transforms differential expression (DE) results into interpretable, literature-backed discussions contextualized by disease and experimental conditions.

## Overview

Traditional pathway analysis often produces generic lists of enriched pathways that lack disease-specific context. DE Interpreter addresses this by:

- **Prioritizing genes** based on statistical and biological importance
- **Mining relevant literature** using the FutureHouse Paper Search API
- **Generating contextual discussions** using Claude-3 Opus
- **Producing comprehensive reports** with gene-specific and cluster-based analyses

## Features

- 📊 **Smart Gene Prioritization**: Combines statistical significance, effect size, and biological relevance
- 📚 **Literature Integration**: Automatically retrieves and incorporates recent research
- 🤖 **AI-Powered Synthesis**: Uses Claude to generate coherent, referenced discussions
- 🔬 **Disease-Specific Context**: Tailors interpretations to your experimental conditions
- 📝 **Professional Reports**: Generates structured markdown reports with citations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/de_interpreter.git
cd de_interpreter

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys:
# ANTHROPIC_API_KEY=your_claude_api_key
# FUTUREHOUSE_API_KEY=your_futurehouse_api_key
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

Optional columns:
- gene_symbol
- baseMean (or base_mean)
- gene_name

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

```bash
# Download example data
wget https://example.com/alzheimers_de_results.csv
wget https://example.com/alzheimers_metadata.json

# Run analysis
python -m de_interpreter.main \
    --de-file alzheimers_de_results.csv \
    --metadata alzheimers_metadata.json \
    --output alzheimers_analysis \
    --top-n 100
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
   DE Parser          Statistical &         FutureHouse      Claude API    Markdown
   Metadata Parser    Biological Scoring    Paper Search                   Reports
```

## License

MIT License

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## Citation

If you use DE Interpreter in your research, please cite:

```
DE Interpreter: AI-Powered Differential Expression Analysis
https://github.com/yourusername/de_interpreter
```