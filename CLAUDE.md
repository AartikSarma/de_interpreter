# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Multi-Omics Interpretation Pipeline that transforms differential omics results into literature-backed, disease-contextualized discussions. The pipeline supports transcriptomics, proteomics, metabolomics, genomics, metagenomics, epigenomics, and lipidomics data. It uses the Claude API for synthesis and PMC (PubMed Central) for literature mining with full-text access.

## Core Architecture

### Key Components
1. **Input Processing**: Parses DE results (CSV/TSV) and experimental metadata
2. **Gene Prioritization**: Filters and ranks genes by statistical and biological importance
3. **Literature Mining**: Uses PMC (PubMed Central) to retrieve full-text research papers
4. **Synthesis Engine**: Uses Claude API to generate coherent discussions
5. **Report Generator**: Produces structured, referenced output documents

### Main Modules
- `src/parsers/`: Input parsing for DE results and metadata
- `src/prioritization/`: Gene ranking and clustering algorithms
- `src/literature/`: PMC integration and paper processing with full-text extraction
- `src/synthesis/`: Claude API integration and prompt engineering
- `src/reporting/`: Output generation and formatting

## Usage Options

### Web Interface (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit web interface
python run_streamlit.py
# OR
streamlit run streamlit_app.py
```

### Command Line Interface
```bash
# Run the agent directly
python -m de_interpreter.main --de-file results.csv --metadata metadata.json --output report.md

# Advanced options
python -m de_interpreter.main \
  --de-file results.csv \
  --metadata metadata.json \
  --output report.md \
  --top-n 100 \               # Prioritize top 100 genes
  --max-analysis 25 \         # Analyze top 25 genes in detail
  --no-cache \                # Disable literature caching
  --use-futurehouse           # Use FutureHouse API instead of PMC
```

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific test
pytest tests/test_parsers.py::test_de_parser

# Lint code
ruff check src/
black src/ --check

# Format code
black src/
ruff check src/ --fix

# Type checking
mypy src/

# Run the agent
python -m de_interpreter.main --de-file results.csv --metadata metadata.json --output report.md
```

## API Configuration

### Environment Variables
Create a `.env` file (not tracked in git):
```
ANTHROPIC_API_KEY=your_claude_api_key
FUTUREHOUSE_API_KEY=your_futurehouse_api_key  # Optional - PMC is used by default
```

### Literature Sources
- **PMC (Default)**: Free access to PubMed Central full-text papers, no API key required
- **FutureHouse API (Optional)**: Commercial API with rate limits, requires API key
- Use `--use-futurehouse` flag or `use_pmc=False` parameter to switch to FutureHouse

### API Usage Guidelines
- **PMC**: Free, full-text access, automatic XML parsing, respectful rate limiting
- **FutureHouse API**: Batch gene queries, implement caching, respect rate limits  
- **Claude API**: Use structured prompts, prefer claude-sonnet-4 for complex synthesis

## Key Design Patterns

### Gene Prioritization Strategy
1. Filter by statistical significance (padj < 0.05)
2. Filter by effect size (|log2FC| > 1)
3. Score by biological importance (disease relevance, druggability)
4. Cluster co-expressed genes

### Literature Search Pattern
```python
# Batch queries for efficiency
gene_batches = chunk_genes(prioritized_genes, batch_size=10)
for batch in gene_batches:
    query = build_query(genes=batch, disease=context.disease)
    papers = futurehouse_api.search(query, limit=20)
    cache_results(papers)
```

### Synthesis Prompt Structure
1. Provide gene information and expression change
2. Include relevant literature excerpts
3. Request structured discussion with:
   - Functional role in disease
   - Interpretation of expression change
   - Therapeutic implications
   - Supporting citations

## Testing Strategy
- Unit tests for all parsers
- Mock API responses for integration tests
- Example DE datasets in `tests/data/`
- Validate output structure and citations

## Important Considerations
- Handle large DE results efficiently (1000+ genes)
- Implement robust error handling for API failures
- Cache literature results to minimize API costs
- Ensure all generated text includes proper citations
- Validate biological claims against source papers