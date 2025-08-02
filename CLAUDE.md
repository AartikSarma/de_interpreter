# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a Differential Expression (DE) Interpretation Agent that transforms gene expression results into literature-backed, disease-contextualized discussions. The agent uses the Claude API for synthesis and FutureHouse Paper Search API for literature mining.

## Core Architecture

### Key Components
1. **Input Processing**: Parses DE results (CSV/TSV) and experimental metadata
2. **Gene Prioritization**: Filters and ranks genes by statistical and biological importance
3. **Literature Mining**: Uses FutureHouse API to find relevant research papers
4. **Synthesis Engine**: Uses Claude API to generate coherent discussions
5. **Report Generator**: Produces structured, referenced output documents

### Main Modules
- `src/parsers/`: Input parsing for DE results and metadata
- `src/prioritization/`: Gene ranking and clustering algorithms
- `src/literature/`: FutureHouse API integration and paper processing
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
FUTUREHOUSE_API_KEY=your_futurehouse_api_key
```

### API Usage Guidelines
- **FutureHouse API**: Batch gene queries, implement caching, respect rate limits
- **Claude API**: Use structured prompts, prefer claude-3-opus for complex synthesis

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