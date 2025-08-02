# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a **Unified Multi-Omics Interpretation Pipeline** that transforms differential omics results into literature-backed, disease-contextualized discussions. The pipeline supports **7 omics types**: transcriptomics, proteomics, metabolomics, genomics, metagenomics, epigenomics, and lipidomics with **automatic omics type detection**. It uses the Claude API for synthesis and PMC (PubMed Central) for literature mining with full-text access and optional AI-powered relevance scoring.

## Core Architecture

### Key Components
1. **Unified Input Processing**: Automatically detects and parses any omics data type with intelligent column mapping
2. **Omics-Aware Prioritization**: Filters and ranks features using omics-specific biological scoring
3. **Literature Mining**: Uses PMC for full-text papers with optional BioBERT/TF-IDF/BM25 relevance scoring
4. **Unified Synthesis Engine**: Generates omics-specific discussions using specialized prompts
5. **Omics-Agnostic Reporting**: Produces structured reports adapted to each omics type

### Main Modules
- `src/parsers/`: **Unified parsing** for all omics data types with auto-detection
- `src/prioritization/`: **Omics-aware** feature ranking with type-specific scoring
- `src/literature/`: PMC integration with **BioBERT scoring** and full-text extraction
- `src/synthesis/`: Claude API integration with **omics-specific prompts**
- `src/reporting/`: **Adaptive reporting** for all omics types

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

**Simplified Pipeline (Recommended)**
```bash
# Auto-detects omics type from data
python -m de_interpreter.main \
  --de-file results.csv \
  --metadata metadata.json \
  --output report.md

# Advanced options with scoring
python -m de_interpreter.main \
  --de-file results.csv \
  --metadata metadata.json \
  --output report.md \
  --top-n 100 \                   # Prioritize top 100 features
  --max-analysis 25 \             # Analyze top 25 features in detail
  --use-scoring \                 # Enable literature relevance scoring
  --scorer-type biobert \         # Use BioBERT for best quality (tfidf, bm25, biobert)
  --biobert-model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --no-cache                      # Disable literature caching
```

### Literature Scoring Options

Install optional scoring dependencies:
```bash
pip install -r requirements-scoring.txt
```

Scoring methods available:
```bash
# TF-IDF scoring (fast, good baseline)
python -m de_interpreter.main --de-file data.csv --use-scoring --scorer-type tfidf

# BM25 scoring (balanced speed/quality)  
python -m de_interpreter.main --de-file data.csv --use-scoring --scorer-type bm25

# BioBERT scoring (best quality, slower)
python -m de_interpreter.main --de-file data.csv --use-scoring --scorer-type biobert \
  --biobert-model dmis-lab/biobert-base-cased-v1.1
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
# PMC with optional scoring for enhanced relevance
async with PMCClient(use_scoring=True, scorer_type="biobert") as client:
    for gene in prioritized_genes:
        query = f"{gene} {disease} expression function"
        papers = await client.search(query, limit=10)
        cache_results(papers)
```

### Scoring Options
- **PMC Only**: Fast, free full-text retrieval without relevance ranking
- **PMC + TF-IDF**: Balanced speed and relevance using traditional scoring
- **PMC + BM25**: Good relevance ranking with moderate computational cost
- **PMC + BioBERT**: Highest quality semantic matching (requires ML dependencies)

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