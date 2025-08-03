# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a **Unified Multi-Omics Interpretation Pipeline** that transforms differential omics results into literature-backed, disease-contextualized discussions. The pipeline supports **7 omics types**: transcriptomics, proteomics, metabolomics, genomics, metagenomics, epigenomics, and lipidomics with **automatic omics type detection**. It uses the Claude API for synthesis and PMC (PubMed Central) for literature mining with full-text access, optional AI-powered relevance scoring, and **MeSH term enhancement** for improved literature search precision.

## Core Architecture

### Key Components
1. **Unified Input Processing**: Automatically detects and parses any omics data type with intelligent column mapping
2. **Omics-Aware Prioritization**: Filters and ranks features using omics-specific biological scoring
3. **Literature Mining**: Uses PMC for full-text papers with optional BioBERT/TF-IDF/BM25 relevance scoring and MeSH term enhancement
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

**Unified Pipeline (All-in-One)**
```bash
# Auto-detects omics type from data
python -m de_interpreter \
  --de-file results.csv \
  --metadata metadata.json \
  --output report.md

# Advanced options with all features
python -m de_interpreter \
  --de-file results.csv \
  --metadata metadata.json \
  --output report.md \
  --max-features 25 \             # Analyze top 25 features in detail
  --use-scoring \                 # Enable literature relevance scoring
  --scorer-type gene_query_similarity \  # Enhanced gene-aware scoring (tfidf, bm25, biobert, gene_query_similarity)
  --biobert-model microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract \
  --use-mesh \                    # Enable MeSH term enhancement for literature searches
  --mesh-terms-count 3 \          # Generate 3 MeSH terms per query
  --no-cache                      # Disable literature caching

# Launch web interface
python -m de_interpreter --web
```

### Literature Scoring Options

Scoring methods available (all dependencies included in main requirements.txt):
```bash
# TF-IDF scoring (fast, good baseline)
python -m de_interpreter --de-file data.csv --max-features 20 --use-scoring --scorer-type tfidf

# BM25 scoring (balanced speed/quality)  
python -m de_interpreter --de-file data.csv --max-features 20 --use-scoring --scorer-type bm25

# BioBERT scoring (best quality, slower)
python -m de_interpreter --de-file data.csv --max-features 20 --use-scoring --scorer-type biobert \
  --biobert-model dmis-lab/biobert-base-cased-v1.1

# Gene-Query Similarity scoring (gene-aware, enhanced for omics)
python -m de_interpreter --de-file data.csv --max-features 20 --use-scoring --scorer-type gene_query_similarity
```

### MeSH Term Enhancement

The pipeline can use Claude Haiku to generate Medical Subject Headings (MeSH) terms for enhanced literature searches:

```bash
# Enable MeSH enhancement (requires ANTHROPIC_API_KEY)
python -m de_interpreter \
  --de-file data.csv \
  --use-mesh \                    # Enable MeSH term generation
  --mesh-terms-count 3            # Generate 3 MeSH terms per query (1-5 supported)

# Example: MeSH enhancement with scoring
python -m de_interpreter \
  --de-file cancer_data.csv \
  --use-mesh \
  --mesh-terms-count 4 \
  --use-scoring \
  --scorer-type biobert
```

**MeSH Enhancement Features:**
- Automatically generates relevant MeSH terms using Claude Haiku
- Enhances PubMed search queries with proper MeSH syntax
- Displays generated MeSH terms in progress output
- Falls back gracefully if MeSH generation fails
- Improves literature search precision and recall

## Utility Tools

The package includes utility tools for advanced users:

```bash
# Run benchmarking of scoring methods
python -m de_interpreter.utils.benchmarking

# Run example analyses
python -m de_interpreter.utils.examples basic      # Basic analysis example
python -m de_interpreter.utils.examples scoring    # Compare scoring methods
python -m de_interpreter.utils.examples mesh       # MeSH enhancement example
python -m de_interpreter.utils.examples benchmark  # Benchmark different methods
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
ruff check de_interpreter/
black de_interpreter/ --check

# Format code
black de_interpreter/
ruff check de_interpreter/ --fix

# Type checking
mypy de_interpreter/

# Run the agent
python -m de_interpreter.main --de-file results.csv --metadata metadata.json --output report.md --max-features 25
```

## API Configuration

### Environment Variables
Create a `.env` file (not tracked in git):
```
ANTHROPIC_API_KEY=your_claude_api_key
# No additional API keys needed - PMC is free and used by default
```

### Literature Sources
- **PMC (PubMed Central)**: Free access to full-text papers, no API key required
- **Enhanced Search**: MeSH term generation using Claude Haiku for precision
- **Smart Scoring**: Multiple algorithms (TF-IDF, BM25, BioBERT, Gene-Query Similarity)

### API Usage Guidelines
- **PMC**: Free, full-text access, automatic XML parsing, respectful rate limiting
- **Claude API**: Use structured prompts, claude-sonnet-4 for synthesis, claude-haiku for MeSH terms

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