# Scripts Directory

This directory contains standalone utility scripts for gene-query literature analysis.

## Gene-Query Similarity Scoring Scripts

### `gene_query_similarity_scorer.py` (Main Script)

**Purpose**: Complete implementation of refined gene-query similarity scoring using PubMed and sentence transformers.

**Usage**:
```bash
# Basic usage
python scripts/gene_query_similarity_scorer.py --genes TP53 BRCA1 MYC --query "cancer progression" --top-papers 20

# With output file
python scripts/gene_query_similarity_scorer.py --genes FAM71A P2RY14 CAB39L --query "COVID-19 inflammatory response" --top-papers 10 --output results.json

# From gene file
python scripts/gene_query_similarity_scorer.py --genes-file my_genes.txt --query "metabolic dysfunction" --top-papers 15
```

**Dependencies**: `sentence-transformers`, `scikit-learn`, `requests`

**Approach**:
1. For each gene G: Search PubMed for "query + G"
2. Pool all abstracts from all searches
3. Score each abstract against "query + all_genes_combined"
4. Sum all similarities for total score

### `minimal_gene_scorer.py` (Lightweight Version)

**Purpose**: Simplified version with fewer dependencies for basic similarity scoring.

**Usage**:
```bash
python scripts/minimal_gene_scorer.py --genes TP53 BRCA1 --query "cancer" --top-papers 10
```

**Dependencies**: `sentence-transformers`, `scikit-learn`, `requests` (same as main script)

### `example_gene_query_scoring.py` (Integration Example)

**Purpose**: Shows how to use existing de_interpreter components for gene-query scoring.

**Usage**:
```bash
python scripts/example_gene_query_scoring.py
```

**Dependencies**: Requires full de_interpreter environment

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install sentence-transformers scikit-learn requests
   ```

2. **Run with sample genes**:
   ```bash
   python scripts/gene_query_similarity_scorer.py --genes TP53 BRCA1 MYC --query "cancer progression" --top-papers 5
   ```

3. **Check output**:
   - Console shows progress and results
   - Use `--output results.json` to save detailed results

## Input Formats

### Gene Lists
- **Command line**: `--genes GENE1 GENE2 GENE3`
- **File**: `--genes-file genes.txt` (one gene per line)

### Queries
- Simple: `"cancer progression"`
- Complex: `"inflammatory response pathway"`
- Disease-specific: `"COVID-19 ARDS"`

## Output

- **Console**: Progress, summary statistics, top papers
- **JSON file**: Detailed results with per-abstract scores, rankings, metadata

## Performance Notes

- **Rate limiting**: 1 second delay between gene searches (PubMed courtesy)
- **Model loading**: ~5-10 seconds for sentence-transformers model
- **Memory usage**: Scales with number of abstracts retrieved
- **Time**: ~30-60 seconds for 5 genes × 10 papers each
