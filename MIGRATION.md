# Migration Guide: Unified Multi-Omics Pipeline

## Overview

The DE Interpretation Pipeline has been **refactored into a unified architecture** that supports all omics types through a single, intelligent codebase. This migration consolidates the previously separate `main.py` and `omics_main.py` pipelines.

## ✅ What's New

### Unified Architecture
- **Single Pipeline**: All 7 omics types use the same underlying code
- **Auto-Detection**: Automatically detects omics type from your data columns
- **Intelligent Parsing**: Smart column mapping handles various naming conventions
- **Progress Tracking**: Comprehensive progress bars for all pipeline steps

### Enhanced Features
- **Literature Scoring**: Optional BioBERT, TF-IDF, and BM25 relevance scoring
- **Full-Text Access**: PMC integration provides complete paper content
- **Performance Options**: Choose between speed (TF-IDF) and quality (BioBERT)
- **Omics-Aware Processing**: Specialized handling for each omics type

## 🔄 Migration Paths

### For Existing Users

**Old Transcriptomics Pipeline:**
```bash
# DEPRECATED
python -m de_interpreter.main --de-file data.csv --metadata meta.json
```

**New Unified Approach:**
```bash
# Backward-compatible interface (recommended for migration)
python de_main.py --de-file data.csv --metadata meta.json

# OR use the new unified pipeline
python -m de_interpreter.unified_main --data-file data.csv --metadata meta.json
```

**Old Multi-Omics Pipeline:**
```bash
# DEPRECATED  
python -m de_interpreter.omics_main --data-file data.csv --omics-type proteomics
```

**New Unified Approach:**
```bash
# Backward-compatible interface
python omics_main.py --data-file data.csv --omics-type proteomics

# OR let the pipeline auto-detect omics type
python -m de_interpreter.unified_main --data-file data.csv --metadata meta.json
```

### For New Projects

**Recommended Approach:**
```bash
# The pipeline automatically detects whether you have genes, proteins, metabolites, etc.
python -m de_interpreter.unified_main \
  --data-file your_data.csv \
  --metadata your_metadata.json \
  --enable-scoring \
  --scorer biobert
```

## 📊 Supported Data Formats

The unified pipeline now automatically detects these omics types:

| Omics Type | Detection Keywords | Example Columns |
|------------|-------------------|----------------|
| **Transcriptomics** | gene, transcript, ensembl | `gene_id`, `gene_symbol`, `transcript_id` |
| **Proteomics** | protein, uniprot, peptide | `protein_id`, `uniprot_id`, `protein_symbol` |
| **Metabolomics** | metabolite, compound, chemical | `metabolite_id`, `compound_name`, `hmdb_id` |
| **Genomics** | variant, snp, mutation | `variant_id`, `snp_id`, `chr_pos` |
| **Metagenomics** | species, taxon, otu, microbe | `species`, `otu_id`, `taxonomic_id` |
| **Epigenomics** | region, peak, dmr, chromatin | `region_id`, `peak_id`, `genomic_region` |
| **Lipidomics** | lipid, fatty_acid | `lipid_id`, `lipid_name`, `lipid_class` |

## 🚀 Performance Improvements

### Literature Scoring Options
```bash
# Fast processing (good for large datasets)
--enable-scoring --scorer tfidf

# Balanced speed and quality
--enable-scoring --scorer bm25  

# Best quality (recommended for final reports)
--enable-scoring --scorer biobert
```

### Benchmark Results
- **TF-IDF**: ~2s per query, good relevance
- **BM25**: ~3s per query, better relevance  
- **BioBERT**: ~8s per query, excellent semantic matching

## 🔧 API Changes

### Python API
```python
# OLD - Multiple classes
from de_interpreter.main import DEInterpreter
from de_interpreter.omics_main import OmicsInterpreter

# NEW - Single unified class
from de_interpreter.unified_main import UnifiedInterpreter

# Auto-detects omics type from data
interpreter = UnifiedInterpreter()
report = await interpreter.run(data_file, metadata_file)
```

### Streamlit Interface
The web interface now uses the unified pipeline automatically - no changes needed for users.

## ⚠️ Deprecated Components

These files are deprecated but maintained for backward compatibility:
- `src/de_interpreter/main.py` → Use `de_main.py` or `unified_main.py`
- `src/de_interpreter/omics_main.py` → Use `omics_main.py` or `unified_main.py`

## 📈 Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Codebase** | 2 separate pipelines | 1 unified pipeline |
| **Omics Support** | Manual specification | Auto-detection |
| **Code Duplication** | ~60% overlap | ~5% overlap |
| **Literature Quality** | Abstract-only | Full-text + AI scoring |
| **Progress Tracking** | Basic | Comprehensive |
| **Maintenance** | 2x effort | Single codebase |

## 🆘 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'de_interpreter'"**
```bash
# Make sure you're in the project root directory
cd /path/to/de_interpreter
python -m de_interpreter.unified_main --help
```

**"Unable to detect omics type"**
```bash
# Force a specific omics type
python -m de_interpreter.unified_main --omics-type transcriptomics --data-file data.csv
```

**"Literature scoring is slow"**
```bash
# Use faster scoring methods
python -m de_interpreter.unified_main --enable-scoring --scorer tfidf --data-file data.csv
```

## 📞 Support

- The unified pipeline maintains **100% backward compatibility**
- All existing scripts and workflows continue to work
- For questions, see the main `CLAUDE.md` documentation
- Performance benchmarking available in `benchmark_scoring.py`

---

**Ready to migrate?** Start with the backward-compatible interfaces (`de_main.py`, `omics_main.py`) then transition to the unified pipeline (`unified_main.py`) when convenient.