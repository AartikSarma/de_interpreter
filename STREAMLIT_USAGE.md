# 🧬 Multi-Omics Interpretation Pipeline - Web Interface

This Streamlit application provides an easy-to-use web interface for the Multi-Omics Interpretation Pipeline, supporting transcriptomics, proteomics, metabolomics, genomics, metagenomics, epigenomics, and lipidomics data.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys
Create a `.env` file in the project root:
```bash
ANTHROPIC_API_KEY=your_claude_api_key
FUTUREHOUSE_API_KEY=your_futurehouse_api_key
```

### 3. Run the App
```bash
# Option 1: Use the runner script
python run_streamlit.py

# Option 2: Run directly
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Features

### Multi-Omics Support
- **Transcriptomics**: Gene expression analysis
- **Proteomics**: Protein abundance analysis  
- **Metabolomics**: Metabolite profiling
- **Genomics**: Genetic variation analysis
- **Metagenomics**: Microbial community analysis
- **Epigenomics**: Chromatin modification analysis
- **Lipidomics**: Lipid composition analysis

### Input Methods
- **File Upload**: Upload your own omics results (CSV/TSV/Excel) and metadata (JSON)
- **Manual Entry**: Enter experimental metadata through web forms with omics-specific fields
- **Example Data**: Load pre-configured COVID-19 ARDS transcriptomics dataset

### Analysis Parameters
- **Omics Type Selection**: Choose your data type for specialized analysis
- **Top N features to prioritize**: Select how many features to prioritize (10-200)
- **Max features for detailed analysis**: Number of top features for literature mining (5-50)
- **Caching**: Enable/disable literature result caching

### Real-time Progress
- Progress bar showing analysis steps
- Status updates during processing
- Error handling with clear messages

### Results Display
- **Executive Summary**: Key findings and biological themes
- **Full Report**: Downloadable Markdown report
- **Interactive Preview**: Expandable sections for easy reading
- **Metadata**: Analysis parameters and statistics

## 📁 Input File Formats

### DE Results File
CSV/TSV/Excel with columns:
- `gene_id` or `ENSEMBL ID`: Gene identifiers
- `gene_symbol` or `Name`: Gene symbols
- `log2FoldChange`: Log2 fold change values
- `pvalue` or `p value`: P-values
- `padj` or `p adj`: Adjusted p-values

### Metadata File (JSON)
```json
{
  "disease": "Disease or condition name",
  "tissue": "Tissue or sample type",
  "cell_type": "Cell type",
  "treatment": "Treatment condition",
  "control": "Control condition",
  "organism": "human/mouse/rat",
  "comparison_description": "Brief description"
}
```

## 🔧 Troubleshooting

### Common Issues

**"API keys missing"**
- Ensure `.env` file exists with valid API keys
- Check that environment variables are loaded correctly

**"Error reading file"**
- Verify file format is CSV, TSV, or Excel
- Check that required columns are present
- Ensure column names match expected format

**"Analysis failed"**
- Check internet connection for API calls
- Verify API keys are valid and have sufficient credits
- Try reducing the number of genes to analyze

**"Pipeline timeout"**
- Large datasets may take longer to process
- Consider reducing `top_n_genes` parameter
- Check API rate limits

### File Format Issues
If your DE results have different column names, you can:
1. Rename columns to match expected format
2. Use the automatic column mapping (supports common variations)
3. Check the example files for reference

## 📊 Example Datasets

### COVID-19 ARDS
- **Context**: COVID-19 ARDS vs Non-COVID ARDS
- **Genes**: 793 genes from tracheal aspirate samples
- **Focus**: Respiratory dysfunction, immune response

## 🎯 Tips for Best Results

1. **Quality Input Data**: Ensure your DE results include proper statistical testing (p-values, adjusted p-values)

2. **Relevant Metadata**: Provide detailed experimental context for better literature mining

3. **Appropriate Gene Count**: Start with 10-20 genes for detailed analysis, prioritize more for comprehensive overview

4. **Caching**: Enable caching to avoid re-fetching literature for repeated analyses

5. **API Limits**: Be mindful of API rate limits when analyzing large gene sets

## 🆘 Support

If you encounter issues:
1. Check the console/terminal for error messages
2. Verify your input file formats match the requirements
3. Ensure API keys are properly configured
4. Try with example datasets first to confirm setup

For technical issues, check the main project documentation in `CLAUDE.md`.