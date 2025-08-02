"""Streamlit web interface for DE Interpretation Pipeline."""

import streamlit as st
import asyncio
import sys
import tempfile
import json
from pathlib import Path
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import both traditional and omics pipelines
from de_interpreter.main import DEInterpreter
from de_interpreter.omics_main import OmicsInterpreter
from de_interpreter.parsers import DEParser, MetadataParser
from de_interpreter.parsers.omics_data import OmicsType
from de_interpreter.parsers.omics_parser import OmicsParser, OmicsMetadataParser


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DE Interpretation Pipeline",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 Multi-Omics Interpretation Pipeline")
    st.markdown("""
    Transform your differential omics results into literature-backed, 
    disease-contextualized discussions using AI-powered analysis.
    
    **Supported Omics Types**: Transcriptomics, Proteomics, Metabolomics, 
    Genomics, Metagenomics, Epigenomics, Lipidomics
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Status
        st.subheader("API Keys")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        futurehouse_key = os.getenv("FUTUREHOUSE_API_KEY")
        
        if anthropic_key:
            st.success("✅ Anthropic API key loaded")
        else:
            st.error("❌ Anthropic API key missing")
            st.info("Add ANTHROPIC_API_KEY to your .env file")
            
        if futurehouse_key:
            st.success("✅ FutureHouse API key loaded")
        else:
            st.error("❌ FutureHouse API key missing")
            st.info("Add FUTUREHOUSE_API_KEY to your .env file")
        
        st.divider()
        
        # Omics Type Selection
        st.subheader("🔬 Omics Type")
        omics_type_options = {
            "Transcriptomics (Gene Expression)": OmicsType.TRANSCRIPTOMICS,
            "Proteomics (Protein Abundance)": OmicsType.PROTEOMICS,
            "Metabolomics (Metabolite Levels)": OmicsType.METABOLOMICS,
            "Genomics (Genetic Variants)": OmicsType.GENOMICS,
            "Metagenomics (Microbial Communities)": OmicsType.METAGENOMICS,
            "Epigenomics (Chromatin Modifications)": OmicsType.EPIGENOMICS,
            "Lipidomics (Lipid Composition)": OmicsType.LIPIDOMICS,
        }
        
        selected_omics_display = st.selectbox(
            "Select your omics data type:",
            options=list(omics_type_options.keys()),
            index=0,
            help="Choose the type of omics data you're analyzing"
        )
        selected_omics_type = omics_type_options[selected_omics_display]
        
        # Display omics-specific info
        feature_names = {
            OmicsType.TRANSCRIPTOMICS: "genes",
            OmicsType.PROTEOMICS: "proteins",
            OmicsType.METABOLOMICS: "metabolites", 
            OmicsType.GENOMICS: "variants",
            OmicsType.METAGENOMICS: "microbes",
            OmicsType.EPIGENOMICS: "genomic regions",
            OmicsType.LIPIDOMICS: "lipids"
        }
        feature_name = feature_names[selected_omics_type]
        
        st.info(f"📊 Analyzing **{feature_name}** with {selected_omics_type.value}-specific methods")
        
        st.divider()
        
        # Pipeline Parameters
        st.subheader("Pipeline Parameters")
        top_n_features = st.slider(f"Top N {feature_name} to prioritize", 10, 200, 50, 
                                   help=f"Number of {feature_name} to prioritize from results")
        max_analysis_features = st.slider(f"Max {feature_name} for detailed analysis", 5, 50, 20,
                                         help=f"Number of top {feature_name} for literature mining and synthesis")
        use_cache = st.checkbox("Use literature cache", value=True)
        
        st.divider()
        
        # Example Data
        st.subheader("📁 Example Data")
        st.write("**Transcriptomics Examples:**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧠 Parkinson's Disease", help="Gene expression in dopaminergic neurons"):
                st.session_state.load_example = "parkinson"
                st.session_state.example_omics_type = OmicsType.TRANSCRIPTOMICS
        with col2:
            if st.button("🦠 COVID-19 ARDS", help="Gene expression in respiratory samples"):
                st.session_state.load_example = "covid"
                st.session_state.example_omics_type = OmicsType.TRANSCRIPTOMICS
        
        st.write("**Other Omics Examples:**")
        st.info("💡 Upload your own data files for proteomics, metabolomics, and other omics types!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input Data")
        
        # File uploaders
        st.subheader(f"1. Upload {selected_omics_type.value.title()} Results")
        
        # Omics-specific help text
        help_text = {
            OmicsType.TRANSCRIPTOMICS: "CSV/TSV with: gene_id, gene_symbol, log2FoldChange, pvalue, padj",
            OmicsType.PROTEOMICS: "CSV/TSV with: protein_id, protein_symbol, log2FoldChange, pvalue, padj",
            OmicsType.METABOLOMICS: "CSV/TSV with: metabolite_id, metabolite_name, log2FoldChange, pvalue, padj",
            OmicsType.GENOMICS: "CSV/TSV with: variant_id, gene_symbol, log2FoldChange, pvalue, padj",
            OmicsType.METAGENOMICS: "CSV/TSV with: species/otu_id, species_name, log2FoldChange, pvalue, padj",
            OmicsType.EPIGENOMICS: "CSV/TSV with: region_id, gene_symbol, log2FoldChange, pvalue, padj",
            OmicsType.LIPIDOMICS: "CSV/TSV with: lipid_id, lipid_name, log2FoldChange, pvalue, padj"
        }
        
        data_file = st.file_uploader(
            f"Differential {selected_omics_type.value.title()} Results",
            type=["csv", "tsv", "xlsx"],
            help=help_text[selected_omics_type]
        )
        
        if data_file:
            try:
                # Preview the uploaded file
                if data_file.name.endswith('.csv'):
                    df = pd.read_csv(data_file)
                elif data_file.name.endswith('.tsv'):
                    df = pd.read_csv(data_file, sep='\t')
                else:
                    df = pd.read_excel(data_file)
                
                st.write("**File Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                st.info(f"Loaded {len(df)} {feature_name}")
                
                # Reset file pointer for later use
                data_file.seek(0)
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        st.subheader("2. Experimental Metadata")
        
        # Option to upload metadata file or enter manually
        metadata_option = st.radio(
            "Metadata input method:",
            ["Upload JSON file", "Enter manually"]
        )
        
        metadata = None
        
        if metadata_option == "Upload JSON file":
            metadata_file = st.file_uploader(
                "Metadata JSON",
                type=["json"],
                help="JSON file with experimental context"
            )
            
            if metadata_file:
                try:
                    metadata = json.load(metadata_file)
                    st.write("**Metadata Preview:**")
                    st.json(metadata)
                except Exception as e:
                    st.error(f"Error reading metadata: {e}")
        
        else:
            # Manual metadata entry
            with st.form("metadata_form"):
                # Basic experimental info
                disease = st.text_input("Disease/Condition", placeholder="e.g., Type 2 Diabetes")
                tissue = st.text_input("Tissue/Sample Type", placeholder="e.g., plasma, liver tissue")
                cell_type = st.text_input("Cell Type", placeholder="e.g., hepatocytes, plasma")
                treatment = st.text_input("Treatment", placeholder="e.g., high glucose diet")
                control = st.text_input("Control", placeholder="e.g., normal diet")
                organism = st.selectbox("Organism", ["human", "mouse", "rat", "other"])
                
                # Omics-specific fields
                st.subheader("Omics-Specific Information")
                platform = st.text_input("Platform/Instrument", 
                                        placeholder="e.g., Orbitrap MS, RNA-seq, etc.")
                analysis_method = st.text_input("Analysis Method", 
                                              placeholder="e.g., MetaboAnalyst, DESeq2, etc.")
                normalization = st.text_input("Normalization Method", 
                                             placeholder="e.g., TMM, quantile, etc.")
                
                if st.form_submit_button("Create Metadata"):
                    metadata = {
                        "omics_type": selected_omics_type.value,
                        "disease": disease,
                        "tissue": tissue,
                        "cell_type": cell_type,
                        "treatment": treatment,
                        "control": control,
                        "organism": organism,
                        "platform": platform,
                        "analysis_method": analysis_method,
                        "normalization": normalization,
                        "comparison_description": f"{treatment} vs {control} in {organism} {tissue}"
                    }
                    st.success("Metadata created!")
                    st.json(metadata)
    
    with col2:
        st.header("🔬 Analysis")
        
        # Handle example data loading
        if hasattr(st.session_state, 'load_example'):
            example = st.session_state.load_example
            example_omics_type = getattr(st.session_state, 'example_omics_type', OmicsType.TRANSCRIPTOMICS)
            
            if example == "parkinson":
                data_file_path = "examples/sample_de_results.csv"
                metadata_file_path = "examples/example_metadata.json"
                st.info("🧠 Loaded Parkinson's disease transcriptomics data")
                
            elif example == "covid":
                data_file_path = "covid_data/covid_deg_fixed.csv"
                metadata_file_path = "covid_data/covid_metadata.json"
                st.info("🦠 Loaded COVID-19 ARDS transcriptomics data")
            
            # Load example files for analysis
            if Path(data_file_path).exists() and Path(metadata_file_path).exists():
                with open(metadata_file_path) as f:
                    metadata = json.load(f)
                
                # Ensure omics type is set in metadata
                if "omics_type" not in metadata:
                    metadata["omics_type"] = example_omics_type.value
                
                # Show loaded data
                df = pd.read_csv(data_file_path)
                st.write("**Loaded Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                st.write("**Metadata:**")
                st.json(metadata)
                
                # Store for pipeline execution
                st.session_state.example_data_path = data_file_path
                st.session_state.example_metadata_path = metadata_file_path
                st.session_state.example_metadata = metadata
            
            # Clear the load example flag but keep paths
            del st.session_state.load_example
        
        # Analysis button
        has_data = (data_file is not None or hasattr(st.session_state, 'example_data_path'))
        has_metadata = (metadata is not None or hasattr(st.session_state, 'example_metadata'))
        can_run = has_data and has_metadata
        
        if not (anthropic_key and futurehouse_key):
            st.error("⚠️ API keys required to run analysis")
            can_run = False
        
        if st.button("🚀 Run Analysis", disabled=not can_run, type="primary"):
            run_omics_analysis(
                data_file, 
                metadata, 
                selected_omics_type,
                top_n_features, 
                max_analysis_features, 
                use_cache,
                feature_name
            )


def run_omics_analysis(data_file, metadata, omics_type, top_n_features, max_analysis_features, use_cache, feature_name):
    """Run the multi-omics interpretation analysis."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded files to temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle data file
            if data_file is not None:
                data_path = temp_path / f"omics_results.csv"
                with open(data_path, "wb") as f:
                    f.write(data_file.getbuffer())
            else:
                # Use example data
                if hasattr(st.session_state, 'example_data_path'):
                    data_path = Path(st.session_state.example_data_path)
                else:
                    st.error(f"No {feature_name} data file available")
                    return
            
            # Handle metadata
            if metadata is not None:
                metadata_path = temp_path / "metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Use example metadata
                if hasattr(st.session_state, 'example_metadata_path'):
                    metadata_path = Path(st.session_state.example_metadata_path)
                else:
                    st.error("No metadata available")
                    return
            
            progress_bar.progress(10)
            status_text.text("Initializing omics pipeline...")
            
            # Create omics interpreter
            interpreter = OmicsInterpreter(
                omics_type=omics_type,
                use_cache=use_cache,
                top_n_features=top_n_features,
                max_analysis_features=max_analysis_features
            )
            
            progress_bar.progress(20)
            status_text.text(f"Running {omics_type.value} analysis...")
            
            # Run analysis in async context
            async def run_pipeline():
                return await interpreter.run(
                    data_file=data_path,
                    metadata_file=metadata_path,
                    output_name="streamlit_omics_analysis"
                )
            
            # Run the pipeline
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                report_path = loop.run_until_complete(run_pipeline())
                progress_bar.progress(100)
                status_text.text(f"{omics_type.value.title()} analysis complete!")
                
                # Display results
                display_omics_results(report_path, omics_type, feature_name)
                
            finally:
                loop.close()
                
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        progress_bar.progress(0)
        status_text.text("Analysis failed")


def display_omics_results(report_path: Path, omics_type: OmicsType, feature_name: str):
    """Display the omics analysis results."""
    st.header(f"📊 {omics_type.value.title()} Analysis Results")
    
    if report_path.exists():
        # Read the report
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Display download button
        st.download_button(
            label=f"📥 Download {omics_type.value.title()} Report",
            data=report_content,
            file_name=f"{report_path.stem}.md",
            mime="text/markdown"
        )
        
        # Display report preview
        st.subheader(f"{omics_type.value.title()} Report Preview")
        
        # Split content into sections for better display
        sections = report_content.split('\n## ')
        
        if len(sections) > 1:
            # Show executive summary
            exec_summary_start = report_content.find('# Executive Summary')
            if exec_summary_start != -1:
                exec_summary_end = report_content.find('\n## ', exec_summary_start)
                if exec_summary_end == -1:
                    exec_summary_end = report_content.find('\n# ', exec_summary_start + 1)
                
                if exec_summary_end != -1:
                    exec_summary = report_content[exec_summary_start:exec_summary_end]
                    st.markdown(exec_summary)
                    
                    # Show expandable sections for the rest
                    with st.expander("📖 View Full Report"):
                        st.markdown(report_content)
                else:
                    st.markdown(report_content[:2000] + "..." if len(report_content) > 2000 else report_content)
            else:
                st.markdown(report_content[:2000] + "..." if len(report_content) > 2000 else report_content)
        else:
            st.markdown(report_content)
        
        # Display metadata file if exists
        metadata_path = report_path.parent / f"{report_path.stem}_metadata.json"
        if metadata_path.exists():
            with st.expander("📋 Analysis Metadata"):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                st.json(metadata)
    
    else:
        st.error("Report file not found")


if __name__ == "__main__":
    main()