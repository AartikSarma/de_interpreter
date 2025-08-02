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

from de_interpreter.main import DEInterpreter
from de_interpreter.parsers import DEParser, MetadataParser


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="DE Interpretation Pipeline",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧬 Differential Expression Interpretation Pipeline")
    st.markdown("""
    Transform your differential expression results into literature-backed, 
    disease-contextualized discussions using AI-powered analysis.
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
        
        # Pipeline Parameters
        st.subheader("Pipeline Parameters")
        top_n_genes = st.slider("Top N genes to analyze", 5, 100, 25)
        n_clusters = st.selectbox(
            "Number of gene clusters", 
            options=["Auto"] + list(range(2, 11)),
            index=0
        )
        use_cache = st.checkbox("Use literature cache", value=True)
        
        st.divider()
        
        # Example Data
        st.subheader("📁 Example Data")
        if st.button("Load Parkinson's Example"):
            st.session_state.load_example = "parkinson"
        if st.button("Load COVID-19 Example"):
            st.session_state.load_example = "covid"
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input Data")
        
        # File uploaders
        st.subheader("1. Upload DE Results")
        de_file = st.file_uploader(
            "Differential Expression Results",
            type=["csv", "tsv", "xlsx"],
            help="CSV/TSV file with columns: gene_id, gene_symbol, log2FoldChange, pvalue, padj"
        )
        
        if de_file:
            try:
                # Preview the uploaded file
                if de_file.name.endswith('.csv'):
                    df = pd.read_csv(de_file)
                elif de_file.name.endswith('.tsv'):
                    df = pd.read_csv(de_file, sep='\t')
                else:
                    df = pd.read_excel(de_file)
                
                st.write("**File Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                st.info(f"Loaded {len(df)} genes")
                
                # Reset file pointer for later use
                de_file.seek(0)
                
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
                disease = st.text_input("Disease/Condition", placeholder="e.g., Parkinson's disease")
                tissue = st.text_input("Tissue/Sample Type", placeholder="e.g., substantia nigra")
                cell_type = st.text_input("Cell Type", placeholder="e.g., dopaminergic neurons")
                treatment = st.text_input("Treatment", placeholder="e.g., alpha-synuclein fibrils")
                control = st.text_input("Control", placeholder="e.g., PBS")
                organism = st.selectbox("Organism", ["human", "mouse", "rat", "other"])
                
                if st.form_submit_button("Create Metadata"):
                    metadata = {
                        "disease": disease,
                        "tissue": tissue,
                        "cell_type": cell_type,
                        "treatment": treatment,
                        "control": control,
                        "organism": organism,
                        "comparison_description": f"{treatment} vs {control} in {organism} {cell_type}"
                    }
                    st.success("Metadata created!")
                    st.json(metadata)
    
    with col2:
        st.header("🔬 Analysis")
        
        # Handle example data loading
        if hasattr(st.session_state, 'load_example'):
            example = st.session_state.load_example
            
            if example == "parkinson":
                de_file_path = "examples/sample_de_results.csv"
                metadata_file_path = "examples/example_metadata.json"
                st.info("Loaded Parkinson's disease example data")
                
            elif example == "covid":
                de_file_path = "covid_data/covid_deg_fixed.csv"
                metadata_file_path = "covid_data/covid_metadata.json"
                st.info("Loaded COVID-19 ARDS example data")
            
            # Load example files for analysis
            if Path(de_file_path).exists() and Path(metadata_file_path).exists():
                with open(metadata_file_path) as f:
                    metadata = json.load(f)
                
                # Show loaded data
                df = pd.read_csv(de_file_path)
                st.write("**Loaded Data Preview:**")
                st.dataframe(df.head(), use_container_width=True)
                st.write("**Metadata:**")
                st.json(metadata)
            
            # Clear the session state
            del st.session_state.load_example
        
        # Analysis button
        can_run = (de_file is not None or hasattr(st.session_state, 'load_example')) and metadata is not None
        
        if not (anthropic_key and futurehouse_key):
            st.error("⚠️ API keys required to run analysis")
            can_run = False
        
        if st.button("🚀 Run Analysis", disabled=not can_run, type="primary"):
            run_analysis(de_file, metadata, top_n_genes, n_clusters, use_cache)


def run_analysis(de_file, metadata, top_n_genes, n_clusters, use_cache):
    """Run the DE interpretation analysis."""
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded files to temporary location
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Handle DE file
            if de_file is not None:
                de_path = temp_path / "de_results.csv"
                with open(de_path, "wb") as f:
                    f.write(de_file.getbuffer())
            else:
                # Use example data
                if hasattr(st.session_state, 'example_de_path'):
                    de_path = Path(st.session_state.example_de_path)
                else:
                    st.error("No DE file available")
                    return
            
            # Save metadata
            metadata_path = temp_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            progress_bar.progress(10)
            status_text.text("Initializing pipeline...")
            
            # Create interpreter
            interpreter = DEInterpreter(
                use_cache=use_cache,
                top_n_genes=top_n_genes,
                n_clusters=None if n_clusters == "Auto" else int(n_clusters)
            )
            
            progress_bar.progress(20)
            status_text.text("Running analysis...")
            
            # Run analysis in async context
            async def run_pipeline():
                return await interpreter.run(
                    de_file=de_path,
                    metadata_file=metadata_path,
                    output_name="streamlit_analysis"
                )
            
            # Run the pipeline
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                report_path = loop.run_until_complete(run_pipeline())
                progress_bar.progress(100)
                status_text.text("Analysis complete!")
                
                # Display results
                display_results(report_path)
                
            finally:
                loop.close()
                
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        progress_bar.progress(0)
        status_text.text("Analysis failed")


def display_results(report_path: Path):
    """Display the analysis results."""
    st.header("📊 Results")
    
    if report_path.exists():
        # Read the report
        with open(report_path, 'r') as f:
            report_content = f.read()
        
        # Display download button
        st.download_button(
            label="📥 Download Full Report",
            data=report_content,
            file_name=f"{report_path.stem}.md",
            mime="text/markdown"
        )
        
        # Display report preview
        st.subheader("Report Preview")
        
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