"""Test the complete DE interpretation pipeline with mocked literature."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv
from unittest.mock import AsyncMock, patch
from datetime import datetime

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.main import DEInterpreter
from de_interpreter.literature import Paper, SearchResult


def create_mock_papers(gene: str, disease: str) -> list:
    """Create mock papers for testing."""
    return [
        Paper(
            paper_id=f"mock_{gene}_1",
            title=f"Role of {gene} in {disease} pathogenesis",
            abstract=f"This study investigates the role of {gene} in {disease}. We found that {gene} expression is altered in {disease} models.",
            authors=["Smith J", "Jones A", "Brown M"],
            year=2023,
            journal="Nature Genetics",
            relevance_score=0.95
        ),
        Paper(
            paper_id=f"mock_{gene}_2", 
            title=f"{gene} regulation and therapeutic implications in {disease}",
            abstract=f"Our research demonstrates that {gene} is a key regulator in {disease} progression and represents a potential therapeutic target.",
            authors=["Davis R", "Wilson K"],
            year=2022,
            journal="Cell",
            relevance_score=0.88
        )
    ]


async def mock_search_gene_disease(gene: str, disease: str, **kwargs):
    """Mock the gene-disease search function."""
    papers = create_mock_papers(gene, disease)
    return SearchResult(
        query=f"What is the role of {gene} in {disease}?",
        papers=papers,
        total_results=len(papers),
        search_time=datetime.now(),
        raw_answer=f"Mock literature summary for {gene} in {disease}"
    )


async def test_complete_pipeline():
    """Test the complete pipeline with mocked literature search."""
    print("🚀 Testing Complete DE Interpretation Pipeline")
    print("="*60)
    
    try:
        # Check if example files exist
        de_file = Path("examples/sample_de_results.csv")
        metadata_file = Path("examples/example_metadata.json")
        
        if not de_file.exists():
            print("❌ Sample DE results file not found")
            return False
        
        if not metadata_file.exists():
            print("❌ Sample metadata file not found")
            return False
        
        print("✅ Input files found")
        
        # Create interpreter with small parameters for testing
        interpreter = DEInterpreter(
            use_cache=False,  # Disable cache for testing
            top_n_genes=3,    # Only process 3 genes for speed
            n_clusters=2      # Create 2 clusters
        )
        
        print("✅ DE Interpreter created")
        
        # Mock the FutureHouse client search method
        with patch.object(
            interpreter._fetch_literature.__self__, 
            '_fetch_literature',
            new=AsyncMock()
        ) as mock_fetch:
            
            # Set up the mock to return our test data
            mock_fetch.return_value = {
                'MT-ND4': create_mock_papers('MT-ND4', "Parkinson's disease"),
                'TTN': create_mock_papers('TTN', "Parkinson's disease"),
                'E2F1': create_mock_papers('E2F1', "Parkinson's disease")
            }
            
            print("✅ Literature search mocked")
            
            # Run the analysis
            print("\n📊 Running DE analysis pipeline...")
            report_path = await interpreter.run(
                de_file=de_file,
                metadata_file=metadata_file,
                output_name="test_pipeline_report"
            )
            
            print(f"\n✅ Pipeline completed successfully!")
            print(f"📄 Report generated: {report_path}")
            
            # Verify the report was created
            if report_path.exists():
                file_size = report_path.stat().st_size
                print(f"📏 Report size: {file_size:,} bytes")
                
                # Read and show a preview of the report
                with open(report_path, 'r') as f:
                    content = f.read()
                
                print(f"📝 Report length: {len(content):,} characters")
                
                # Show first few lines
                lines = content.split('\n')[:15]
                preview = '\n'.join(lines)
                print(f"\n📖 Report Preview:\n{preview}...")
                
                # Check for key sections
                required_sections = [
                    "# Differential Expression Analysis Report",
                    "## Executive Summary", 
                    "## Analysis Overview",
                    "## Gene-by-Gene Analysis"
                ]
                
                missing_sections = []
                for section in required_sections:
                    if section not in content:
                        missing_sections.append(section)
                
                if missing_sections:
                    print(f"⚠️  Missing sections: {missing_sections}")
                else:
                    print("✅ All required report sections present")
                
                return True
            else:
                print("❌ Report file was not created")
                return False
                
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run the complete pipeline test."""
    success = await test_complete_pipeline()
    
    print("\n" + "="*60)
    if success:
        print("🎉 Complete pipeline test SUCCESSFUL!")
        print("✅ Your DE interpretation agent is fully functional!")
    else:
        print("❌ Pipeline test failed")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)