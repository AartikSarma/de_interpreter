"""Test the DE interpreter with actual API calls."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.main import DEInterpreter
from de_interpreter.literature import FutureHouseClient
from de_interpreter.synthesis import ClaudeSynthesizer
from de_interpreter.parsers import DEParser, MetadataParser


async def test_literature_api():
    """Test FutureHouse API integration."""
    print("🔍 Testing FutureHouse API...")
    
    try:
        async with FutureHouseClient() as client:
            # Test basic search
            result = await client.search("TP53 cancer", limit=3)
            print(f"   ✅ Found {len(result.papers)} papers for TP53 cancer")
            
            if result.papers:
                paper = result.papers[0]
                print(f"   📄 First paper: {paper.title[:80]}...")
                print(f"   👥 Authors: {', '.join(paper.authors[:2])}{'...' if len(paper.authors) > 2 else ''}")
                print(f"   📅 Year: {paper.year}")
            
            # Test gene-disease search
            result2 = await client.search_gene_disease("BRCA1", "breast cancer", limit=2)
            print(f"   ✅ Found {len(result2.papers)} papers for BRCA1 in breast cancer")
            
    except Exception as e:
        print(f"   ❌ FutureHouse API error: {e}")
        return False
    
    return True


async def test_claude_api():
    """Test Claude API integration."""
    print("🤖 Testing Claude API...")
    
    try:
        # Create a simple test gene for synthesis
        from de_interpreter.parsers import DEResult
        from de_interpreter.prioritization import PrioritizedGene
        from de_interpreter.parsers import ExperimentalContext
        
        # Mock data
        de_result = DEResult(
            gene_id="ENSG00000141510",
            gene_symbol="TP53",
            log2_fold_change=2.1,
            p_value=0.001,
            padj=0.01,
            base_mean=450.2
        )
        
        prioritized_gene = PrioritizedGene(
            de_result=de_result,
            statistical_score=0.8,
            biological_score=0.7,
            combined_score=0.76,
            rank=1
        )
        
        context = ExperimentalContext(
            disease="cancer",
            tissue="tumor",
            treatment="chemotherapy",
            control="untreated"
        )
        
        # Test Claude synthesis
        async with ClaudeSynthesizer() as synthesizer:
            # Test with minimal papers (empty list for quick test)
            discussion = await synthesizer.synthesize_gene_discussion(
                prioritized_gene, context, []
            )
            
            print(f"   ✅ Generated discussion for {discussion.gene_symbol}")
            print(f"   📝 Discussion length: {len(discussion.discussion_text)} characters")
            print(f"   🎯 Confidence score: {discussion.confidence_score:.2f}")
            print(f"   💡 Key findings: {len(discussion.key_findings)}")
            
            # Print first 150 chars of discussion
            preview = discussion.discussion_text[:150] + "..." if len(discussion.discussion_text) > 150 else discussion.discussion_text
            print(f"   📄 Preview: {preview}")
            
    except Exception as e:
        print(f"   ❌ Claude API error: {e}")
        return False
    
    return True


async def test_full_pipeline():
    """Test the complete pipeline with API calls."""
    print("🚀 Testing full pipeline with API calls...")
    
    try:
        # Use the sample data we created
        de_file = Path("examples/sample_de_results.csv")
        metadata_file = Path("examples/example_metadata.json")
        
        if not de_file.exists() or not metadata_file.exists():
            print("   ⚠️  Sample files not found, skipping full pipeline test")
            return True
        
        # Create interpreter with minimal settings for quick test
        interpreter = DEInterpreter(
            use_cache=True,
            top_n_genes=3,  # Only test top 3 genes
            n_clusters=2
        )
        
        print("   📊 Running analysis on sample data...")
        
        # Run the analysis
        report_path = await interpreter.run(
            de_file=de_file,
            metadata_file=metadata_file,
            output_name="api_test_analysis"
        )
        
        print(f"   ✅ Analysis completed successfully!")
        print(f"   📄 Report saved to: {report_path}")
        
        # Check if report was created
        if report_path.exists():
            file_size = report_path.stat().st_size
            print(f"   📏 Report size: {file_size:,} bytes")
            
            # Read first few lines of report
            with open(report_path, 'r') as f:
                first_lines = f.read(300)
            print(f"   📖 Report preview:\n{first_lines}...")
        else:
            print("   ❌ Report file was not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


async def main():
    """Run all API integration tests."""
    print("🧪 Starting API Integration Tests\n" + "="*50)
    
    results = []
    
    # Test individual APIs
    results.append(await test_literature_api())
    print()
    
    results.append(await test_claude_api())
    print()
    
    # Test full pipeline
    results.append(await test_full_pipeline())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All API integration tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)