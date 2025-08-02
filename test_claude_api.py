"""Test the DE interpreter with Claude API calls only."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.synthesis import ClaudeSynthesizer
from de_interpreter.parsers import DEResult, ExperimentalContext
from de_interpreter.prioritization import PrioritizedGene


async def test_claude_api():
    """Test Claude API integration."""
    print("🤖 Testing Claude API...")
    
    try:
        # Create test data
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
        
        # Test Claude synthesis (fix async context manager)
        synthesizer = ClaudeSynthesizer()
        
        try:
            # Test gene discussion generation
            discussion = await synthesizer.synthesize_gene_discussion(
                prioritized_gene, context, []  # Empty papers list for quick test
            )
            
            print(f"   ✅ Generated discussion for {discussion.gene_symbol}")
            print(f"   📝 Discussion length: {len(discussion.discussion_text)} characters")
            print(f"   🎯 Confidence score: {discussion.confidence_score:.2f}")
            print(f"   💡 Key findings: {len(discussion.key_findings)}")
            
            # Print first 200 chars of discussion
            preview = discussion.discussion_text[:200] + "..." if len(discussion.discussion_text) > 200 else discussion.discussion_text
            print(f"   📄 Preview: {preview}")
            
            # Test if the discussion contains expected content
            if discussion.gene_symbol and discussion.discussion_text:
                print("   ✅ Discussion structure is valid")
                return True
            else:
                print("   ❌ Discussion structure is invalid")
                return False
                
        except Exception as e:
            print(f"   ❌ Claude API synthesis error: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Claude API setup error: {e}")
        return False


async def test_claude_executive_summary():
    """Test Claude executive summary generation."""
    print("📊 Testing Claude executive summary...")
    
    try:
        from de_interpreter.synthesis import GeneDiscussion
        
        # Create mock discussions
        discussions = [
            GeneDiscussion(
                gene_id="ENSG001",
                gene_symbol="TP53",
                discussion_text="TP53 is a tumor suppressor gene that plays a critical role in cancer.",
                key_findings=["Upregulated in response to DNA damage"],
                therapeutic_implications="Potential target for cancer therapy",
                citations=["Smith et al. (2023)"],
                confidence_score=0.9
            ),
            GeneDiscussion(
                gene_id="ENSG002", 
                gene_symbol="BRCA1",
                discussion_text="BRCA1 is involved in DNA repair mechanisms.",
                key_findings=["Associated with breast cancer risk"],
                therapeutic_implications=None,
                citations=["Jones et al. (2023)"],
                confidence_score=0.8
            )
        ]
        
        context = ExperimentalContext(
            disease="breast cancer",
            tissue="mammary tissue",
            treatment="radiation",
            control="sham"
        )
        
        de_summary = {
            'total_genes': 1000,
            'significant_genes': 150,
            'upregulated': 75,
            'downregulated': 75
        }
        
        synthesizer = ClaudeSynthesizer()
        
        try:
            summary = await synthesizer.generate_executive_summary(
                discussions, context, de_summary
            )
            
            print(f"   ✅ Generated executive summary")
            print(f"   📝 Summary length: {len(summary)} characters")
            
            # Check if summary contains key elements
            if "breast cancer" in summary.lower() and len(summary) > 100:
                print("   ✅ Executive summary contains expected content")
                
                # Print first 300 chars
                preview = summary[:300] + "..." if len(summary) > 300 else summary
                print(f"   📄 Summary preview: {preview}")
                return True
            else:
                print("   ❌ Executive summary is too short or missing key content")
                return False
                
        except Exception as e:
            print(f"   ❌ Executive summary generation error: {e}")
            return False
            
    except Exception as e:
        print(f"   ❌ Executive summary setup error: {e}")
        return False


async def main():
    """Run Claude API tests."""
    print("🧪 Testing Claude API Integration\n" + "="*50)
    
    results = []
    
    # Test Claude gene discussion
    results.append(await test_claude_api())
    print()
    
    # Test Claude executive summary
    results.append(await test_claude_executive_summary())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Claude API tests passed!")
        print("✅ The DE interpreter's AI synthesis functionality is working correctly!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)