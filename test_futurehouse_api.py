"""Test the updated FutureHouse API integration."""

import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.literature import FutureHouseClient


async def test_futurehouse_api():
    """Test FutureHouse API integration."""
    print("🔍 Testing FutureHouse API...")
    
    try:
        async with FutureHouseClient() as client:
            # Test basic search
            print("   📝 Testing basic search...")
            result = await client.search("What is the role of TP53 in cancer?", limit=3)
            
            print(f"   ✅ Query: {result.query}")
            print(f"   📄 Raw answer length: {len(result.raw_answer)} characters")
            print(f"   📚 Extracted papers: {len(result.papers)}")
            
            # Show first part of the response
            if result.raw_answer:
                preview = result.raw_answer[:300] + "..." if len(result.raw_answer) > 300 else result.raw_answer
                print(f"   📖 Response preview: {preview}")
            
            # Show extracted papers
            for i, paper in enumerate(result.papers[:3]):
                print(f"   📄 Paper {i+1}: {paper.title}")
                print(f"      Authors: {', '.join(paper.authors)}")
                print(f"      Year: {paper.year}")
            
            print("\n   📝 Testing gene-disease search...")
            
            # Test gene-disease search
            result2 = await client.search_gene_disease("BRCA1", "breast cancer", limit=2)
            print(f"   ✅ Gene-disease query: {result2.query}")
            print(f"   📚 Papers found: {len(result2.papers)}")
            
            if result2.papers:
                print(f"   📄 First paper: {result2.papers[0].title}")
            
            return True
            
    except Exception as e:
        print(f"   ❌ FutureHouse API error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_futurehouse_batch():
    """Test batch search functionality."""
    print("🔄 Testing FutureHouse batch search...")
    
    try:
        async with FutureHouseClient() as client:
            queries = [
                "What is the role of TP53 in cancer?",
                "What is the role of BRCA1 in breast cancer?"
            ]
            
            print(f"   📝 Running {len(queries)} searches...")
            results = await client.batch_search(queries, limit_per_query=2)
            
            print(f"   ✅ Completed {len(results)} searches")
            
            for i, result in enumerate(results):
                print(f"   📄 Result {i+1}: {len(result.papers)} papers for '{result.query}'")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Batch search error: {e}")
        return False


async def main():
    """Run FutureHouse API tests."""
    print("🧪 Testing FutureHouse API Integration\n" + "="*50)
    
    results = []
    
    # Test basic API functionality
    results.append(await test_futurehouse_api())
    print()
    
    # Test batch functionality
    results.append(await test_futurehouse_batch())
    print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("="*50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All FutureHouse API tests passed!")
        print("✅ The literature search functionality is working!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)