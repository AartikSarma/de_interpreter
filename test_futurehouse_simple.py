"""Simple test for FutureHouse API integration."""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from de_interpreter.literature import FutureHouseClient


def test_futurehouse_simple():
    """Test FutureHouse API with a simple synchronous approach."""
    print("🔍 Testing FutureHouse API (simple test)...")
    
    try:
        # Create client (this is synchronous)
        client = FutureHouseClient()
        print("   ✅ Client created successfully")
        
        # Test simple search
        print("   📝 Running simple search...")
        
        # Use the direct client approach (synchronous)
        task_response = client.client.run_tasks_until_done([{
            "name": "job-futurehouse-paperqa2",  # CROW job
            "query": "What is TP53?"
        }])
        
        print(f"   ✅ Task completed, got {len(task_response)} responses")
        
        if task_response and task_response[0]:
            response = task_response[0]
            answer = getattr(response, 'answer', 'No answer')
            print(f"   📄 Answer length: {len(answer)} characters")
            
            # Show preview
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"   📖 Preview: {preview}")
            
            return True
        else:
            print("   ❌ No response received")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_futurehouse_simple()
    print("\n" + "="*50)
    if success:
        print("🎉 FutureHouse API test successful!")
    else:
        print("❌ FutureHouse API test failed")
    sys.exit(0 if success else 1)