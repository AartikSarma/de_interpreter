#!/usr/bin/env python3
"""Simple script to run the Streamlit app."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    app_path = Path(__file__).parent / "streamlit_app.py"
    
    if not app_path.exists():
        print("Error: streamlit_app.py not found")
        sys.exit(1)
    
    print("🚀 Starting DE Interpretation Pipeline Web Interface...")
    print("📱 The app will open in your default browser")
    print("🛑 Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the application...")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()