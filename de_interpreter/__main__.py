"""Unified entry point for the DE Interpreter package.

This module provides a single entry point for both CLI and programmatic usage.
It combines all functionality with smart argument parsing and auto-detection.

Usage:
    python -m de_interpreter [args]
    python -m de_interpreter --web  # Launch Streamlit interface
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional

def main():
    """Main entry point for the DE Interpreter package."""
    
    # Check if web interface is requested
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        launch_web_interface()
        return
    
    # Import and run the CLI main function
    from .main import main as cli_main
    asyncio.run(cli_main())


def launch_web_interface():
    """Launch the Streamlit web interface."""
    import subprocess
    
    app_path = Path(__file__).parent.parent / "streamlit_app.py"
    
    if not app_path.exists():
        print("❌ Error: streamlit_app.py not found")
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