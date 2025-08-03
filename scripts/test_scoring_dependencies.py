#!/usr/bin/env python3
"""
Quick check to see if all dependencies are installed for the de_interpreter scoring module.
Checks core ML dependencies, API clients, and optional packages.
"""

def check_import(module_name, package_name=None):
    """check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {module_name}")
        return True
    except ImportError as e:
        pkg = package_name or module_name
        print(f"❌ {module_name} - install with: pip install {pkg}")
        print(f"   error: {e}")
        return False

def check_versions():
    """check versions of key packages"""
    print("\n--- package versions ---")
    
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"cuda devices: {torch.cuda.device_count()}")
    except:
        pass
    
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except:
        pass
    
    try:
        import sentence_transformers
        print(f"sentence-transformers: {sentence_transformers.__version__}")
    except:
        pass
    
    try:
        import sklearn
        print(f"scikit-learn: {sklearn.__version__}")
    except:
        pass
    
    try:
        import numpy as np
        print(f"numpy: {np.__version__}")
    except:
        pass
    
    try:
        import pandas as pd
        print(f"pandas: {pd.__version__}")
    except:
        pass
    
    try:
        import redis
        print(f"redis: {redis.__version__}")
    except:
        pass

def main():
    print("checking dependencies for de_interpreter scoring module...")
    
    # core ML dependencies
    core_deps = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
    ]
    
    # optional dependencies
    optional_deps = [
        ("sentence_transformers", "sentence-transformers"),
        ("redis", "redis"),
    ]
    
    # API and utility dependencies
    api_deps = [
        ("anthropic", "anthropic"),
        ("httpx", "httpx"),
        ("tenacity", "tenacity"),
    ]
    
    print("\n--- Core ML Dependencies ---")
    core_good = True
    for module, package in core_deps:
        if not check_import(module, package):
            core_good = False
    
    print("\n--- Optional Dependencies ---")
    optional_good = True
    for module, package in optional_deps:
        if not check_import(module, package):
            optional_good = False
    
    print("\n--- API Dependencies ---")
    api_good = True
    for module, package in api_deps:
        if not check_import(module, package):
            api_good = False
    
    if core_good and api_good:
        print(f"\n🎉 All essential dependencies installed!")
        if optional_good:
            print(f"✅ All optional dependencies also available!")
        else:
            print(f"⚠️  Some optional dependencies missing (Redis, sentence-transformers)")
            print(f"   Basic functionality will work, but some features may be limited")
        
        check_versions()
        print(f"\n🚀 Ready to use de_interpreter scoring module!")
        print(f"   - BioBERT scoring available")
        print(f"   - Traditional scoring methods available")
        print(f"   - Ensemble methods available")
        if optional_good:
            print(f"   - Redis caching available")
            print(f"   - Sentence transformers available")
    else:
        print(f"\n❌ Missing critical dependencies")
        if not core_good:
            print(f"Core ML packages missing - install with conda:")
            print(f"  conda install pytorch transformers numpy pandas scikit-learn -c pytorch -c conda-forge")
        if not api_good:
            print(f"API packages missing - install with pip:")
            print(f"  pip install anthropic httpx tenacity")
        
        print(f"\nOr use the environment.yml file:")
        print(f"  conda env create -f environment.yml")

if __name__ == "__main__":
    main()