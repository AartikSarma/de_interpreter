#!/usr/bin/env python3
"""
Check where models are cached and what's stored
"""

import sentence_transformers
import transformers
import os

print('=== Sentence Transformers Cache ===')
st_cache = sentence_transformers.util.default_cache_folder
print(f'Default cache folder: {st_cache}')
print(f'Cache exists: {os.path.exists(st_cache)}')

if os.path.exists(st_cache):
    models = [d for d in os.listdir(st_cache) if os.path.isdir(os.path.join(st_cache, d))]
    print(f'Cached models ({len(models)}):')
    for model in models[:10]:  # show first 10
        print(f'  - {model}')
    if len(models) > 10:
        print(f'  ... and {len(models) - 10} more')

print('\n=== Transformers Cache ===')
hf_cache = transformers.utils.hub.default_cache_path
print(f'HuggingFace cache: {hf_cache}')
print(f'Cache exists: {os.path.exists(hf_cache)}')

if os.path.exists(hf_cache):
    # List some cached models
    cache_files = []
    for root, dirs, files in os.walk(hf_cache):
        for d in dirs:
            if '--' in d and len(d) > 20:  # looks like a model hash
                cache_files.append(d)
        break  # just check top level
    
    print(f'Cached model entries: {len(cache_files)}')
    if cache_files:
        print('Recent cache entries:')
        for entry in cache_files[:5]:
            print(f'  - {entry}')

print('\n=== Cache Sizes ===')
def get_dir_size(path):
    if not os.path.exists(path):
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total

if os.path.exists(st_cache):
    size_mb = get_dir_size(st_cache) / (1024 * 1024)
    print(f'Sentence-transformers cache: {size_mb:.1f} MB')

if os.path.exists(hf_cache):
    size_mb = get_dir_size(hf_cache) / (1024 * 1024)
    print(f'HuggingFace cache: {size_mb:.1f} MB')

print('\n=== Environment Variables ===')
print(f'SENTENCE_TRANSFORMERS_HOME: {os.environ.get("SENTENCE_TRANSFORMERS_HOME", "not set")}')
print(f'HF_HOME: {os.environ.get("HF_HOME", "not set")}')
print(f'TRANSFORMERS_CACHE: {os.environ.get("TRANSFORMERS_CACHE", "not set")}')
