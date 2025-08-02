├── scoring/                    # new scoring module
│   ├── __init__.py
│   ├── base.py                # abstract base scorer class
│   ├── biobert/               # biobert-specific scoring
│   │   ├── __init__.py
│   │   ├── scorer.py          # main biobert scorer implementation
│   │   ├── models.py          # model loading/caching logic
│   │   └── utils.py           # biobert-specific utilities
│   ├── traditional/           # baseline/traditional scorers
│   │   ├── __init__.py
│   │   ├── tfidf.py          # tf-idf based scoring
│   │   ├── bm25.py           # bm25 scoring
│   │   └── jaccard.py        # simple overlap metrics
│   ├── ensemble/              # combining multiple scorers
│   │   ├── __init__.py
│   │   ├── weighted.py       # weighted ensemble
│   │   └── rank_fusion.py    # rank-based fusion
│   ├── cache/                 # embedding cache management
│   │   ├── __init__.py
│   │   ├── manager.py        # cache operations
│   │   └── storage.py        # different storage backends
│   └── evaluation/            # scorer evaluation tools
│       ├── __init__.py
│       ├── metrics.py        # evaluation metrics
│       └── benchmarks.py     # benchmark datasets