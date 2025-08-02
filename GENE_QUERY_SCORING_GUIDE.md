# Gene + Query Literature Similarity Scoring Implementation Guide

## Refined Task Overview (Updated)

Your refined approach is:
1. **For each gene G**: Search PubMed for `query + G` (e.g., "cancer progression TP53")
2. **Pool all abstracts** from all gene searches into one collection  
3. **Create combined query**: `query + " " + " ".join(genes)` (e.g., "cancer progression TP53 BRCA1 MYC")
4. **Score each abstract** against this combined query using similarity
5. **Sum all similarities** for final score

**This is a much better approach because:**
- ✅ Avoids double-counting genes in similarity scoring
- ✅ Considers full gene context when evaluating relevance
- ✅ Better for multi-gene pathway analysis  
- ✅ Papers mentioning multiple genes from your list score higher

## Example Workflow

**Input**: genes = ["TP53", "BRCA1", "MYC"], query = "cancer progression"

**Step 1 - Individual Gene Searches**:
- Search 1: "cancer progression TP53" → 20 papers (abstracts A1-A20)
- Search 2: "cancer progression BRCA1" → 20 papers (abstracts B1-B20)  
- Search 3: "cancer progression MYC" → 20 papers (abstracts C1-C20)

**Step 2 - Pool Abstracts**:
- Combined pool: [A1, A2, ..., A20, B1, B2, ..., B20, C1, C2, ..., C20] (60 total)

**Step 3 - Create Combined Query**:
- Combined query: "cancer progression TP53 BRCA1 MYC"

**Step 4 - Score Against Combined Query**:
- Score A1 vs "cancer progression TP53 BRCA1 MYC" → similarity S1
- Score A2 vs "cancer progression TP53 BRCA1 MYC" → similarity S2
- ... (continue for all 60 abstracts)

**Step 5 - Sum Results**:
- Final score = S1 + S2 + ... + S60

## Key Insight

An abstract about "TP53-BRCA1 interaction in cancer progression" will score highly because it's relevant to multiple genes in your list AND your query, whereas other approaches might miss this multi-gene relevance.

## Relevant Files in the Codebase

### Core Components You Need:

1. **Literature Search**: 
   - `src/de_interpreter/literature/pmc_client.py` - PMC search client
   - `src/de_interpreter/literature/pmc_retriever.py` - PubMed retrieval
   - `src/de_interpreter/literature/futurehouse_client.py` - Alternative API

2. **Similarity Scoring**:
   - `src/de_interpreter/scoring/biobert/scorer.py` - BioBERT semantic similarity
   - `src/de_interpreter/scoring/traditional/jaccard.py` - Jaccard similarity
   - `src/de_interpreter/scoring/traditional/tfidf.py` - TF-IDF similarity
   - `src/de_interpreter/scoring/traditional/bm25.py` - BM25 similarity

3. **Example Implementations**:
   - `score_query_vs_pmid.py` - Query vs abstract scoring example
   - `test_scoring_relevant_vs_irrelvant.py` - BioBERT testing

### Existing Methods That Do Similar Tasks:

1. **Batch Gene Searches**: 
   ```python
   # From src/de_interpreter/literature/pmc_client.py
   async def batch_search(self, queries: List[str], limit_per_query: int = 5)
   
   # From src/de_interpreter/literature/pmc_client.py  
   async def search_gene_disease(self, gene: str, disease: str, ...)
   ```

2. **Similarity Scoring**:
   ```python
   # From src/de_interpreter/scoring/biobert/scorer.py
   def score_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[float]
   ```

## Implementation Approaches

### Approach 1: Use Existing Components (Recommended)

**Advantages**: 
- Leverages existing code
- Already handles caching, rate limiting, error handling
- Multiple similarity scoring options

**Files to use**:
- `PMCClient` for literature search
- `BioBERTScorer` or `JaccardScorer` for similarity
- `LiteratureCache` for caching results

```python
async def score_gene_list_similarity(genes: List[str], query: str, max_papers: int = 20):
    async with PMCClient() as client:
        scorer = BioBERTScorer()  # or JaccardScorer()
        total_score = 0.0
        
        for gene in genes:
            # Search: gene + query
            result = await client.search_gene_disease(gene, query, limit=max_papers)
            
            # Convert to documents format
            documents = [{'title': p.title, 'abstract': p.abstract} for p in result.papers]
            
            # Score all abstracts against query
            scores = scorer.score_documents(query, documents)
            
            # Sum for this gene
            total_score += sum(scores)
    
    return total_score
```

### Approach 2: Direct PubMed API (Custom)

**Advantages**:
- More control over search parameters
- Direct access to PubMed E-utilities
- Can customize similarity scoring

**Components**:
- Custom PubMed search using E-utilities API
- Sentence transformers for similarity scoring
- Direct abstract fetching

This is what the `gene_query_similarity_scorer.py` script I created implements.

### Approach 3: Hybrid Approach

Use existing `pmc_retriever.py` for PubMed searches but implement custom similarity scoring.

## Recommended Implementation Steps

### Step 1: Choose Your Similarity Method

**BioBERT (Semantic)**: Best for understanding biological context
```python
from de_interpreter.scoring.biobert.scorer import BioBERTScorer
scorer = BioBERTScorer()
```

**Jaccard (Term Overlap)**: Faster, simpler, good baseline
```python
from de_interpreter.scoring.traditional.jaccard import JaccardScorer
scorer = JaccardScorer()
```

**Sentence Transformers (External)**: Good middle ground
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### Step 2: Implement Gene-Query Search

```python
async def search_gene_query_papers(gene: str, query: str, max_papers: int = 20):
    async with PMCClient() as client:
        # Method 1: Use existing search_gene_disease
        result = await client.search_gene_disease(
            gene=gene, 
            disease=query,  # treat query as disease context
            limit=max_papers
        )
        
        # Method 2: Use batch_search with custom query
        search_query = f"{gene} {query}"
        result = await client.search(search_query, limit=max_papers)
        
        return result.papers
```

### Step 3: Compute Similarities

```python
def compute_paper_similarities(query: str, papers: List[Paper], scorer):
    documents = []
    for paper in papers:
        doc = {
            'title': paper.title,
            'abstract': paper.abstract
        }
        documents.append(doc)
    
    # Get similarity scores
    scores = scorer.score_documents(query, documents)
    return scores
```

### Step 4: Aggregate Results

```python
async def score_gene_list(genes: List[str], query: str, max_papers_per_gene: int = 20):
    total_score = 0.0
    scorer = BioBERTScorer()  # or your choice
    
    for gene in genes:
        papers = await search_gene_query_papers(gene, query, max_papers_per_gene)
        similarities = compute_paper_similarities(query, papers, scorer)
        gene_score = sum(similarities)
        total_score += gene_score
        
        print(f"Gene {gene}: {len(papers)} papers, score = {gene_score:.3f}")
    
    return total_score
```

## Usage Examples

### Example 1: Cancer Research
```python
genes = ["TP53", "BRCA1", "MYC", "RB1", "APC"]
query = "tumor suppressor function"
score = await score_gene_list(genes, query, max_papers_per_gene=20)
```

### Example 2: Inflammatory Response
```python
genes = ["TNF", "IL1B", "IL6", "NFKB1", "TLR4"]
query = "inflammatory response pathway"
score = await score_gene_list(genes, query, max_papers_per_gene=15)
```

### Example 3: Metabolic Pathways
```python
genes = ["INSULIN", "GLUT4", "PPARA", "SREBF1"]
query = "glucose metabolism regulation"
score = await score_gene_list(genes, query, max_papers_per_gene=25)
```

## Performance Considerations

1. **Rate Limiting**: PubMed API has rate limits (~3 requests/second)
2. **Caching**: Use `LiteratureCache` to avoid re-fetching papers
3. **Batch Processing**: Process genes in batches to manage memory
4. **Similarity Model**: BioBERT is slower but more accurate than Jaccard

## Files I Created for You

1. **`gene_query_similarity_scorer.py`**: Complete standalone implementation
   - Direct PubMed API usage
   - Sentence transformers for similarity
   - Command-line interface
   - JSON output for results

2. **`example_gene_query_scoring.py`**: Example using existing components
   - Shows how to use PMCClient and BioBERTScorer
   - Async implementation
   - Demonstrates the workflow

## Next Steps

1. **Test with small gene lists first** (2-3 genes) to verify the approach
2. **Choose your similarity method** based on accuracy vs speed needs
3. **Implement caching** if you'll be running this multiple times
4. **Add result persistence** to save detailed scoring information
5. **Consider parallel processing** for large gene lists

The codebase already has most of what you need - it's just a matter of combining the existing literature search and scoring components in the right way for your specific use case!
