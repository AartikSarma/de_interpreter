#!/usr/bin/env python3
"""
Example comparing the two approaches for gene-query similarity scoring
"""

# Example to illustrate the difference between the approaches:

# Original Approach:
# For genes = ["TP53", "BRCA1", "MYC"] and query = "cancer progression"
# 1. Search "TP53 AND cancer progression" → get abstracts A1, A2, A3
# 2. Score A1, A2, A3 against "cancer progression" → scores S1, S2, S3
# 3. Search "BRCA1 AND cancer progression" → get abstracts B1, B2, B3  
# 4. Score B1, B2, B3 against "cancer progression" → scores T1, T2, T3
# 5. Search "MYC AND cancer progression" → get abstracts C1, C2, C3
# 6. Score C1, C2, C3 against "cancer progression" → scores U1, U2, U3
# 7. Final score = S1+S2+S3 + T1+T2+T3 + U1+U2+U3

# Refined Approach (Your New Specification):
# For genes = ["TP53", "BRCA1", "MYC"] and query = "cancer progression"  
# 1. Search "cancer progression TP53" → get abstracts A1, A2, A3
# 2. Search "cancer progression BRCA1" → get abstracts B1, B2, B3
# 3. Search "cancer progression MYC" → get abstracts C1, C2, C3
# 4. Pool all abstracts: [A1, A2, A3, B1, B2, B3, C1, C2, C3]
# 5. Create combined query: "cancer progression TP53 BRCA1 MYC"
# 6. Score ALL abstracts against combined query → scores S1, S2, S3, T1, T2, T3, U1, U2, U3
# 7. Final score = S1+S2+S3+T1+T2+T3+U1+U2+U3

print("""
Gene-Query Similarity Scoring: Approach Comparison

REFINED APPROACH (Implemented):
=================================
1. Search Phase:
   - Gene TP53: Search "cancer progression TP53" → 20 papers
   - Gene BRCA1: Search "cancer progression BRCA1" → 20 papers  
   - Gene MYC: Search "cancer progression MYC" → 20 papers

2. Pooling Phase:
   - Combine all abstracts: 60 abstracts total (may have duplicates)

3. Scoring Phase:
   - Combined query: "cancer progression TP53 BRCA1 MYC"
   - Score each of 60 abstracts vs combined query
   - Sum all 60 similarity scores

ADVANTAGES:
✓ Considers full gene context when scoring
✓ More holistic relevance assessment
✓ Avoids gene-specific bias in scoring
✓ Better for multi-gene pathway analysis

EXAMPLE USAGE:
python gene_query_similarity_scorer.py --genes TP53 BRCA1 MYC --query "cancer progression" --top-papers 20
""")

# The key insight is that by scoring against the full gene context,
# an abstract about "TP53-BRCA1 interaction in cancer" would score highly
# because it's relevant to multiple genes in your list, whereas in the 
# original approach it might only score well for one gene search.
