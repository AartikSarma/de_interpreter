#!/usr/bin/env python3
"""
test script to see which biobert models actually work with sentence-transformers
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# test models we found that should work
BIOBERT_MODELS = [
    "gsarti/biobert-nli",                                      # the one i mentioned earlier
    "pritamdeka/S-BioBert-snli-multinli-stsb",                # looks good
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", # most comprehensive
    "xlreator/biosyn-biobert-snomed",                          # medical ontology focus
    "AHDMK/Sentence-GISTEmbedLoss-BioBert-Allnli-scinli",     # newer training approach
]

def test_biobert_model(model_name):
    """test if a biobert model works with sentence-transformers"""
    print(f"\n--- testing {model_name} ---")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # test sentences - biomedical context
        sentences = [
            "Differential gene expression analysis revealed upregulation of inflammatory pathways",
            "RNA-seq data shows significant changes in metabolic gene expression",
            "The cat sat on the mat"  # non-biomedical control
        ]
        
        print(f"loading model...")
        model = SentenceTransformer(model_name)
        
        print(f"encoding {len(sentences)} sentences...")
        embeddings = model.encode(sentences)
        
        print(f"embeddings shape: {embeddings.shape}")
        print(f"embedding dimension: {embeddings.shape[1]}")
        
        # compute similarities
        similarities = cosine_similarity(embeddings)
        print(f"similarity between biomedical sentences: {similarities[0,1]:.3f}")
        print(f"similarity bio vs non-bio: {similarities[0,2]:.3f}")
        
        # check if biomedical sentences are more similar to each other
        bio_sim = similarities[0,1]
        non_bio_sim = similarities[0,2]
        
        if bio_sim > non_bio_sim:
            print("✅ model correctly identifies biomedical similarity")
        else:
            print("⚠️  model doesn't distinguish biomedical context well")
            
        return True, embeddings.shape[1]
        
    except Exception as e:
        print(f"❌ failed: {e}")
        return False, None

def test_manual_biobert():
    """test using transformers library directly with biobert"""
    print(f"\n--- testing manual biobert (transformers library) ---")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        # use microsoft's pubmedbert - should be solid
        model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        
        print(f"loading tokenizer and model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        sentences = [
            "Differential gene expression analysis revealed upregulation",
            "RNA-seq shows metabolic pathway changes"
        ]
        
        print(f"tokenizing...")
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        print(f"running model...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        print(f"embeddings shape: {embeddings.shape}")
        
        # compute similarity
        sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        print(f"similarity: {sim.item():.3f}")
        
        return True, embeddings.shape[1]
        
    except Exception as e:
        print(f"❌ failed: {e}")
        return False, None

def main():
    """test different biobert approaches"""
    print("testing biobert models for similarity scoring...")
    
    working_models = []
    
    # test sentence-transformers models
    for model_name in BIOBERT_MODELS:
        success, dim = test_biobert_model(model_name)
        if success:
            working_models.append((model_name, dim, "sentence-transformers"))
    
    # test manual approach
    success, dim = test_manual_biobert()
    if success:
        working_models.append(("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", dim, "transformers"))
    
    print(f"\n=== summary ===")
    if working_models:
        print(f"found {len(working_models)} working biobert models:")
        for model, dim, library in working_models:
            print(f"  - {model} ({dim}d, {library})")
        
        print(f"\nrecommendation:")
        # prefer sentence-transformers for simplicity
        st_models = [m for m in working_models if m[2] == "sentence-transformers"]
        if st_models:
            best = st_models[0]  # just pick the first working one
            print(f"use: {best[0]} with sentence-transformers")
            print(f"embedding dim: {best[1]}")
        else:
            print(f"use transformers library with manual pooling")
    else:
        print("no working models found - check dependencies")

if __name__ == "__main__":
    main()