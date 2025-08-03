#!/usr/bin/env python3
"""
Test script to evaluate BioBERT models on distinguishing relevant vs related content.

Tests whether models can correctly identify that a pancreatic cancer abstract about ARG1
is more relevant to the query "What is the relevance of ARG1 in pancreatic cancer?"
than a sepsis-ARDS abstract that also mentions ARG1 but in a different context.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# test models - prioritizing ones with safetensors and smaller sizes
BIOBERT_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",                 # baseline - not biobert but reliable
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",   # smaller pubmedbert variant  
    "dmis-lab/biobert-base-cased-v1.1",                       # original biobert
]

# backup models to try if network issues persist
BACKUP_MODELS = [
    "gsarti/biobert-nli",                                      
    "pritamdeka/S-BioBert-snli-multinli-stsb",                
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb", 
]

def test_biobert_model(model_name):
    """test if a biobert model works with sentence-transformers"""
    print(f"\n--- testing {model_name} ---")
    
    try:
        from sentence_transformers import SentenceTransformer
        import time
        
        # test sentences - biomedical context
        sentences = [
            "What is the relevance of ARG1 in pancreatic cancer?",
            "An extensive fibroinflammatory stroma rich in macrophages is a hallmark of pancreatic cancer. In this disease, it is well appreciated that macrophages are immunosuppressive and contribute to the poor response to immunotherapy; however, the mechanisms of immune suppression are complex and not fully understood. Immunosuppressive macrophages are classically defined by the expression of the enzyme Arginase 1 (ARG1), which we demonstrated is potently expressed in pancreatic tumor-associated macrophages from both human patients and mouse models. While routinely used as a polarization marker, ARG1 also catabolizes arginine, an amino acid required for T cell activation and proliferation. To investigate this metabolic function, we used a genetic and a pharmacologic approach to target Arg1 in pancreatic cancer. Genetic inactivation of Arg1 in macrophages, using a dual recombinase genetically engineered mouse model of pancreatic cancer, delayed formation of invasive disease, while increasing CD8+ T cell infiltration. Additionally, Arg1 deletion induced compensatory mechanisms, including Arg1 overexpression in epithelial cells, namely Tuft cells, and Arg2 overexpression in a subset of macrophages. To overcome these compensatory mechanisms, we used a pharmacological approach to inhibit arginase. Treatment of established tumors with the arginase inhibitor CB-1158 exhibited further increased CD8+ T cell infiltration, beyond that seen with the macrophage-specific knockout, and sensitized the tumors to anti-PD1 immune checkpoint blockade. Our data demonstrate that Arg1 drives immune suppression in pancreatic cancer by depleting arginine and inhibiting T cell activation.",
            "While acute respiratory distress syndrome (ARDS) is the highest mortality with the worse outcomes among other causes of ARDS, a few studies focused on the risk identification of sepsis-ARDS. Here, this study determined the levels of plasma arginase 1 (ARG1) to apply as a novel biomarker for sepsis-ARDS. A total of 46 endotracheal intubated patients with ARDS categorized as sepsis-ARDS (n = 28) and non-sepsis ARDS (n = 18) were enrolled. The clinical outcomes were obtained prospectively and ARG1 level was determined by ELISA. Plasma ARG1 in sepsis-ARDS was higher than non-sepsis ARDS and correlated with ARDS severity, including APACHE II score, SOFA score, interleukin-6, lactate, and the reduced PaO2/FiO2 ratio. Additionally, the higher plasma ARG1 in sepsis-ARDS indicated the higher mortality and the longer duration of ventilator use. There was a non-significant correlation in patients with non-sepsis ARDS. The area under the curves (AUC) in a receiver operating characteristic (ROC) curve of ARG1 for the prediction of 28-days mortality and ventilator free day in sepsis-ARDS were 0.80 and 0.67, respectively, while AUC to diagnose sepsis-ARDS was 0.72, All the performances were improved when combined the ARG1 levels with SOFA score. Moreover, the relationship between plasma ARG1 and neutrophils was demonstrated. Flow cytometry demonstrated a high level of neutrophil ARG1 production with high degranulation levels, supporting the role of neutrophils in ARG1 production during sepsis-ARDS. In conclusion, the plasma ARG1 levels may be a potential marker for predicting the worsen outcomes of sepsis-ARDS. Early detection of plasma ARG1 could help clinicians to manage sepsis-ARDS."
        ]
        
        print(f"loading model (this may take a while for first download)...")
        start_time = time.time()
        
        # add timeout for model loading
        try:
            model = SentenceTransformer(model_name)
        except Exception as e:
            if "timeout" in str(e).lower() or "connection" in str(e).lower():
                print(f"❌ network timeout - try again later or use cached model")
                return False, None
            else:
                raise e
        
        load_time = time.time() - start_time
        print(f"model loaded in {load_time:.1f}s")
        
        print(f"encoding {len(sentences)} sentences...")
        embeddings = model.encode(sentences)
        
        print(f"embeddings shape: {embeddings.shape}")
        print(f"embedding dimension: {embeddings.shape[1]}")
        
        # compute similarities
        similarities = cosine_similarity(embeddings)
        print(f"similarity query vs pancreatic cancer abstract: {similarities[0,1]:.3f}")
        print(f"similarity query vs sepsis-ARDS abstract: {similarities[0,2]:.3f}")
        
        # check if the relevant abstract (pancreatic cancer) is more similar to query than related abstract (sepsis-ARDS)
        relevant_sim = similarities[0,1]  # query vs pancreatic cancer
        related_sim = similarities[0,2]   # query vs sepsis-ARDS
        
        if relevant_sim > related_sim:
            print("✅ model correctly identifies relevant vs related content")
            print(f"   Relevant abstract is {relevant_sim - related_sim:.3f} more similar")
        else:
            print("⚠️  model cannot distinguish relevant from related content")
            print(f"   Related abstract is {related_sim - relevant_sim:.3f} more similar")
            
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
            "What is the relevance of ARG1 in pancreatic cancer?",
            "An extensive fibroinflammatory stroma rich in macrophages is a hallmark of pancreatic cancer. In this disease, it is well appreciated that macrophages are immunosuppressive and contribute to the poor response to immunotherapy; however, the mechanisms of immune suppression are complex and not fully understood. Immunosuppressive macrophages are classically defined by the expression of the enzyme Arginase 1 (ARG1), which we demonstrated is potently expressed in pancreatic tumor-associated macrophages from both human patients and mouse models. While routinely used as a polarization marker, ARG1 also catabolizes arginine, an amino acid required for T cell activation and proliferation. To investigate this metabolic function, we used a genetic and a pharmacologic approach to target Arg1 in pancreatic cancer. Genetic inactivation of Arg1 in macrophages, using a dual recombinase genetically engineered mouse model of pancreatic cancer, delayed formation of invasive disease, while increasing CD8+ T cell infiltration. Additionally, Arg1 deletion induced compensatory mechanisms, including Arg1 overexpression in epithelial cells, namely Tuft cells, and Arg2 overexpression in a subset of macrophages. To overcome these compensatory mechanisms, we used a pharmacological approach to inhibit arginase. Treatment of established tumors with the arginase inhibitor CB-1158 exhibited further increased CD8+ T cell infiltration, beyond that seen with the macrophage-specific knockout, and sensitized the tumors to anti-PD1 immune checkpoint blockade. Our data demonstrate that Arg1 drives immune suppression in pancreatic cancer by depleting arginine and inhibiting T cell activation.",
            "While acute respiratory distress syndrome (ARDS) is the highest mortality with the worse outcomes among other causes of ARDS, a few studies focused on the risk identification of sepsis-ARDS. Here, this study determined the levels of plasma arginase 1 (ARG1) to apply as a novel biomarker for sepsis-ARDS. A total of 46 endotracheal intubated patients with ARDS categorized as sepsis-ARDS (n = 28) and non-sepsis ARDS (n = 18) were enrolled. The clinical outcomes were obtained prospectively and ARG1 level was determined by ELISA. Plasma ARG1 in sepsis-ARDS was higher than non-sepsis ARDS and correlated with ARDS severity, including APACHE II score, SOFA score, interleukin-6, lactate, and the reduced PaO2/FiO2 ratio. Additionally, the higher plasma ARG1 in sepsis-ARDS indicated the higher mortality and the longer duration of ventilator use. There was a non-significant correlation in patients with non-sepsis ARDS. The area under the curves (AUC) in a receiver operating characteristic (ROC) curve of ARG1 for the prediction of 28-days mortality and ventilator free day in sepsis-ARDS were 0.80 and 0.67, respectively, while AUC to diagnose sepsis-ARDS was 0.72, All the performances were improved when combined the ARG1 levels with SOFA score. Moreover, the relationship between plasma ARG1 and neutrophils was demonstrated. Flow cytometry demonstrated a high level of neutrophil ARG1 production with high degranulation levels, supporting the role of neutrophils in ARG1 production during sepsis-ARDS. In conclusion, the plasma ARG1 levels may be a potential marker for predicting the worsen outcomes of sepsis-ARDS. Early detection of plasma ARG1 could help clinicians to manage sepsis-ARDS."

        ]
        
        print(f"tokenizing...")
        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        print(f"running model...")
        with torch.no_grad():
            outputs = model(**inputs)
        
        # mean pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        print(f"embeddings shape: {embeddings.shape}")
        
        # compute similarities between all pairs
        query_emb = embeddings[0]
        pancreatic_emb = embeddings[1] 
        sepsis_emb = embeddings[2]
        
        sim_query_pancreatic = torch.cosine_similarity(query_emb, pancreatic_emb, dim=0)
        sim_query_sepsis = torch.cosine_similarity(query_emb, sepsis_emb, dim=0)
        
        print(f"similarity query vs pancreatic cancer abstract: {sim_query_pancreatic.item():.3f}")
        print(f"similarity query vs sepsis-ARDS abstract: {sim_query_sepsis.item():.3f}")
        
        if sim_query_pancreatic > sim_query_sepsis:
            print("✅ model correctly identifies relevant vs related content")
            print(f"   Relevant abstract is {(sim_query_pancreatic - sim_query_sepsis).item():.3f} more similar")
        else:
            print("⚠️  model cannot distinguish relevant from related content")
            print(f"   Related abstract is {(sim_query_sepsis - sim_query_pancreatic).item():.3f} more similar")
        
        return True, embeddings.shape[1]
        
    except Exception as e:
        print(f"❌ failed: {e}")
        return False, None

def main():
    """test different biobert approaches on relevance vs relatedness task"""
    print("Testing BioBERT models on distinguishing relevant vs related content...")
    print("Query: 'What is the relevance of ARG1 in pancreatic cancer?'")
    print("Relevant: Pancreatic cancer abstract about ARG1")
    print("Related: Sepsis-ARDS abstract about ARG1 (different disease context)")
    print("="*80)
    
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