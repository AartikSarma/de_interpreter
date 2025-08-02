"""Prompt templates and builders for Claude synthesis."""

from typing import List, Dict, Any
from ..prioritization import PrioritizedGene, GeneCluster
from ..prioritization.omics_prioritizer import PrioritizedOmicsFeature
from ..parsers import ExperimentalContext
from ..parsers.omics_data import OmicsExperimentContext, OmicsType
from ..literature import Paper


class PromptBuilder:
    """Build structured prompts for Claude API."""

    def get_system_prompt(self) -> str:
        """Get system prompt for scientific discussion."""
        return """You are an expert molecular biologist specializing in gene expression analysis and disease mechanisms. 
Your role is to interpret differential expression results in the context of current scientific literature.

Guidelines:
1. Provide evidence-based interpretations grounded in the provided literature
2. Clearly distinguish between established facts and hypotheses
3. Highlight therapeutic implications when relevant
4. Note any contradictions or gaps in current understanding
5. Use precise scientific language while remaining accessible
6. Always cite sources using author-year format

Structure your responses with clear sections and bullet points where appropriate."""

    def build_gene_prompt(
        self, gene: PrioritizedGene, context: ExperimentalContext, papers: List[Paper]
    ) -> str:
        """Build prompt for single gene discussion."""
        # Gene information
        gene_info = f"""
GENE INFORMATION:
- Gene: {gene.gene_symbol or gene.gene_id}
- Log2 Fold Change: {gene.de_result.log2_fold_change:.2f} ({gene.de_result.fold_change:.1f}-fold {'increase' if gene.de_result.is_upregulated else 'decrease'})
- Adjusted p-value: {gene.de_result.padj:.2e}
- Priority rank: {gene.rank}
"""

        # Experimental context
        context_info = f"""
EXPERIMENTAL CONTEXT:
{context.get_context_string()}
"""

        # Literature summary
        lit_summary = self._summarize_papers(papers, limit=5)

        # Build full prompt
        prompt = f"""{gene_info}

{context_info}

RELEVANT LITERATURE:
{lit_summary}

Please provide a comprehensive discussion of this gene's expression change in the context of {context.disease}. 

Include:
1. The gene's known function and role in {context.disease}
2. Interpretation of the observed expression change
3. How this aligns with or contradicts existing literature
4. Potential biological mechanisms
5. Therapeutic implications if any

Format your response with:
- A main discussion paragraph
- Key findings as bullet points
- Therapeutic implications (if relevant)
- Citations in author-year format"""

        return prompt

    def build_cluster_prompt(
        self,
        cluster: GeneCluster,
        context: ExperimentalContext,
        gene_papers: Dict[str, List[Paper]],
    ) -> str:
        """Build prompt for gene cluster discussion."""
        # Cluster summary
        top_genes = cluster.get_top_genes(n=5)
        gene_list = ", ".join([g.gene_symbol or g.gene_id for g in top_genes])

        cluster_info = f"""
GENE CLUSTER INFORMATION:
- Cluster size: {cluster.size} genes
- Direction: Predominantly {cluster.predominant_direction}regulated
- Mean log2FC: {cluster.mean_log2fc:.2f}
- Top genes: {gene_list}
"""

        # Experimental context
        context_info = f"""
EXPERIMENTAL CONTEXT:
{context.get_context_string()}
"""

        # Literature for top genes
        lit_summaries = []
        for gene in top_genes[:3]:  # Limit to top 3 for space
            gene_key = gene.gene_symbol or gene.gene_id
            papers = gene_papers.get(gene_key, [])[:3]
            if papers:
                lit_summaries.append(
                    f"\n{gene_key}:\n{self._summarize_papers(papers, limit=3)}"
                )

        literature_section = (
            "\n".join(lit_summaries)
            if lit_summaries
            else "No specific literature found."
        )

        prompt = f"""{cluster_info}

{context_info}

RELEVANT LITERATURE:
{literature_section}

Please provide a cohesive discussion of this gene cluster in the context of {context.disease}.

Focus on:
1. Common functional themes or pathways among these genes
2. Collective biological significance of their coordinated expression change
3. Potential upstream regulators or shared mechanisms
4. Overall implications for disease pathophysiology
5. Therapeutic opportunities targeting this gene set

Provide a structured discussion with clear sections."""

        return prompt

    def build_summary_prompt(
        self,
        discussions: List["GeneDiscussion"],
        context: ExperimentalContext,
        de_summary: Dict[str, Any],
    ) -> str:
        """Build prompt for executive summary."""
        # DE statistics
        stats_info = f"""
DIFFERENTIAL EXPRESSION SUMMARY:
- Total genes analyzed: {de_summary.get('total_genes', 'N/A')}
- Significantly changed: {de_summary.get('significant_genes', 'N/A')}
- Upregulated: {de_summary.get('upregulated', 'N/A')}
- Downregulated: {de_summary.get('downregulated', 'N/A')}
"""

        # Key findings from top genes
        top_findings = []
        for disc in discussions[:10]:  # Top 10 genes
            if disc.key_findings:
                gene_name = disc.gene_symbol or disc.gene_id
                top_findings.append(f"{gene_name}: {disc.key_findings[0]}")

        findings_text = "\n".join([f"- {f}" for f in top_findings])

        # Context
        context_info = f"EXPERIMENTAL CONTEXT: {context.get_context_string()}"

        prompt = f"""{stats_info}

{context_info}

KEY FINDINGS FROM TOP GENES:
{findings_text}

Based on the comprehensive analysis of differential expression results and literature review, please provide an executive summary that:

1. Synthesizes the major biological themes emerging from the data
2. Highlights the most significant and actionable findings
3. Places results in the broader context of {context.disease} pathophysiology
4. Identifies key pathways or processes affected
5. Suggests future research directions
6. Notes any surprising or contradictory findings

Format as a structured executive summary with clear sections and bullet points for key takeaways."""

        return prompt

    def _summarize_papers(self, papers: List[Paper], limit: int = 5) -> str:
        """Create concise summary of papers."""
        if not papers:
            return "No relevant papers found."

        summaries = []
        for paper in papers[:limit]:
            # Use full text if available (PMC papers), otherwise use abstract
            text_source = paper.full_text if hasattr(paper, 'full_text') and paper.full_text else paper.abstract
            
            if text_source:
                # For full text, extract more relevant sections
                if hasattr(paper, 'full_text') and paper.full_text and len(paper.full_text) > 1000:
                    # Extract key sections from full text
                    text_lower = text_source.lower()
                    relevant_excerpts = []
                    
                    # Look for sections mentioning key terms
                    sentences = text_source.split(". ")
                    for sent in sentences[:100]:  # Limit to first 100 sentences to avoid too much text
                        if any(keyword in sent.lower() for keyword in [
                            "expression", "regulate", "pathway", "mechanism", "function", 
                            "role", "associated", "involved", "target", "therapeutic"
                        ]):
                            relevant_excerpts.append(sent.strip())
                            if len(relevant_excerpts) >= 3:  # Limit to 3 key excerpts
                                break
                    
                    if relevant_excerpts:
                        summary = f"• {paper.get_citation()}: {' | '.join(relevant_excerpts[:2])}"
                    else:
                        # Fall back to abstract
                        abstract_sentences = paper.abstract.split(". ")
                        relevant_sentence = abstract_sentences[0] if abstract_sentences else "No abstract available."
                        summary = f"• {paper.get_citation()}: {relevant_sentence}"
                else:
                    # Use abstract approach for shorter texts
                    abstract_sentences = text_source.split(". ")
                    relevant_sentence = abstract_sentences[0]
                    for sent in abstract_sentences:
                        if any(
                            keyword in sent.lower()
                            for keyword in ["expression", "regulate", "pathway", "mechanism"]
                        ):
                            relevant_sentence = sent
                            break
                    summary = f"• {paper.get_citation()}: {relevant_sentence}"
            else:
                summary = f"• {paper.get_citation()}: No abstract available."
            
            summaries.append(summary)

        return "\n".join(summaries)

    def get_omics_system_prompt(self, omics_type: OmicsType) -> str:
        """Get omics-specific system prompt."""
        omics_expertise = {
            OmicsType.TRANSCRIPTOMICS: "transcriptomics and gene expression analysis",
            OmicsType.GENOMICS: "genomics and genetic variation analysis", 
            OmicsType.PROTEOMICS: "proteomics and protein abundance analysis",
            OmicsType.METABOLOMICS: "metabolomics and metabolite profiling",
            OmicsType.METAGENOMICS: "metagenomics and microbial community analysis",
            OmicsType.EPIGENOMICS: "epigenomics and chromatin modification analysis",
            OmicsType.LIPIDOMICS: "lipidomics and lipid metabolism analysis"
        }

        expertise_area = omics_expertise.get(omics_type, "multi-omics analysis")

        return f"""You are an expert molecular biologist specializing in {expertise_area} and disease mechanisms. 
Your role is to interpret differential {omics_type.value} results in the context of current scientific literature.

Guidelines:
1. Provide evidence-based interpretations grounded in the provided literature
2. Consider {omics_type.value}-specific biological mechanisms and pathways
3. Clearly distinguish between established facts and hypotheses
4. Highlight therapeutic implications when relevant
5. Note any contradictions or gaps in current understanding
6. Use precise scientific language while remaining accessible
7. Always cite sources using author-year format

Structure your responses with clear sections and bullet points where appropriate."""

    def build_omics_feature_prompt(
        self, 
        feature: PrioritizedOmicsFeature, 
        context: OmicsExperimentContext, 
        papers: List[Paper]
    ) -> str:
        """Build prompt for single omics feature discussion."""
        
        # Feature information
        feature_info = f"""
{context.get_feature_type_name().upper()} INFORMATION:
- {context.get_feature_type_name().title()}: {feature.display_name}
- ID: {feature.feature_id}
- Log2 Fold Change: {feature.omics_feature.log2_fold_change:.2f} ({feature.omics_feature.fold_change:.1f}-fold {'increase' if feature.omics_feature.is_upregulated else 'decrease'})
- Adjusted p-value: {feature.omics_feature.padj:.2e}
- Priority rank: {feature.rank}
- Omics type: {context.omics_type.value}
"""

        # Experimental context
        context_info = f"""
EXPERIMENTAL CONTEXT:
{context.get_context_string()}
- Platform: {context.platform or 'Not specified'}
- Analysis method: {context.analysis_method or 'Standard differential analysis'}
"""

        # Literature summary
        lit_summary = self._summarize_papers(papers, limit=5)

        # Omics-specific questions
        omics_questions = self._get_omics_specific_questions(context.omics_type)

        # Build full prompt
        prompt = f"""{feature_info}

{context_info}

RELEVANT LITERATURE:
{lit_summary}

Please provide a comprehensive discussion of this {context.get_feature_type_name()} addressing:

1. **Biological Function**: What is known about the normal function of this {context.get_feature_type_name()}?

2. **{context.get_omics_context().title()} Change Interpretation**: How should we interpret the observed {context.get_omics_context()} change in the context of {context.disease}?

3. **Disease Relevance**: What is the relationship between this {context.get_feature_type_name()} and {context.disease} pathophysiology?

{omics_questions}

4. **Literature Support**: What does the current literature say about this {context.get_feature_type_name()} in relation to {context.disease}?

5. **Therapeutic Implications**: Are there potential therapeutic targets or biomarker applications?

6. **Confidence Assessment**: Rate your confidence in this interpretation (1-10) and explain any uncertainties.

Format your response with clear headers and cite all sources using author-year format."""

        return prompt

    def build_omics_summary_prompt(
        self,
        discussions: List,
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
    ) -> str:
        """Build executive summary prompt for omics analysis."""
        
        # Statistical overview
        stats_info = f"""
ANALYSIS OVERVIEW:
- Omics type: {context.omics_type.value}
- Total {feature_summary.get('feature_type', 'features')} analyzed: {feature_summary.get('total_features', 'N/A')}
- Significant {feature_summary.get('feature_type', 'features')}: {feature_summary.get('significant_features', 'N/A')}
- Upregulated: {omics_summary.get('upregulated', 'N/A')}
- Downregulated: {omics_summary.get('downregulated', 'N/A')}
- Mean |log2FC|: {omics_summary.get('max_abs_log2fc', 'N/A'):.2f}
"""

        # Key findings from top features
        top_findings = []
        for disc in discussions[:10]:  # Top 10 features
            if disc.key_findings:
                feature_name = disc.gene_symbol or disc.gene_id
                top_findings.append(f"{feature_name}: {disc.key_findings[0]}")

        findings_text = "\n".join([f"- {f}" for f in top_findings])

        # Context
        context_info = f"EXPERIMENTAL CONTEXT: {context.get_context_string()}"

        # Omics-specific summary questions
        omics_summary_questions = self._get_omics_summary_questions(context.omics_type, context.disease)

        prompt = f"""{stats_info}

{context_info}

KEY FINDINGS FROM TOP {context.get_feature_type_name().upper()}S:
{findings_text}

Based on the comprehensive {context.omics_type.value} analysis and literature review, please provide an executive summary that:

1. **Major Biological Themes**: Synthesizes the key biological themes emerging from the {context.omics_type.value} data

2. **Significant Findings**: Highlights the most significant and actionable findings

3. **Disease Context**: Places results in the broader context of {context.disease} pathophysiology

{omics_summary_questions}

4. **Future Directions**: Suggests follow-up experiments or research directions

5. **Clinical Relevance**: Notes potential biomarker or therapeutic applications

6. **Limitations**: Acknowledges any limitations of the current analysis

Format as a structured executive summary with clear sections and bullet points for key takeaways."""

        return prompt

    def _get_omics_specific_questions(self, omics_type: OmicsType) -> str:
        """Get omics-type specific analysis questions."""
        questions = {
            OmicsType.TRANSCRIPTOMICS: "3. **Regulatory Mechanisms**: What upstream regulators or pathways might be driving this expression change?",
            
            OmicsType.PROTEOMICS: "3. **Protein Function**: How might changes in protein abundance affect cellular processes and pathways?",
            
            OmicsType.METABOLOMICS: "3. **Metabolic Impact**: How does this metabolite change affect metabolic flux and energy production?",
            
            OmicsType.GENOMICS: "3. **Functional Impact**: What is the predicted functional impact of this genetic variant?",
            
            OmicsType.METAGENOMICS: "3. **Microbial Function**: What role does this microorganism play in host health and disease?",
            
            OmicsType.EPIGENOMICS: "3. **Regulatory Impact**: How might this epigenetic change affect gene expression and chromatin structure?",
            
            OmicsType.LIPIDOMICS: "3. **Membrane Impact**: How do changes in lipid composition affect membrane function and signaling?"
        }
        
        return questions.get(omics_type, "3. **Mechanistic Insights**: What mechanisms might underlie this observed change?")

    def _get_omics_summary_questions(self, omics_type: OmicsType, disease: str) -> str:
        """Get omics-specific executive summary questions."""
        questions = {
            OmicsType.TRANSCRIPTOMICS: f"3. **Pathway Analysis**: What key transcriptional pathways are dysregulated in {disease}?",
            
            OmicsType.PROTEOMICS: f"3. **Protein Networks**: What protein complexes and interaction networks are altered in {disease}?",
            
            OmicsType.METABOLOMICS: f"3. **Metabolic Pathways**: What metabolic pathways are perturbed in {disease}?",
            
            OmicsType.GENOMICS: f"3. **Genetic Architecture**: What genes and pathways are genetically associated with {disease} risk?",
            
            OmicsType.METAGENOMICS: f"3. **Microbiome Dysbiosis**: How is the microbial community structure altered in {disease}?",
            
            OmicsType.EPIGENOMICS: f"3. **Epigenetic Landscape**: What chromatin modifications and regulatory elements are altered in {disease}?",
            
            OmicsType.LIPIDOMICS: f"3. **Lipid Metabolism**: How is lipid homeostasis disrupted in {disease}?"
        }
        
        return questions.get(omics_type, f"3. **Molecular Mechanisms**: What molecular mechanisms are altered in {disease}?")
