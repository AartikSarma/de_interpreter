"""Prompt templates and builders for Claude synthesis."""

from typing import List, Dict, Any
from ..prioritization import PrioritizedGene, GeneCluster
from ..parsers import ExperimentalContext
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
            # Extract key sentence from abstract
            abstract_sentences = paper.abstract.split(". ")
            # Try to find most relevant sentence (simple heuristic)
            relevant_sentence = abstract_sentences[0]
            for sent in abstract_sentences:
                if any(
                    keyword in sent.lower()
                    for keyword in ["expression", "regulate", "pathway", "mechanism"]
                ):
                    relevant_sentence = sent
                    break

            summary = f"• {paper.get_citation()}: {relevant_sentence}"
            summaries.append(summary)

        return "\n".join(summaries)
