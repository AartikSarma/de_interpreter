"""Simplified synthesis engine using Claude API."""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import anthropic

from ..prioritization.prioritizer import PrioritizedFeature
from ..literature.cache import SearchResult
from ..parsers.omics_data import OmicsExperimentContext


@dataclass
class CitationInfo:
    """Container for citation information from Claude API."""
    source_id: str
    quote: str
    start_char: int
    end_char: int
    paper_citation: str
    paper_url: str

@dataclass
class FeatureDiscussion:
    """Container for feature discussion results."""
    feature_id: str
    feature_symbol: Optional[str]
    discussion_text: str
    key_findings: List[str]
    citations: List[str] = None
    citation_info: List[CitationInfo] = None
    therapeutic_implications: Optional[str] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
        if self.citation_info is None:
            self.citation_info = []


class OmicsSynthesizer:
    """Simplified synthesis engine for omics analysis."""
    
    def __init__(self, api_key: str, progress_callback: Optional[Callable] = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.progress_callback = progress_callback
    
    async def synthesize_feature_discussions(
        self,
        prioritized_features: List[PrioritizedFeature],
        literature_results: List[SearchResult],
        context: OmicsExperimentContext
    ) -> List[FeatureDiscussion]:
        """Synthesize feature discussions from prioritized features and literature."""
        discussions = []
        total_features = len(prioritized_features)
        
        for i, (feature, lit_result) in enumerate(zip(prioritized_features, literature_results)):
            if self.progress_callback:
                progress = 50 + int((i / total_features) * 30)  # 50-80%
                self.progress_callback(f"Synthesizing discussion for {feature.feature.display_name}...", progress)
            
            discussion = await self._synthesize_single_feature(feature, lit_result, context)
            discussions.append(discussion)
            
            # Small delay to avoid rate limits
            await asyncio.sleep(0.5)
        
        return discussions
    
    async def _synthesize_single_feature(
        self,
        prioritized_feature: PrioritizedFeature,
        literature_result: SearchResult,
        context: OmicsExperimentContext
    ) -> FeatureDiscussion:
        """Synthesize discussion for a single feature."""
        feature = prioritized_feature.feature
        
        # Prepare papers for citation support
        papers = literature_result.papers[:3]  # Use top 3 papers
        source_papers = {f"pmid_{paper.pmid}": paper for paper in papers}
        
        # Build literature context for prompt
        literature_context = ""
        basic_citations = []
        
        for paper in papers:
            if paper.abstract:
                literature_context += f"\n\nTitle: {paper.title}\nAbstract: {paper.abstract[:500]}..."
                basic_citations.append(paper.citation)
        
        # Build synthesis prompt
        prompt = self._build_synthesis_prompt(feature, prioritized_feature, context, literature_context)
        
        try:
            # Prepare sources for Claude API citations
            sources = [paper.to_claude_source() for paper in papers if paper.text_content]
            
            # Call Claude API with sources for citations
            if sources:
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}],
                    sources=sources
                )
            else:
                # Fallback to basic API call without sources
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[{"role": "user", "content": prompt}]
                )
            
            discussion_text = response.content[0].text
            
            # Parse citations from response
            citation_info = self._parse_citations_from_response(response, source_papers)
            
            # Extract key findings (simple heuristic)
            key_findings = self._extract_key_findings(discussion_text, feature)
            
            # Extract therapeutic implications
            therapeutic_implications = self._extract_therapeutic_implications(discussion_text)
            
            return FeatureDiscussion(
                feature_id=feature.feature_id,
                feature_symbol=feature.feature_symbol,
                discussion_text=discussion_text,
                key_findings=key_findings,
                citations=basic_citations,
                citation_info=citation_info,
                therapeutic_implications=therapeutic_implications
            )
            
        except Exception as e:
            print(f"Error synthesizing discussion for {feature.display_name}: {e}")
            
            # Fallback to basic discussion
            return self._create_fallback_discussion(prioritized_feature, context)
    
    def _build_synthesis_prompt(
        self,
        feature: "OmicsFeature",
        prioritized_feature: PrioritizedFeature,
        context: OmicsExperimentContext,
        literature_context: str
    ) -> str:
        """Build synthesis prompt for Claude."""
        feature_type = context.get_feature_type_name()
        direction = "upregulated" if feature.is_upregulated else "downregulated"
        
        prompt = f"""
Please provide a scientific discussion of the {feature_type} {feature.display_name} based on the following information:

**Experimental Context:**
- Omics Type: {context.omics_type.value}
- Comparison: {context.treatment} vs {context.control}
- Tissue/Sample: {context.tissue}
- Organism: {context.organism}
- Disease Context: {context.disease or 'Not specified'}

**Expression Data:**
- {feature_type} Name: {feature.display_name}
- Expression Change: {direction} by {abs(feature.log2_fold_change):.2f} log2 fold change
- Statistical Significance: adjusted p-value = {feature.padj:.2e}
- Biological Priority Score: {prioritized_feature.combined_score:.2f}

**Literature Context:**
{literature_context}

Please provide:
1. A comprehensive discussion of this {feature_type}'s role in {context.disease or 'the experimental condition'}
2. Interpretation of why this expression change might occur
3. Potential therapeutic implications
4. Key biological insights

Focus on connecting the expression change to the biological context and disease relevance.
        """.strip()
        
        return prompt
    
    def _parse_citations_from_response(self, response, source_papers: Dict[str, "Paper"]) -> List[CitationInfo]:
        """Parse citation information from Claude API response."""
        citation_info = []
        
        try:
            # Check if response has citations
            if hasattr(response.content[0], 'citations') and response.content[0].citations:
                for citation in response.content[0].citations:
                    source_id = citation.get('source_id', '')
                    if source_id in source_papers:
                        paper = source_papers[source_id]
                        citation_info.append(CitationInfo(
                            source_id=source_id,
                            quote=citation.get('quote', ''),
                            start_char=citation.get('start_char', 0),
                            end_char=citation.get('end_char', 0),
                            paper_citation=paper.citation,
                            paper_url=paper.pmc_url
                        ))
        except Exception as e:
            print(f"Warning: Could not parse citations from response: {e}")
        
        return citation_info
    
    def _extract_key_findings(self, discussion_text: str, feature: "OmicsFeature") -> List[str]:
        """Extract key findings from discussion text."""
        # Simple heuristic extraction
        findings = []
        
        direction = "upregulated" if feature.is_upregulated else "downregulated"
        findings.append(f"Significantly {direction} (padj = {feature.padj:.2e})")
        findings.append(f"Effect size: {abs(feature.log2_fold_change):.2f} log2FC")
        
        # Look for sentences that seem like key findings
        sentences = discussion_text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in ['important', 'key', 'critical', 'significant', 'suggests', 'indicates']):
                if len(sentence) < 150:  # Keep it concise
                    findings.append(sentence)
                    if len(findings) >= 5:  # Limit to 5 findings
                        break
        
        return findings
    
    def _extract_therapeutic_implications(self, discussion_text: str) -> Optional[str]:
        """Extract therapeutic implications from discussion text."""
        # Look for therapeutic-related content
        sentences = discussion_text.split('.')
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['therapeutic', 'treatment', 'drug', 'therapy', 'target']):
                return sentence.strip()
        
        return None
    
    def _create_fallback_discussion(
        self,
        prioritized_feature: PrioritizedFeature,
        context: OmicsExperimentContext
    ) -> FeatureDiscussion:
        """Create fallback discussion when synthesis fails."""
        feature = prioritized_feature.feature
        direction = "upregulated" if feature.is_upregulated else "downregulated"
        feature_type = context.get_feature_type_name()
        
        discussion_text = f"""
**{feature.display_name}** shows significant {direction} expression with a log2 fold change of {feature.log2_fold_change:.2f} 
(adjusted p-value: {feature.padj:.2e}) in the comparison of {context.treatment} vs {context.control}.

This {feature_type} has been prioritized with a combined score of {prioritized_feature.combined_score:.2f}, indicating 
potential biological significance in the context of {context.get_context_string()}.

The {direction} expression pattern suggests this {feature_type} may play a role in the biological response 
to {context.treatment}. Further experimental validation and literature analysis would be beneficial to understand 
the functional implications of this expression change.
        """.strip()
        
        key_findings = [
            f"Significant {direction} expression (padj < 0.05)",
            f"Effect size: {abs(feature.log2_fold_change):.2f} log2FC",
            f"Priority score: {prioritized_feature.combined_score:.2f}"
        ]
        
        return FeatureDiscussion(
            feature_id=feature.feature_id,
            feature_symbol=feature.feature_symbol,
            discussion_text=discussion_text,
            key_findings=key_findings,
            citations=[],
            citation_info=[],
            therapeutic_implications="Further analysis required for therapeutic insights."
        )
    
    async def generate_executive_summary(
        self,
        feature_discussions: List[FeatureDiscussion],
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any]
    ) -> str:
        """Generate executive summary from feature discussions."""
        if self.progress_callback:
            self.progress_callback("Generating executive summary...", 85)
        
        # Extract key themes from discussions
        all_findings = []
        for discussion in feature_discussions:
            all_findings.extend(discussion.key_findings)
        
        # Build summary prompt
        prompt = self._build_summary_prompt(context, feature_summary, omics_summary, all_findings)
        
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return self._create_fallback_summary(context, feature_summary, omics_summary)
    
    def _build_summary_prompt(
        self,
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
        all_findings: List[str]
    ) -> str:
        """Build executive summary prompt."""
        feature_type = context.get_feature_type_name()
        
        prompt = f"""
Please write an executive summary for this {context.omics_type.value} analysis:

**Analysis Overview:**
- Comparison: {context.treatment} vs {context.control}
- Tissue/Context: {context.tissue}
- Disease Context: {context.disease or 'Experimental condition'}
- Total {feature_type}: {feature_summary['total_features']:,}
- Significant {feature_type}: {feature_summary['significant_features']:,}
- Upregulated: {omics_summary['upregulated']}
- Downregulated: {omics_summary['downregulated']}

**Key Findings from Analysis:**
{chr(10).join(['- ' + finding for finding in all_findings[:10]])}

Please provide a concise executive summary (2-3 paragraphs) that:
1. Summarizes the key biological findings
2. Highlights the most important expression changes
3. Discusses potential implications for {context.disease or 'the experimental condition'}
4. Suggests next steps for validation or further research

Focus on the biological significance and clinical relevance.
        """.strip()
        
        return prompt
    
    def _create_fallback_summary(
        self,
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any]
    ) -> str:
        """Create fallback summary when synthesis fails."""
        feature_type = context.get_feature_type_name()
        
        return f"""
## Executive Summary

This {context.omics_type.value} analysis of {context.treatment} vs {context.control} in {context.tissue} samples 
identified {feature_summary['significant_features']} significantly altered {feature_type} out of {feature_summary['total_features']:,} total features analyzed.

The expression profile shows {omics_summary['upregulated']} upregulated and {omics_summary['downregulated']} downregulated {feature_type}, 
suggesting a substantial biological response to {context.treatment}. These changes may be relevant to {context.disease or 'the experimental condition'} 
and warrant further investigation.

The prioritized {feature_type} represent candidates for functional validation and potential therapeutic targeting. 
Detailed feature-by-feature analysis is provided below, along with literature-supported interpretations where available.
        """.strip()