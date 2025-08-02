"""Claude API integration for gene discussion synthesis."""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import asyncio
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from ..prioritization import PrioritizedGene, GeneCluster
from ..prioritization.omics_prioritizer import PrioritizedOmicsFeature
from ..parsers import ExperimentalContext
from ..parsers.omics_data import OmicsExperimentContext
from ..literature import Paper
from .prompts import PromptBuilder


@dataclass
class GeneDiscussion:
    """Container for synthesized gene discussion."""

    gene_id: str
    gene_symbol: Optional[str]
    discussion_text: str
    key_findings: List[str]
    therapeutic_implications: Optional[str]
    citations: List[str]
    confidence_score: float


class ClaudeSynthesizer:
    """Synthesize gene discussions using Claude API."""

    def __init__(
        self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-20250514"
    ):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = AsyncAnthropic(api_key=self.api_key)
        self.model = model
        self.prompt_builder = PromptBuilder()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # AsyncAnthropic client handles cleanup automatically
        pass

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def synthesize_gene_discussion(
        self, gene: PrioritizedGene, context: ExperimentalContext, papers: List[Paper]
    ) -> GeneDiscussion:
        """Generate discussion for a single gene."""
        # Build prompt
        prompt = self.prompt_builder.build_gene_prompt(gene, context, papers)

        try:
            # Call Claude API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.7,
                system=self.prompt_builder.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            discussion = self._parse_response(response.content[0].text, gene)
            return discussion

        except Exception as e:
            print(
                f"Error synthesizing discussion for {gene.gene_symbol or gene.gene_id}: {e}"
            )
            # Return minimal discussion on error
            return GeneDiscussion(
                gene_id=gene.gene_id,
                gene_symbol=gene.gene_symbol,
                discussion_text=f"Unable to generate discussion for {gene.gene_symbol or gene.gene_id}.",
                key_findings=[],
                therapeutic_implications=None,
                citations=[],
                confidence_score=0.0,
            )

    async def synthesize_cluster_discussion(
        self,
        cluster: GeneCluster,
        context: ExperimentalContext,
        gene_papers: Dict[str, List[Paper]],
    ) -> str:
        """Generate discussion for a gene cluster."""
        # Get top genes from cluster
        top_genes = cluster.get_top_genes(n=5)

        # Build cluster prompt
        prompt = self.prompt_builder.build_cluster_prompt(cluster, context, gene_papers)

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=3000,
                temperature=0.7,
                system=self.prompt_builder.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            print(f"Error synthesizing cluster discussion: {e}")
            return f"Unable to generate discussion for cluster {cluster.cluster_id}."

    async def batch_synthesize(
        self,
        genes: List[PrioritizedGene],
        context: ExperimentalContext,
        gene_papers: Dict[str, List[Paper]],
        max_concurrent: int = 5,
    ) -> List[GeneDiscussion]:
        """Synthesize discussions for multiple genes in parallel."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def synthesize_with_semaphore(gene: PrioritizedGene):
            async with semaphore:
                papers = gene_papers.get(gene.gene_symbol or gene.gene_id, [])
                return await self.synthesize_gene_discussion(gene, context, papers)

        tasks = [synthesize_with_semaphore(gene) for gene in genes]
        return await asyncio.gather(*tasks)

    def _parse_response(
        self, response_text: str, gene: PrioritizedGene
    ) -> GeneDiscussion:
        """Parse Claude's response into structured discussion."""
        # This is a simplified parser - in production would use more sophisticated parsing

        # Extract sections using simple heuristics
        lines = response_text.strip().split("\n")

        # Find main discussion
        discussion_text = ""
        key_findings = []
        therapeutic_implications = ""
        citations = []

        current_section = "main"

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect section headers
            if "key finding" in line.lower() or "main finding" in line.lower():
                current_section = "findings"
                continue
            elif "therapeutic" in line.lower() or "clinical" in line.lower():
                current_section = "therapeutic"
                continue
            elif "reference" in line.lower() or "citation" in line.lower():
                current_section = "citations"
                continue

            # Add content to appropriate section
            if current_section == "main":
                discussion_text += line + " "
            elif current_section == "findings" and line.startswith(
                ("•", "-", "*", "1", "2", "3")
            ):
                key_findings.append(line.lstrip("•-* 123456789."))
            elif current_section == "therapeutic":
                therapeutic_implications += line + " "
            elif current_section == "citations":
                citations.append(line)

        # Clean up
        discussion_text = discussion_text.strip()
        therapeutic_implications = therapeutic_implications.strip() or None

        # Simple confidence scoring based on content
        confidence_score = min(
            1.0, len(discussion_text) / 500 + len(key_findings) * 0.1
        )

        return GeneDiscussion(
            gene_id=gene.gene_id,
            gene_symbol=gene.gene_symbol,
            discussion_text=discussion_text,
            key_findings=key_findings[:5],  # Limit to 5
            therapeutic_implications=therapeutic_implications,
            citations=citations,
            confidence_score=confidence_score,
        )

    def _parse_omics_response(
        self, response_text: str, feature: PrioritizedOmicsFeature
    ) -> GeneDiscussion:
        """Parse Claude's response into structured discussion for omics features."""
        # This is similar to _parse_response but adapted for omics features
        
        # Extract sections using simple heuristics
        lines = response_text.strip().split("\n")

        # Find main discussion
        discussion_text = ""
        key_findings = []
        therapeutic_implications = ""
        citations = []

        current_section = "main"

        for line in lines:
            line = line.strip()

            if not line:
                continue

            # Detect section headers
            if "key finding" in line.lower() or "main finding" in line.lower():
                current_section = "findings"
                continue
            elif "therapeutic" in line.lower() or "clinical" in line.lower():
                current_section = "therapeutic"
                continue
            elif "reference" in line.lower() or "citation" in line.lower():
                current_section = "citations"
                continue

            # Add content to appropriate section
            if current_section == "main":
                discussion_text += line + " "
            elif current_section == "findings" and line.startswith(
                ("•", "-", "*", "1", "2", "3")
            ):
                key_findings.append(line.lstrip("•-* 123456789."))
            elif current_section == "therapeutic":
                therapeutic_implications += line + " "
            elif current_section == "citations":
                citations.append(line)

        # Clean up
        discussion_text = discussion_text.strip()
        therapeutic_implications = therapeutic_implications.strip() or None

        # Simple confidence scoring based on content
        confidence_score = min(
            1.0, len(discussion_text) / 500 + len(key_findings) * 0.1
        )

        return GeneDiscussion(
            gene_id=feature.feature_id,
            gene_symbol=feature.feature_symbol,
            discussion_text=discussion_text,
            key_findings=key_findings[:5],  # Limit to 5
            therapeutic_implications=therapeutic_implications,
            citations=citations,
            confidence_score=confidence_score,
        )

    async def generate_executive_summary(
        self,
        discussions: List[GeneDiscussion],
        context: ExperimentalContext,
        de_summary: Dict[str, Any],
    ) -> str:
        """Generate executive summary of all findings."""
        prompt = self.prompt_builder.build_summary_prompt(
            discussions, context, de_summary
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.7,
                system=self.prompt_builder.get_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            print(f"Error generating executive summary: {e}")
            return "Unable to generate executive summary."

    async def batch_synthesize_omics(
        self,
        prioritized_features: List[PrioritizedOmicsFeature],
        context: OmicsExperimentContext,
        feature_papers: Dict[str, List[Paper]],
    ) -> List[GeneDiscussion]:
        """Synthesize discussions for prioritized omics features."""
        discussions = []
        
        for feature in prioritized_features:
            feature_key = feature.display_name
            papers = feature_papers.get(feature_key, [])
            
            discussion = await self.synthesize_omics_feature_discussion(
                feature, context, papers
            )
            discussions.append(discussion)
            
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.1)
        
        return discussions

    async def synthesize_omics_feature_discussion(
        self,
        feature: PrioritizedOmicsFeature,
        context: OmicsExperimentContext,
        papers: List[Paper],
    ) -> GeneDiscussion:
        """Synthesize discussion for a single omics feature."""
        prompt = self.prompt_builder.build_omics_feature_prompt(
            feature, context, papers
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.6,
                system=self.prompt_builder.get_omics_system_prompt(context.omics_type),
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse the response
            discussion = self._parse_omics_response(response_text, feature)
            discussion_text = discussion.discussion_text
            key_findings = discussion.key_findings
            therapeutic_implications = discussion.therapeutic_implications
            citations = discussion.citations
            confidence_score = discussion.confidence_score

        except Exception as e:
            print(f"Error synthesizing discussion for {feature.display_name}: {e}")
            discussion_text = f"Unable to generate discussion for {feature.display_name}."
            key_findings = []
            therapeutic_implications = None
            citations = []
            confidence_score = 0.0

        return GeneDiscussion(
            gene_id=feature.feature_id,
            gene_symbol=feature.feature_symbol,
            discussion_text=discussion_text,
            key_findings=key_findings,
            therapeutic_implications=therapeutic_implications,
            citations=citations,
            confidence_score=confidence_score,
        )

    async def generate_omics_executive_summary(
        self,
        discussions: List[GeneDiscussion],
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
    ) -> str:
        """Generate executive summary for omics analysis."""
        prompt = self.prompt_builder.build_omics_summary_prompt(
            discussions, context, feature_summary, omics_summary
        )

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.7,
                system=self.prompt_builder.get_omics_system_prompt(context.omics_type),
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text

        except Exception as e:
            print(f"Error generating omics executive summary: {e}")
            return "Unable to generate executive summary."
