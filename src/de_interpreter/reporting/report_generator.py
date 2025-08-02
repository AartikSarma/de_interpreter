"""Generate comprehensive reports from analysis results."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from ..synthesis import GeneDiscussion
from ..prioritization import PrioritizedGene
from ..prioritization.omics_prioritizer import PrioritizedOmicsFeature
from ..parsers import ExperimentalContext
from ..parsers.omics_data import OmicsExperimentContext
from .markdown_formatter import MarkdownFormatter


class ReportGenerator:
    """Generate analysis reports in various formats."""

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.formatter = MarkdownFormatter()

    def generate_report(
        self,
        executive_summary: str,
        gene_discussions: List[GeneDiscussion],
        context: ExperimentalContext,
        de_summary: Dict[str, Any],
        analysis_genes: Optional[List[PrioritizedGene]] = None,
        output_name: str = "de_analysis_report",
    ) -> Path:
        """Generate comprehensive markdown report."""

        # Build report sections
        sections = []

        # Title and metadata
        sections.append(self._generate_header(context))

        # Table of contents
        sections.append(self._generate_toc())

        # Executive summary
        sections.append(
            self.formatter.format_section(
                "Executive Summary", executive_summary, level=1
            )
        )

        # Analysis overview
        sections.append(self._generate_overview(de_summary, context))

        # Gene-by-gene discussions
        if gene_discussions:
            sections.append(self._generate_gene_discussions(gene_discussions))

        # Gene summary table
        if analysis_genes:
            sections.append(self._generate_gene_summary_table(analysis_genes))

        # Methods
        sections.append(self._generate_methods())

        # References
        sections.append(self._generate_references(gene_discussions))

        # Combine all sections
        full_report = "\n\n".join(sections)

        # Save report
        output_path = self.output_dir / f"{output_name}.md"
        with open(output_path, "w") as f:
            f.write(full_report)

        # Also save JSON metadata
        self._save_metadata(output_name, context, de_summary, len(gene_discussions))

        return output_path

    def _generate_header(self, context: ExperimentalContext) -> str:
        """Generate report header."""
        date = datetime.now().strftime("%Y-%m-%d")

        header = f"""# Differential Expression Analysis Report

**Generated**: {date}  
**Analysis**: {context.get_context_string()}  
**Organism**: {context.organism.title()}

---
"""
        return header

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        return """## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Analysis Overview](#analysis-overview)
3. [Gene-by-Gene Analysis](#gene-by-gene-analysis)
4. [Gene Summary Table](#gene-summary-table)
5. [Methods](#methods)
6. [References](#references)

---
"""

    def _generate_overview(
        self, de_summary: Dict[str, Any], context: ExperimentalContext
    ) -> str:
        """Generate analysis overview section."""
        content = f"""## Analysis Overview

### Differential Expression Summary

- **Total genes analyzed**: {de_summary.get('total_genes', 'N/A'):,}
- **Significantly differentially expressed**: {de_summary.get('significant_genes', 'N/A'):,}
  - Upregulated: {de_summary.get('upregulated', 'N/A'):,}
  - Downregulated: {de_summary.get('downregulated', 'N/A'):,}
- **Fold change range**: {de_summary.get('min_log2fc', 'N/A'):.2f} to {de_summary.get('max_log2fc', 'N/A'):.2f} (log2)
- **Median adjusted p-value**: {de_summary.get('median_padj', 'N/A'):.2e}

### Experimental Design

{self._format_experimental_design(context)}
"""
        return content

    def _format_experimental_design(self, context: ExperimentalContext) -> str:
        """Format experimental design details."""
        lines = []

        if context.treatment and context.control:
            lines.append(f"- **Comparison**: {context.treatment} vs {context.control}")

        lines.append(f"- **Disease/Condition**: {context.disease}")

        if context.tissue:
            lines.append(f"- **Tissue**: {context.tissue}")

        if context.cell_type:
            lines.append(f"- **Cell Type**: {context.cell_type}")

        if context.time_point:
            lines.append(f"- **Time Point**: {context.time_point}")

        if context.sample_size:
            sizes = [f"{group}: n={n}" for group, n in context.sample_size.items()]
            lines.append(f"- **Sample Size**: {', '.join(sizes)}")

        return "\n".join(lines)

    def _generate_gene_discussions(self, discussions: List[GeneDiscussion]) -> str:
        """Generate gene-by-gene analysis section."""
        content = ["## Gene-by-Gene Analysis\n"]

        # Group by confidence/priority
        high_conf = [d for d in discussions if d.confidence_score > 0.7]
        med_conf = [d for d in discussions if 0.3 < d.confidence_score <= 0.7]

        if high_conf:
            content.append("### High-Confidence Findings\n")
            for disc in high_conf[:20]:  # Limit to top 20
                content.append(self._format_gene_discussion(disc))

        if med_conf:
            content.append("### Additional Findings\n")
            for disc in med_conf[:10]:  # Limit to 10
                content.append(self._format_gene_discussion(disc, detailed=False))

        return "\n".join(content)

    def _format_gene_discussion(
        self, discussion: GeneDiscussion, detailed: bool = True
    ) -> str:
        """Format individual gene discussion."""
        gene_name = discussion.gene_symbol or discussion.gene_id

        if detailed:
            sections = [f"#### {gene_name}\n"]
            sections.append(discussion.discussion_text + "\n")

            if discussion.key_findings:
                sections.append("**Key Findings:**")
                for finding in discussion.key_findings:
                    sections.append(f"- {finding}")
                sections.append("")

            if discussion.therapeutic_implications:
                sections.append(
                    f"**Therapeutic Implications:** {discussion.therapeutic_implications}\n"
                )

            if discussion.citations:
                sections.append(
                    "**References:** " + "; ".join(discussion.citations[:3]) + "\n"
                )

            sections.append("---\n")

            return "\n".join(sections)
        else:
            # Condensed format
            summary = f"**{gene_name}**: {discussion.discussion_text[:200]}..."
            if discussion.key_findings:
                summary += f" Key finding: {discussion.key_findings[0]}"
            return summary + "\n"

    def _generate_gene_summary_table(self, analysis_genes: List[PrioritizedGene]) -> str:
        """Generate gene summary table section."""
        content = ["## Gene Summary Table\n"]
        content.append("| Gene Symbol | Gene ID | log2FC | Adj P-value | Combined Score | Direction |")
        content.append("|-------------|---------|--------|-------------|----------------|-----------|")
        
        for gene in analysis_genes:
            symbol = gene.gene_symbol or "N/A"
            gene_id = gene.gene_id
            log2fc = f"{gene.de_result.log2_fold_change:.2f}"
            padj = f"{gene.de_result.padj:.2e}"
            score = f"{gene.combined_score:.2f}"
            direction = "↑" if gene.de_result.is_upregulated else "↓"
            
            content.append(f"| {symbol} | {gene_id} | {log2fc} | {padj} | {score} | {direction} |")
        
        content.append("\n")
        return "\n".join(content)

    def _generate_methods(self) -> str:
        """Generate methods section."""
        return """## Methods

### Differential Expression Analysis
Results were filtered using adjusted p-value < 0.05 and |log2 fold change| > 1.0.

### Gene Prioritization
Genes were prioritized based on:
1. Statistical significance (-log10 adjusted p-value)
2. Effect size (absolute log2 fold change)
3. Expression level (base mean)
4. Biological relevance to disease context

### Literature Mining
Relevant literature was retrieved using the FutureHouse Paper Search API, focusing on papers from the last 10 years that mention both the gene and disease context.

### Synthesis
Gene discussions were generated using Claude-3 Opus, integrating differential expression results with current literature to provide context-specific interpretations.

### Clustering
Genes were clustered based on expression patterns using hierarchical clustering with Ward linkage.
"""

    def _generate_references(self, discussions: List[GeneDiscussion]) -> str:
        """Generate references section."""
        all_citations = set()

        for disc in discussions:
            all_citations.update(disc.citations)

        content = ["## References\n"]

        for i, citation in enumerate(sorted(all_citations)[:50], 1):  # Limit to 50
            content.append(f"{i}. {citation}")

        return "\n".join(content)

    def _save_metadata(
        self,
        output_name: str,
        context: ExperimentalContext,
        de_summary: Dict[str, Any],
        n_discussions: int,
    ) -> None:
        """Save analysis metadata as JSON."""
        metadata = {
            "analysis_date": datetime.now().isoformat(),
            "output_name": output_name,
            "context": {
                "disease": context.disease,
                "tissue": context.tissue,
                "cell_type": context.cell_type,
                "treatment": context.treatment,
                "organism": context.organism,
            },
            "summary_stats": de_summary,
            "n_gene_discussions": n_discussions,
        }

        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def generate_omics_report(
        self,
        executive_summary: str,
        feature_discussions: List[GeneDiscussion],
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
        analysis_features: Optional[List[PrioritizedOmicsFeature]] = None,
        output_name: str = "omics_analysis_report",
    ) -> Path:
        """Generate comprehensive omics analysis report."""

        # Build report sections
        sections = []

        # Title and metadata
        sections.append(self._generate_omics_header(context))

        # Table of contents
        sections.append(self._generate_omics_toc())

        # Executive summary
        sections.append(
            self.formatter.format_section(
                "Executive Summary", executive_summary, level=1
            )
        )

        # Analysis overview
        sections.append(self._generate_omics_overview(feature_summary, omics_summary, context))

        # Feature-by-feature discussions
        if feature_discussions:
            sections.append(self._generate_omics_feature_discussions(feature_discussions, context))

        # Feature summary table
        if analysis_features:
            sections.append(self._generate_omics_feature_summary_table(analysis_features, context))

        # Methods
        sections.append(self._generate_omics_methods(context))

        # References
        sections.append(self._generate_references(feature_discussions))

        # Combine all sections
        full_report = "\n\n".join(sections)

        # Save report
        output_path = self.output_dir / f"{output_name}.md"
        with open(output_path, "w") as f:
            f.write(full_report)

        # Also save JSON metadata
        self._save_omics_metadata(output_name, context, feature_summary, omics_summary, len(feature_discussions))

        return output_path

    def _generate_omics_header(self, context: OmicsExperimentContext) -> str:
        """Generate omics report header."""
        date = datetime.now().strftime("%Y-%m-%d")

        header = f"""# {context.omics_type.value.title()} Analysis Report

**Generated**: {date}  
**Analysis**: {context.get_context_string()}  
**Organism**: {context.organism.title()}
**Omics Type**: {context.omics_type.value.title()}

---
"""
        return header

    def _generate_omics_toc(self) -> str:
        """Generate omics table of contents."""
        return """## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Analysis Overview](#analysis-overview)
3. [Feature-by-Feature Analysis](#feature-by-feature-analysis)
4. [Feature Summary Table](#feature-summary-table)
5. [Methods](#methods)
6. [References](#references)

---
"""

    def _generate_omics_overview(
        self, 
        feature_summary: Dict[str, Any], 
        omics_summary: Dict[str, Any],
        context: OmicsExperimentContext
    ) -> str:
        """Generate omics analysis overview."""
        feature_type = omics_summary.get('feature_type_name', 'features')
        
        content = f"""## Analysis Overview

### Dataset Summary
- **Omics Type**: {context.omics_type.value.title()}
- **Total {feature_type}**: {feature_summary.get('total_features', 'N/A'):,}
- **Significant {feature_type}**: {feature_summary.get('significant_features', 'N/A'):,}
- **Analysis Platform**: {context.platform or 'Not specified'}
- **Analysis Method**: {context.analysis_method or 'Standard differential analysis'}

### Statistical Overview
- **Upregulated {feature_type}**: {omics_summary.get('upregulated', 'N/A')}
- **Downregulated {feature_type}**: {omics_summary.get('downregulated', 'N/A')}
- **Mean log2 fold change**: {omics_summary.get('mean_log2fc', 'N/A'):.2f}
- **Maximum |log2FC|**: {omics_summary.get('max_abs_log2fc', 'N/A'):.2f}
- **Minimum adjusted p-value**: {omics_summary.get('min_padj', 'N/A'):.2e}

### Experimental Design
- **Condition**: {context.treatment} vs {context.control}
- **Tissue/Sample**: {context.tissue}
- **Cell Type**: {context.cell_type}
- **Organism**: {context.organism}
"""
        
        if context.sample_size:
            content += f"- **Sample Size**: {context.sample_size.get('treatment', 'N/A')} treatment, {context.sample_size.get('control', 'N/A')} control\n"
        
        if context.time_point:
            content += f"- **Time Point**: {context.time_point}\n"

        return content

    def _generate_omics_feature_discussions(
        self, 
        discussions: List[GeneDiscussion], 
        context: OmicsExperimentContext
    ) -> str:
        """Generate feature-by-feature discussions for omics."""
        feature_type = context.get_feature_type_name()
        content = [f"## {feature_type.title()}-by-{feature_type.title()} Analysis\n"]

        for i, discussion in enumerate(discussions, 1):
            feature_name = discussion.gene_symbol or discussion.gene_id
            content.append(f"### {i}. {feature_name}\n")
            content.append(discussion.discussion_text)
            
            if discussion.key_findings:
                content.append("\n**Key Findings:**")
                for finding in discussion.key_findings:
                    content.append(f"- {finding}")
            
            if discussion.therapeutic_implications:
                content.append(f"\n**Therapeutic Implications:** {discussion.therapeutic_implications}")
            
            content.append("\n---\n")

        return "\n".join(content)

    def _generate_omics_feature_summary_table(
        self, 
        analysis_features: List[PrioritizedOmicsFeature], 
        context: OmicsExperimentContext
    ) -> str:
        """Generate omics feature summary table."""
        feature_type = context.get_feature_type_name().title()
        content = [f"## {feature_type} Summary Table\n"]
        content.append(f"| {feature_type} Name | {feature_type} ID | log2FC | Adj P-value | Combined Score | Direction |")
        content.append("|-------------|---------|--------|-------------|----------------|-----------|")
        
        for feature in analysis_features:
            name = feature.display_name
            feature_id = feature.feature_id
            log2fc = f"{feature.omics_feature.log2_fold_change:.2f}"
            padj = f"{feature.omics_feature.padj:.2e}"
            score = f"{feature.combined_score:.2f}"
            direction = "↑" if feature.omics_feature.is_upregulated else "↓"
            
            content.append(f"| {name} | {feature_id} | {log2fc} | {padj} | {score} | {direction} |")
        
        content.append("\n")
        return "\n".join(content)

    def _generate_omics_methods(self, context: OmicsExperimentContext) -> str:
        """Generate omics methods section."""
        feature_type = context.get_feature_type_name()
        
        return f"""## Methods

### Data Processing
- **Omics Type**: {context.omics_type.value.title()}
- **Platform**: {context.platform or 'Not specified'}
- **Analysis Pipeline**: {context.analysis_method or 'Standard differential analysis'}
- **Normalization**: {context.normalization or 'Standard normalization'}

### Statistical Analysis
- **Differential Analysis**: Statistical testing for {feature_type} abundance changes
- **Multiple Testing Correction**: Benjamini-Hochberg FDR correction
- **Significance Threshold**: Adjusted p-value < 0.05
- **Effect Size Threshold**: |log2 fold change| > 1.0

### Feature Prioritization
- **Statistical Score**: Based on adjusted p-value and effect size
- **Biological Score**: Based on known disease associations and functional annotations
- **Combined Score**: Weighted combination of statistical and biological scores

### Literature Mining
- **Database**: FutureHouse scientific literature database
- **Search Strategy**: {feature_type.title()}-specific queries combined with disease context
- **Synthesis**: AI-powered interpretation using Claude-4 language model

### Report Generation
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Software**: DE Interpretation Pipeline (Multi-omics version)
"""

    def _save_omics_metadata(
        self,
        output_name: str,
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
        n_discussions: int,
    ) -> None:
        """Save omics analysis metadata as JSON."""
        metadata = {
            "analysis_date": datetime.now().isoformat(),
            "output_name": output_name,
            "omics_type": context.omics_type.value,
            "context": {
                "disease": context.disease,
                "tissue": context.tissue,
                "cell_type": context.cell_type,
                "treatment": context.treatment,
                "control": context.control,
                "organism": context.organism,
                "platform": context.platform,
                "analysis_method": context.analysis_method,
                "normalization": context.normalization,
            },
            "feature_summary": feature_summary,
            "omics_summary": omics_summary,
            "n_feature_discussions": n_discussions,
        }

        metadata_path = self.output_dir / f"{output_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
