"""Generate comprehensive reports from analysis results."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from ..synthesis.synthesizer import FeatureDiscussion
from ..prioritization.prioritizer import PrioritizedFeature
from ..parsers.omics_data import OmicsExperimentContext


class ReportGenerator:
    """Generate analysis reports in markdown format."""

    def __init__(self, output_dir: Path = Path("output")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_omics_report(
        self,
        executive_summary: str,
        feature_discussions: List[FeatureDiscussion],
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
        analysis_features: List[PrioritizedFeature],
        output_name: str = "omics_analysis_report",
    ) -> Path:
        """Generate comprehensive omics analysis report."""

        # Build report sections
        sections = []

        # Title and metadata
        sections.append(self._generate_header(context))

        # Table of contents
        sections.append(self._generate_toc())

        # Executive summary
        sections.append(f"# Executive Summary\n\n{executive_summary}")

        # Analysis overview
        sections.append(self._generate_overview(feature_summary, omics_summary, context))

        # Feature-by-feature discussions
        if feature_discussions:
            sections.append(self._generate_feature_discussions(feature_discussions, context))

        # Feature summary table
        if analysis_features:
            sections.append(self._generate_feature_summary_table(analysis_features, context))

        # Methods
        sections.append(self._generate_methods(context))

        # References
        sections.append(self._generate_references(feature_discussions))

        # Combine all sections
        full_report = "\n\n".join(sections)

        # Save report
        output_path = self.output_dir / f"{output_name}.md"
        with open(output_path, "w") as f:
            f.write(full_report)

        # Also save JSON metadata
        self._save_metadata(output_name, context, feature_summary, omics_summary, len(feature_discussions))

        return output_path

    def _generate_header(self, context: OmicsExperimentContext) -> str:
        """Generate report header."""
        date = datetime.now().strftime("%Y-%m-%d")

        header = f"""# {context.omics_type.value.title()} Analysis Report

**Generated**: {date}  
**Analysis**: {context.get_context_string()}  
**Organism**: {context.organism.title()}
**Omics Type**: {context.omics_type.value.title()}

---
"""
        return header

    def _generate_toc(self) -> str:
        """Generate table of contents."""
        return """## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Analysis Overview](#analysis-overview)
3. [Feature-by-Feature Analysis](#feature-by-feature-analysis)
4. [Feature Summary Table](#feature-summary-table)
5. [Methods](#methods)
6. [References](#references)

---
"""

    def _generate_overview(
        self, 
        feature_summary: Dict[str, Any], 
        omics_summary: Dict[str, Any],
        context: OmicsExperimentContext
    ) -> str:
        """Generate analysis overview."""
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

    def _generate_feature_discussions(
        self, 
        discussions: List[FeatureDiscussion], 
        context: OmicsExperimentContext
    ) -> str:
        """Generate feature-by-feature discussions."""
        feature_type = context.get_feature_type_name()
        content = [f"## {feature_type.title()}-by-{feature_type.title()} Analysis\n"]

        for i, discussion in enumerate(discussions, 1):
            feature_name = discussion.feature_symbol or discussion.feature_id
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

    def _generate_feature_summary_table(
        self, 
        analysis_features: List[PrioritizedFeature], 
        context: OmicsExperimentContext
    ) -> str:
        """Generate feature summary table."""
        feature_type = context.get_feature_type_name().title()
        content = [f"## {feature_type} Summary Table\n"]
        content.append(f"| {feature_type} Name | {feature_type} ID | log2FC | Adj P-value | Combined Score | Direction |")
        content.append("|-------------|---------|--------|-------------|----------------|-----------|")
        
        for prioritized_feature in analysis_features:
            feature = prioritized_feature.feature
            name = feature.display_name
            feature_id = feature.feature_id
            log2fc = f"{feature.log2_fold_change:.2f}"
            padj = f"{feature.padj:.2e}"
            score = f"{prioritized_feature.combined_score:.2f}"
            direction = "↑" if feature.is_upregulated else "↓"
            
            content.append(f"| {name} | {feature_id} | {log2fc} | {padj} | {score} | {direction} |")
        
        content.append("\n")
        return "\n".join(content)

    def _generate_methods(self, context: OmicsExperimentContext) -> str:
        """Generate methods section."""
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
- **Database**: PubMed Central (PMC) scientific literature database
- **Search Strategy**: {feature_type.title()}-specific queries combined with disease context
- **Synthesis**: AI-powered interpretation using Claude-4 language model

### Report Generation
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d")}
- **Software**: Multi-Omics Interpretation Pipeline v2.0
"""

    def _generate_references(self, discussions: List[FeatureDiscussion]) -> str:
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
        context: OmicsExperimentContext,
        feature_summary: Dict[str, Any],
        omics_summary: Dict[str, Any],
        n_discussions: int,
    ) -> None:
        """Save analysis metadata as JSON."""
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