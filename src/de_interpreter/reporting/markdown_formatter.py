"""Markdown formatting utilities."""

from typing import List, Dict, Any, Optional
import re


class MarkdownFormatter:
    """Format content for markdown reports."""

    @staticmethod
    def format_section(title: str, content: str, level: int = 2) -> str:
        """Format a section with title."""
        header = "#" * level
        return f"{header} {title}\n\n{content}"

    @staticmethod
    def format_table(headers: List[str], rows: List[List[Any]]) -> str:
        """Format data as markdown table."""
        # Convert all to strings
        headers = [str(h) for h in headers]
        rows = [[str(cell) for cell in row] for row in rows]

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        # Build table
        lines = []

        # Header
        header_cells = [h.ljust(w) for h, w in zip(headers, widths)]
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator
        sep_cells = ["-" * w for w in widths]
        lines.append("| " + " | ".join(sep_cells) + " |")

        # Rows
        for row in rows:
            row_cells = [cell.ljust(w) for cell, w in zip(row, widths)]
            lines.append("| " + " | ".join(row_cells) + " |")

        return "\n".join(lines)

    @staticmethod
    def format_gene_list(genes: List[Dict[str, Any]], max_genes: int = 10) -> str:
        """Format list of genes with stats."""
        lines = []

        for i, gene in enumerate(genes[:max_genes], 1):
            symbol = gene.get("symbol", gene.get("gene_id", "Unknown"))
            log2fc = gene.get("log2fc", 0)
            padj = gene.get("padj", 1)

            direction = "↑" if log2fc > 0 else "↓"

            lines.append(
                f"{i}. **{symbol}** {direction} (log2FC: {log2fc:.2f}, padj: {padj:.2e})"
            )

        return "\n".join(lines)

    @staticmethod
    def escape_markdown(text: str) -> str:
        """Escape special markdown characters."""
        special_chars = ["*", "_", "[", "]", "(", ")", "#", "+", "-", ".", "!"]
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    @staticmethod
    def format_citation(
        authors: List[str], year: int, title: str, journal: Optional[str] = None
    ) -> str:
        """Format a citation."""
        if len(authors) > 2:
            author_str = f"{authors[0]} et al."
        elif len(authors) == 2:
            author_str = f"{authors[0]} and {authors[1]}"
        else:
            author_str = authors[0] if authors else "Unknown"

        citation = f"{author_str} ({year}). {title}"
        if journal:
            citation += f". *{journal}*"

        return citation

    @staticmethod
    def create_anchor(text: str) -> str:
        """Create anchor link for heading."""
        # Convert to lowercase and replace spaces with hyphens
        anchor = text.lower()
        anchor = re.sub(r"[^a-z0-9\s-]", "", anchor)
        anchor = re.sub(r"\s+", "-", anchor)
        return anchor
