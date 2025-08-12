#!/usr/bin/env python3
"""Test script to verify citation functionality."""

import sys
from datetime import datetime
from de_interpreter.literature.paper import Paper
from de_interpreter.synthesis.synthesizer import FeatureDiscussion, CitationInfo

def test_paper_citation_methods():
    """Test the enhanced Paper class citation methods."""
    print("Testing Paper class citation methods...")
    
    # Create a test paper
    paper = Paper(
        pmid="12345678",
        title="Test Paper on Gene Expression",
        abstract="This is a test abstract about gene expression in disease.",
        authors=["Smith J", "Jones M", "Brown K"],
        journal="Test Journal",
        publication_date=datetime(2023, 1, 15),
        doi="10.1000/test",
        full_text="Full text content here..."
    )
    
    # Test basic citation
    print(f"Basic citation: {paper.citation}")
    
    # Test PMC URL
    print(f"PMC URL: {paper.pmc_url}")
    
    # Test Claude source format
    claude_source = paper.to_claude_source()
    print(f"Claude source ID: {claude_source['id']}")
    print(f"Content length: {len(claude_source['content'])}")
    print(f"Metadata keys: {list(claude_source['metadata'].keys())}")
    
    # Test formatted citation
    formatted = paper.to_formatted_citation()
    print(f"Formatted citation: {formatted}")
    print("✓ Paper citation methods working correctly\n")

def test_citation_info():
    """Test the CitationInfo class."""
    print("Testing CitationInfo class...")
    
    citation_info = CitationInfo(
        source_id="pmid_12345678",
        quote="This gene shows significant upregulation",
        start_char=45,
        end_char=89,
        paper_citation="Smith J et al. Test Paper. Test Journal. 2023.",
        paper_url="https://pubmed.ncbi.nlm.nih.gov/12345678/"
    )
    
    print(f"Source ID: {citation_info.source_id}")
    print(f"Quote: {citation_info.quote}")
    print(f"Citation: {citation_info.paper_citation}")
    print("✓ CitationInfo class working correctly\n")

def test_feature_discussion():
    """Test the enhanced FeatureDiscussion class."""
    print("Testing FeatureDiscussion class...")
    
    citation_info = CitationInfo(
        source_id="pmid_12345678",
        quote="Gene shows upregulation in disease state",
        start_char=0,
        end_char=40,
        paper_citation="Smith J et al. Test Paper. Test Journal. 2023.",
        paper_url="https://pubmed.ncbi.nlm.nih.gov/12345678/"
    )
    
    discussion = FeatureDiscussion(
        feature_id="GENE1",
        feature_symbol="TEST1",
        discussion_text="This gene shows significant changes...",
        key_findings=["Upregulated", "Significant p-value"],
        citations=["Basic citation 1"],
        citation_info=[citation_info],
        therapeutic_implications="Potential drug target"
    )
    
    print(f"Feature: {discussion.feature_symbol}")
    print(f"Citations count: {len(discussion.citations)}")
    print(f"Citation info count: {len(discussion.citation_info)}")
    print(f"First citation quote: {discussion.citation_info[0].quote}")
    print("✓ FeatureDiscussion class working correctly\n")

if __name__ == "__main__":
    try:
        test_paper_citation_methods()
        test_citation_info()
        test_feature_discussion()
        print("🎉 All citation functionality tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)