#!/usr/bin/env python3
"""
Simplest PaperQA2 Implementation

Following the most basic pattern from PaperQA2 documentation.
"""

import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
import json
from datetime import datetime

# Load environment variables
load_dotenv()

from pubmed_retriever import PubMedRetriever
from paperqa import Settings, ask


def convert_xml_to_text(xml_file_path):
    """
    Convert PMC XML file to plain text file for PaperQA2 processing.
    
    Args:
        xml_file_path: Path to the XML file
        
    Returns:
        Path to the created text file
    """
    # Read XML content
    with open(xml_file_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Parse XML
    soup = BeautifulSoup(xml_content, 'lxml-xml')
    
    # Extract text content
    text_parts = []
    
    # Title
    title = soup.find('article-title')
    if title:
        text_parts.append(f"Title: {title.get_text()}\n")
    
    # Authors
    authors = soup.find_all('contrib', {'contrib-type': 'author'})
    if authors:
        author_names = []
        for author in authors:
            surname = author.find('surname')
            given_names = author.find('given-names')
            if surname:
                name = surname.get_text()
                if given_names:
                    name = f"{given_names.get_text()} {name}"
                author_names.append(name)
        if author_names:
            text_parts.append(f"Authors: {', '.join(author_names)}\n")
    
    # Abstract
    abstract = soup.find('abstract')
    if abstract:
        # Remove any nested tags like <p> but keep the text
        abstract_text = ' '.join(p.get_text() for p in abstract.find_all('p'))
        if not abstract_text:
            abstract_text = abstract.get_text()
        text_parts.append(f"\nAbstract:\n{abstract_text}\n")
    
    # Body sections
    body = soup.find('body')
    if body:
        text_parts.append("\nMain Text:\n")
        
        # Process all sections
        for section in body.find_all('sec'):
            # Section title
            title = section.find('title')
            if title:
                text_parts.append(f"\n{title.get_text()}\n")
            
            # Section paragraphs
            for p in section.find_all('p', recursive=False):
                text_parts.append(p.get_text() + "\n")
    
    # Join all parts
    full_text = '\n'.join(text_parts)
    
    # Clean up excessive whitespace
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    full_text = re.sub(r' {2,}', ' ', full_text)
    
    # Save as text file
    text_file_path = xml_file_path.replace('.xml', '.txt')
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    return text_file_path


async def main():
    """Main function to run PaperQA2 analysis on ARDS papers."""
    
    # Step 1: Download papers from PubMed
    print("\nStep 1: Downloading papers from PubMed")
    print("-" * 50)
    
    # Create retriever
    retriever = PubMedRetriever(output_dir="simple_papers")
    
    # Search for ARDS papers
    query = "ARDS molecular phenotypes endotypes precision medicine"
    pmc_ids = retriever.search_pubmed(query, max_results=6)
    
    if not pmc_ids:
        print("No papers found")
        return
    
    print(f"Found {len(pmc_ids)} papers to download")
    
    output_dir = Path("simple_papers")
    
    # Download papers
    downloaded = 0
    for pmc_id in pmc_ids:
        info = retriever.get_pmc_info(pmc_id)
        if info:
            print(f"\nDownloading: {info['title'][:60]}...")
            file_path = retriever.download_fulltext(pmc_id, info['title'])
            if file_path:
                print(f"✓ Saved to: {Path(file_path).name}")
                downloaded += 1
    
    print(f"\nTotal downloaded: {downloaded} papers")
    
    # Convert XML files to text files
    print("\nConverting XML files to text...")
    for file in output_dir.glob("*.xml"):
        text_file = convert_xml_to_text(str(file))
        print(f"Converted {file.name} -> {Path(text_file).name}")
    
    # Step 2: Use PaperQA2's ask() function directly
    print("\n\nStep 2: Analyzing with PaperQA2")
    print("-" * 50)
    
    question = "What are the key molecular phenotypes and endotypes of ARDS described in these papers? How do they differ in terms of inflammatory markers and clinical outcomes?"
    
    # Create settings with Claude
    settings = Settings(
        llm="claude-3-5-sonnet-20241022",
        summary_llm="claude-3-5-sonnet-20241022",
        paper_directory=str(output_dir),
        answer_max_sources=3,
        evidence_k=5
    )
    
    print(f"Question: {question}")
    print(f"Paper directory: {output_dir}")
    print(f"Model: claude-3-5-sonnet-20241022")
    print("\nCalling ask()...")
    
    try:
        # Use the simplest form of ask()
        answer_response = await ask(
            question,
            settings=settings
        )
        
        # Display results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        if hasattr(answer_response, 'session'):
            session = answer_response.session
            print(f"\nAnswer:\n{session.answer}")
            print(f"\nFormatted Answer:\n{session.formatted_answer}")
            print(f"\nCost: ${session.cost:.4f}")
            print(f"Contexts used: {len(session.contexts) if session.contexts else 0}")
            
            # Save results to JSON file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = output_dir / f"ards_analysis_results_{timestamp}.json"
            
            results = {
                "timestamp": timestamp,
                "question": question,
                "answer": session.answer,
                "formatted_answer": session.formatted_answer,
                "cost": session.cost,
                "contexts_used": len(session.contexts) if session.contexts else 0,
                "contexts": [{
                    "text": ctx.context,
                    "score": ctx.score if hasattr(ctx, 'score') else 0,
                    "citation": str(ctx.citation) if hasattr(ctx, 'citation') else ""
                } for ctx in (session.contexts or [])]
            }
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nResults saved to: {results_file}")
        else:
            print(f"\nResponse: {answer_response}")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        
        # Save error results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"ards_analysis_error_{timestamp}.json"
        
        error_results = {
            "timestamp": timestamp,
            "question": question,
            "error": str(e),
            "status": "failed"
        }
        
        with open(results_file, 'w') as f:
            json.dump(error_results, f, indent=2)
        
        print(f"Error details saved to: {results_file}")


if __name__ == "__main__":
    print("Simplest PaperQA2 Implementation")
    print("Following basic documentation pattern")
    print("="*60)
    
    asyncio.run(main())