"""MeSH term enhancement for literature queries using Claude."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Optional Claude import
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


@dataclass
class MeshEnhancedQuery:
    """Container for MeSH-enhanced query information."""
    original_query: str
    mesh_terms: List[str]
    enhanced_query: str
    enhancement_method: str = "claude"


class MeshQueryEnhancer:
    """Enhance literature queries with MeSH terms using Claude."""
    
    def __init__(self, api_key: Optional[str] = None, enable_mesh: bool = False):
        self.api_key = api_key
        self.enable_mesh = enable_mesh and api_key and CLAUDE_AVAILABLE
        self.client = None
        
        if self.enable_mesh:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                logging.warning(f"Failed to initialize Claude client: {e}")
                self.enable_mesh = False
    
    def is_available(self) -> bool:
        """Check if MeSH enhancement is available."""
        return self.enable_mesh and self.client is not None
    
    async def enhance_query(
        self, 
        query: str, 
        n_terms: int = 3,
        progress_callback: Optional[callable] = None
    ) -> MeshEnhancedQuery:
        """Enhance a query with MeSH terms using Claude."""
        
        if not self.is_available():
            return MeshEnhancedQuery(
                original_query=query,
                mesh_terms=[],
                enhanced_query=query,
                enhancement_method="none"
            )
        
        if progress_callback:
            progress_callback(f"Generating {n_terms} MeSH terms for query enhancement", 5)
        
        try:
            # Generate MeSH terms using Claude
            mesh_terms = await self._generate_mesh_terms(query, n_terms)
            
            if progress_callback:
                progress_callback(f"Generated {len(mesh_terms)} MeSH terms", 10)
            
            # Create enhanced query
            enhanced_query = self._create_enhanced_query(query, mesh_terms)
            
            return MeshEnhancedQuery(
                original_query=query,
                mesh_terms=mesh_terms,
                enhanced_query=enhanced_query,
                enhancement_method="claude"
            )
            
        except Exception as e:
            logging.error(f"Error generating MeSH terms: {e}")
            return MeshEnhancedQuery(
                original_query=query,
                mesh_terms=[],
                enhanced_query=query,
                enhancement_method="failed"
            )
    
    async def _generate_mesh_terms(self, query: str, n_terms: int) -> List[str]:
        """Generate MeSH terms using Claude Haiku."""
        
        prompt = f"""You are a medical literature search expert. Given the following biomedical query, provide exactly {n_terms} relevant MeSH (Medical Subject Headings) terms that would improve PubMed search results.

Query: "{query}"

Please respond with exactly {n_terms} MeSH terms, one per line, without any additional explanation or formatting. Use the exact MeSH terminology as it appears in PubMed.

Example format:
Neoplasms
Cell Death
Gene Expression Regulation"""

        try:
            message = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,  # Lower temperature for more consistent terminology
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse the response to extract MeSH terms
            response_text = message.content[0].text.strip()
            mesh_terms = [term.strip() for term in response_text.split('\n') if term.strip()]
            
            # Take only the requested number of terms
            mesh_terms = mesh_terms[:n_terms]
            
            if len(mesh_terms) < n_terms:
                logging.warning(f"Claude returned {len(mesh_terms)} MeSH terms, expected {n_terms}")
            
            return mesh_terms
            
        except Exception as e:
            logging.error(f"Error calling Claude API: {e}")
            return []
    
    def _create_enhanced_query(self, original_query: str, mesh_terms: List[str]) -> str:
        """Create enhanced PubMed query using MeSH terms."""
        
        if not mesh_terms:
            return original_query
        
        # Create OR-concatenated MeSH query
        mesh_query = " OR ".join(f'"{term}"[MeSH Terms]' for term in mesh_terms)
        
        # Combine with original query for broader coverage
        enhanced_query = f"({mesh_query}) OR ({original_query})"
        
        return enhanced_query
    
    def display_enhancement_info(self, enhanced_query: MeshEnhancedQuery, progress_callback: Optional[callable] = None) -> None:
        """Display information about query enhancement."""
        
        if enhanced_query.enhancement_method == "none":
            if progress_callback:
                progress_callback("Using original query (MeSH enhancement disabled)", 0)
            return
        
        if enhanced_query.enhancement_method == "failed":
            if progress_callback:
                progress_callback("MeSH enhancement failed, using original query", 0)
            return
        
        if enhanced_query.mesh_terms:
            mesh_display = ", ".join(enhanced_query.mesh_terms)
            if progress_callback:
                progress_callback(f"🏷️ Enhanced with MeSH terms: {mesh_display}", 0)
            else:
                print(f"🏷️ MeSH terms: {mesh_display}")
                print(f"📝 Enhanced query: {enhanced_query.enhanced_query}")


# Factory function for easy creation
def create_mesh_enhancer(api_key: Optional[str] = None, enable_mesh: bool = False) -> MeshQueryEnhancer:
    """Factory function to create a MeSH query enhancer."""
    return MeshQueryEnhancer(api_key=api_key, enable_mesh=enable_mesh)


# Utility function for testing
async def test_mesh_enhancement():
    """Test MeSH enhancement functionality."""
    import os
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        return
    
    enhancer = create_mesh_enhancer(api_key=api_key, enable_mesh=True)
    
    if not enhancer.is_available():
        print("❌ MeSH enhancement not available")
        return
    
    # Test queries
    test_queries = [
        "cancer gene expression",
        "COVID-19 immune response", 
        "alzheimer neurodegeneration"
    ]
    
    for query in test_queries:
        print(f"\n🧪 Testing: '{query}'")
        enhanced = await enhancer.enhance_query(query, n_terms=3)
        enhancer.display_enhancement_info(enhanced)


if __name__ == "__main__":
    # Run test
    asyncio.run(test_mesh_enhancement())