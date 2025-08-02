from bs4 import BeautifulSoup
import os
import time
import argparse
import re
import sys
import requests


def check_dependencies():
    """
    Check if required dependencies are installed.
    
    The script relies on lxml for XML parsing which is used by BeautifulSoup.
    This function tests if it's available before running the main process.
    
    Returns:
        bool: True if dependencies are installed, False otherwise
    """
    try:
        import lxml
        return True
    except ImportError:
        print("Error: lxml library is not installed.")
        print("Please install it using: pip install lxml")
        print("Or install all dependencies with: pip install requests beautifulsoup4 lxml")
        return False


class PubMedRetriever:
    """
    A class to search PubMed Central and retrieve article PDFs.
    
    This class provides functionality to:
    1. Search PMC using NCBI's E-utilities API
    2. Retrieve metadata for PMC articles
    3. Download articles in PDF, XML, or plain text formats
    4. Handle various edge cases and fallbacks when a format is unavailable
    """
    
    # Base URLs for NCBI E-utilities API endpoints
    BASE_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"  # For searching articles
    BASE_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"    # For fetching article data
    PMC_URL = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC"                       # Base URL for PMC articles
    
    def __init__(self, email=None, api_key=None, output_dir="./papers"):
        """
        Initialize the retriever with optional API key for higher rate limits.
        
        NCBI recommends providing an email and API key to get higher request limits
        and to help them contact you if there are issues with your usage pattern.
        
        Args:
            email (str): User email for NCBI API - helps with rate limits and contact
            api_key (str): NCBI API key for higher request limits (optional)
            output_dir (str): Directory to save downloaded PDFs (created if not exists)
        """
        self.email = email
        self.api_key = api_key
        self.output_dir = output_dir
        
        # Common headers to mimic a browser request
        # Updated headers to better mimic a real browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
        }
        
        # Create a session to maintain cookies across requests
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Ensure directory paths are valid and usable
        self.output_dir = os.path.abspath(output_dir)
        print(f"Files will be saved to: {self.output_dir}")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def search_pubmed(self, query, max_results=10):
        """
        Search PubMed for articles matching the query using NCBI's E-utilities.
        
        This method constructs a request to the ESearch endpoint to find articles
        matching the search terms, then parses the response to extract PMC IDs.
        
        Args:
            query (str): The search query - can use PMC search syntax
            max_results (int): Maximum number of results to return
            
        Returns:
            list: List of PMC IDs (as strings)
        """
        # Set up search parameters for E-utilities API
        params = {
            "db": "pmc",  # Search PMC database (full-text articles)
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"  # Sort by relevance to query
        }
        
        # Add authentication parameters if available
        # These help with rate limits and avoiding blocks
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            # Make the search request with a timeout to avoid hanging
            response = self.session.get(self.BASE_SEARCH_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Error searching PubMed: {response.status_code}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to PubMed API: {e}")
            return []
        
        try:
            # Parse JSON response to extract PMC IDs
            data = response.json()
            pmc_ids = data.get("esearchresult", {}).get("idlist", [])
            return pmc_ids
        except Exception as e:
            print(f"Error parsing search results: {str(e)}")
            return []
    
    def get_pmc_info(self, pmc_id):
        """
        Get article information from PMC ID using the EFetch utility.
        
        Retrieves metadata for an article including title and constructs
        a URL for accessing the article on PMC.
        
        Args:
            pmc_id (str): PMC ID of the article
            
        Returns:
            dict: Article information including PMC ID, title, and URL,
                  or None if retrieval fails
        """
        # Set up parameters for E-utilities API to fetch article metadata
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml"  # Request XML format for structured data
        }
        
        # Add authentication if available
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            # Request article metadata
            response = self.session.get(self.BASE_FETCH_URL, params=params, timeout=30)
            
            if response.status_code != 200:
                print(f"Error fetching article {pmc_id}: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to PubMed API: {e}")
            return None
        
        try:
            # Parse XML response to extract article title
            # Using lxml-xml parser for better performance with XML
            soup = BeautifulSoup(response.content, 'lxml-xml')
            article = soup.find('article')
            
            if not article:
                return None
                
            title_element = article.find('article-title')
            title = title_element.text if title_element else "Unknown Title"
            
            # Return structured article info
            return {
                "pmc_id": pmc_id,
                "title": title,
                "url": f"{self.PMC_URL}{pmc_id}/"
            }
        
        except Exception as e:
            print(f"Error parsing article info: {str(e)}")
            return None
    
    def download_fulltext(self, pmc_id, title=None):
        """
        Download full text for a PMC article using a cascade of formats.
        
        This method tries to download in the following order:
        1. PDF (preferred format)
        2. XML (fallback if PDF unavailable)
        3. Plain text (last resort)
        
        Args:
            pmc_id (str): PMC ID of the article
            title (str): Title of the article (for filename)
            
        Returns:
            str: Path to downloaded file or None if all formats failed
        """
        # Try to download PDF first (preferred format)
        pdf_path = self.try_download_pdf(pmc_id, title)
        if pdf_path:
            return pdf_path
            
        print(f"PDF not available for PMC{pmc_id}, trying XML...")
        
        # Try to download XML as fallback
        xml_path = self.try_download_xml(pmc_id, title)
        if xml_path:
            return xml_path
            
        print(f"XML not available for PMC{pmc_id}, trying plain text...")
        
        # Try to download text as last resort
        text_path = self.try_download_text(pmc_id, title)
        if text_path:
            return text_path
            
        print(f"Unable to retrieve any fulltext format for PMC{pmc_id}")
        return None
        
    def try_download_pdf(self, pmc_id, title=None):
        """
        Attempt to download PDF for a PMC article using multiple methods.
        
        This method uses several approaches to find and download PDFs:
        1. Visit article page and look for PDF links 
        2. Try different HTML selectors for finding PDF links
        3. Attempt direct construction of PDF URL
        
        Args:
            pmc_id (str): PMC ID of the article
            title (str): Title of the article (for filename)
            
        Returns:
            str: Path to downloaded PDF or None if failed
        """
        # First visit the PMC article page to locate PDF link
        # This is necessary because direct PDF URLs can vary
        article_url = f"{self.PMC_URL}{pmc_id}/"
        try:
            response = self.session.get(article_url, timeout=30)
            
            if response.status_code == 403:
                # 403 Forbidden typically means anti-scraping measures triggered
                print(f"Access forbidden (403) - PubMed is blocking automated access")
                print(f"Try opening the article manually at: {article_url}")
                return None
            elif response.status_code != 200:
                print(f"Error accessing article page: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to PMC: {e}")
            return None
        
        # Parse the page to find the PDF link
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for PDF links - PMC articles have PDF links with specific patterns
        pdf_filename = None
        
        # Method 1: Look for links containing 'pdf' in href
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Look for links that end with .pdf and contain manuscript ID
            if href.endswith('.pdf') and ('nihms' in href or 'PMC' in href):
                pdf_filename = href
                break
        
        # Method 2: Look for PDF button/link with specific classes
        if not pdf_filename:
            pdf_link = soup.find('a', {'class': lambda x: x and any('pdf' in cls.lower() for cls in x)})
            if pdf_link and pdf_link.get('href'):
                pdf_filename = pdf_link['href']
        
        if not pdf_filename:
            print(f"Could not find PDF link for PMC{pmc_id}")
            return None
        
        # Construct the full PDF URL
        if pdf_filename.startswith('/'):
            pdf_url = f"https://www.ncbi.nlm.nih.gov{pdf_filename}"
        elif not pdf_filename.startswith('http'):
            # The PDF is relative to the article URL
            pdf_url = f"{article_url}{pdf_filename}"
        else:
            pdf_url = pdf_filename
        
        print(f"Found PDF URL: {pdf_url}")
        
        # Download the PDF
        try:
            pdf_response = self.session.get(pdf_url, timeout=60, allow_redirects=True)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading PDF: {e}")
            return None
        
        if pdf_response.status_code != 200:
            print(f"Error downloading PDF: {pdf_response.status_code}")
            return None
        
        # Check if we actually got a PDF
        if not pdf_response.content.startswith(b'%PDF'):
            print(f"Downloaded content is not a PDF (got {pdf_response.headers.get('content-type', 'unknown')})")
            
            # Check if it's the POW challenge page
            if b'Preparing to download' in pdf_response.content:
                print("PMC is showing a Proof of Work challenge page.")
                print("This is an anti-bot mechanism. Manual download may be required.")
                print(f"Please download manually from: {article_url}")
            
            return None
        
        # Create a filename based on PMC ID and title
        if title:
            # Clean title to make a valid filename by removing illegal characters
            clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
            clean_title = clean_title[:100]  # Limit length to avoid path issues
            filename = f"PMC{pmc_id}_{clean_title}.pdf"
        else:
            filename = f"PMC{pmc_id}.pdf"
            
        file_path = os.path.join(self.output_dir, filename)
        
        # Save the downloaded PDF to disk
        with open(file_path, 'wb') as file:
            file.write(pdf_response.content)
        
        print(f"Downloaded: {filename}")
        return file_path
        
    def try_download_xml(self, pmc_id, title=None):
        """
        Attempt to download XML full text for a PMC article.
        
        This method tries two approaches:
        1. Use E-utilities API to request XML directly
        2. If that fails, try to find XML download link on article page
        
        Args:
            pmc_id (str): PMC ID of the article
            title (str): Title of the article (for filename)
            
        Returns:
            str: Path to downloaded XML or None if failed
        """
        # Try to get XML via E-utilities API first
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "xml",
            "rettype": "full"  # Request full text
        }
        
        # Add authentication if available
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            print(f"Attempting to download XML via E-utilities...")
            response = self.session.get(self.BASE_FETCH_URL, params=params, timeout=30)
            
            # Check if response is valid and has substantial content
            # Small responses likely indicate error messages rather than article XML
            if response.status_code != 200 or len(response.content) < 1000:
                print(f"E-utilities XML retrieval failed or returned incomplete data")
                
                # Fallback: Try via direct PMC page
                article_url = f"{self.PMC_URL}{pmc_id}/"
                print(f"Trying to find XML link on article page: {article_url}")
                page_response = self.session.get(article_url, timeout=30)
                
                if page_response.status_code != 200:
                    print(f"Failed to access article page: {page_response.status_code}")
                    return None
                
                # Parse the page to find XML download link
                soup = BeautifulSoup(page_response.content, 'html.parser')
                
                # Look for XML download link in page text
                xml_link = None
                for link in soup.select('a'):
                    link_text = link.get_text().lower()
                    if 'xml' in link_text or 'full text' in link_text:
                        xml_link = link
                        break
                
                if not xml_link:
                    print("Could not find XML link on article page")
                    return None
                
                # Get the XML URL and handle relative URLs
                xml_url = xml_link.get('href')
                if xml_url.startswith('/'):
                    xml_url = "https://www.ncbi.nlm.nih.gov" + xml_url
                elif not (xml_url.startswith('http://') or xml_url.startswith('https://')):
                    xml_url = f"{article_url}{xml_url}"
                
                print(f"Found XML link: {xml_url}")
                xml_response = self.session.get(xml_url, timeout=30)
                
                if xml_response.status_code != 200:
                    print(f"Failed to download XML: {xml_response.status_code}")
                    return None
                
                # Use the new response for saving
                response = xml_response
            
            # Create a filename based on PMC ID and title
            if title:
                clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
                clean_title = clean_title[:100]
                filename = f"PMC{pmc_id}_{clean_title}.xml"
            else:
                filename = f"PMC{pmc_id}.xml"
                
            file_path = os.path.join(self.output_dir, filename)
            
            # Save the XML file
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            print(f"Downloaded XML: {filename}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading XML: {e}")
            return None
    
    def try_download_text(self, pmc_id, title=None):
        """
        Attempt to download plain text for a PMC article.
        
        This method tries two approaches:
        1. Use E-utilities API to request text directly
        2. If that fails, extract text from HTML article page
        
        Args:
            pmc_id (str): PMC ID of the article
            title (str): Title of the article (for filename)
            
        Returns:
            str: Path to downloaded text file or None if failed
        """
        # Try direct text retrieval from E-utilities
        params = {
            "db": "pmc",
            "id": pmc_id,
            "retmode": "text",
            "rettype": "full"
        }
        
        # Add authentication if available
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
            
        try:
            print(f"Attempting to download text via E-utilities...")
            response = self.session.get(self.BASE_FETCH_URL, params=params, timeout=30)
            
            # Check if response is valid and has substantial content
            if response.status_code != 200 or len(response.content) < 500:
                print(f"E-utilities text retrieval failed or returned incomplete data")
                
                # Fallback: Extract text from HTML
                article_url = f"{self.PMC_URL}{pmc_id}/"
                print(f"Extracting text from article page as fallback: {article_url}")
                
                page_response = self.session.get(article_url, timeout=30)
                if page_response.status_code != 200:
                    print(f"Failed to access article page: {page_response.status_code}")
                    return None
                
                # Parse the HTML to extract text content
                soup = BeautifulSoup(page_response.content, 'html.parser')
                
                # Extract main text content using various selectors
                article_text = ""
                
                # Try to find the main article container using several common selectors
                # PMC's structure can vary, so we try multiple likely containers
                article_div = soup.select_one('div.jig-ncbiinpagenav')
                if not article_div:
                    article_div = soup.select_one('article')
                if not article_div:
                    article_div = soup.select_one('div.article')
                if not article_div:
                    article_div = soup  # Fallback to entire page
                
                # Extract paragraphs
                paragraphs = article_div.select('p')
                for p in paragraphs:
                    article_text += p.get_text() + "\n\n"
                
                # Extract headings to preserve structure
                headings = article_div.select('h1, h2, h3, h4, h5, h6')
                for h in headings:
                    if h.get_text() not in article_text:
                        article_text += h.get_text() + "\n\n"
                
                # Verify we got meaningful content
                if not article_text or len(article_text) < 500:
                    print("Could not extract meaningful text from the article page")
                    return None
                
                # Use the extracted text instead of the API response
                content = article_text.encode('utf-8')
            else:
                content = response.content
            
            # Create a filename based on PMC ID and title
            if title:
                clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
                clean_title = clean_title[:100]
                filename = f"PMC{pmc_id}_{clean_title}.txt"
            else:
                filename = f"PMC{pmc_id}.txt"
                
            file_path = os.path.join(self.output_dir, filename)
            
            # Save the text file
            with open(file_path, 'wb') as file:
                file.write(content)
            
            print(f"Downloaded text: {filename}")
            return file_path
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading text: {e}")
            return None
    
    def search_and_download(self, query, max_results=5):
        """
        Search PubMed and download papers for matching articles.
        
        This is the main method that combines searching and downloading:
        1. Searches for articles matching query
        2. Displays results to user
        3. Allows selection of which articles to download
        4. Downloads selected articles in best available format
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of articles to search for
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"Searching for: {query}")
        pmc_ids = self.search_pubmed(query, max_results)
        
        if not pmc_ids:
            print("No results found.")
            return []
        
        print(f"Found {len(pmc_ids)} results.")
        
        # Get article info for all results
        articles = []
        
        for i, pmc_id in enumerate(pmc_ids):
            article_info = self.get_pmc_info(pmc_id)
            if article_info:
                articles.append(article_info)
                print(f"{i+1}. PMC{pmc_id}: {article_info['title']}")
            
            # Rate limiting to avoid being blocked
            if i < len(pmc_ids) - 1:
                time.sleep(1)
        
        if not articles:
            print("No article information found.")
            return []
        
        # Interactive selection: Ask user which articles to download
        while True:
            try:
                selection = input("\nEnter the numbers to download (comma-separated), 'all' for all, or 'q' to quit: ")
                if selection.lower() == 'q':
                    return []
                
                if selection.lower() == 'all':
                    selected_indices = list(range(len(articles)))
                    break
                
                # Parse comma-separated list of indices
                selected_indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
                # Validate indices are within range
                if all(0 <= idx < len(articles) for idx in selected_indices):
                    break
                else:
                    print(f"Invalid selection. Please enter numbers between 1 and {len(articles)}.")
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers, 'all', or 'q'.")
        
        print(f"\nDownloading {len(selected_indices)} articles...")
        downloaded_files = []
        
        # Download selected articles
        for idx in selected_indices:
            article = articles[idx]
            pmc_id = article['pmc_id']
            
            print(f"\nProcessing: PMC{pmc_id}")
            print(f"Title: {article['title']}")
            
            # Download article in best available format
            pdf_path = self.download_fulltext(pmc_id, article['title'])
            
            if pdf_path:
                downloaded_files.append(pdf_path)
            
            # More aggressive rate limiting for downloads to avoid blocks
            if idx < len(selected_indices) - 1:
                time.sleep(3)  # 3-second delay between downloads
        
        print(f"Downloaded {len(downloaded_files)} files to {self.output_dir}")
        return downloaded_files


def main():
    """
    Main function to parse command line arguments and run the PubMed retriever.
    
    This function:
    1. Checks for required dependencies
    2. Sets up command-line argument parsing
    3. Initializes the PubMedRetriever with user options
    4. Executes the search and download process
    """
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
        
    # Print usage warnings about possible anti-scraping measures
    print("Note: PubMed may implement anti-bot measures including:")
    print("1. Proof of Work challenges for PDF downloads")
    print("2. Rate limiting and IP blocking")
    print("3. Using the --email parameter with your email address may help")
    print("4. XML format is often more reliable than PDF")
    print("")
    
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Search PubMed Central and download article PDFs')
    parser.add_argument('query', type=str, help='Search query')
    parser.add_argument('--max', type=int, default=5, help='Maximum number of articles to download')
    parser.add_argument('--email', type=str, help='Email for NCBI API')
    parser.add_argument('--api-key', type=str, help='NCBI API key')
    parser.add_argument('--output', type=str, default='./papers', help='Output directory')
    args = parser.parse_args()
    
    # Initialize the retriever with command-line options
    retriever = PubMedRetriever(
        email=args.email,
        api_key=args.api_key,
        output_dir=args.output
    )
    
    # Run the search and download process
    retriever.search_and_download(args.query, max_results=args.max)


if __name__ == "__main__":
    main()