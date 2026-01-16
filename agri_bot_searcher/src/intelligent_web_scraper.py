"""
Intelligent Web Scraper with Agent-based Article Selection
Implements smart URL selection and comprehensive content extraction
Supports HTML pages and PDF documents
"""

import logging
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlparse
import re
import time
import io

# PDF extraction libraries
try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


@dataclass
class ArticleCandidate:
    """Represents a potential article for extraction"""
    url: str
    title: str
    snippet: str
    source_domain: str
    relevance_score: float
    estimated_quality: float
    selected_for_extraction: bool = False


@dataclass
class ExtractedArticle:
    """Fully extracted article content"""
    url: str
    title: str
    content: str
    metadata: Dict
    word_count: int
    extraction_timestamp: float


class ArticleSelectionAgent:
    """Agent that intelligently selects which articles to fully scrape"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        
        # Trusted domains get priority
        self.trusted_domains = [
            'icar.org.in', 'agricoop.gov.in', 'agriculture.gov.in',
            'iari.res.in', 'krishi.icar.gov.in', 'farmer.gov.in',
            'pib.gov.in', 'pmc.ncbi.nlm.nih.gov', 'researchgate.net',
            'sciencedirect.com', 'springer.com', 'mdpi.com',
            'tandfonline.com', 'wiley.com', 'nature.com', 'fao.org'
        ]
    
    def select_articles(self, candidates: List[ArticleCandidate], 
                       query: str, max_articles: int = 5) -> List[ArticleCandidate]:
        """
        Use LLM to intelligently select which articles to fully extract
        """
        
        # First, filter by quality threshold
        quality_threshold = 0.3
        quality_candidates = [c for c in candidates if c.estimated_quality >= quality_threshold]
        
        if len(quality_candidates) <= max_articles:
            # If we have few candidates, extract all
            for candidate in quality_candidates:
                candidate.selected_for_extraction = True
            return quality_candidates
        
        # Use LLM to rank candidates based on relevance to query
        try:
            candidates_text = "\n".join([
                f"{i+1}. [{c.title}]({c.url})\n   Domain: {c.source_domain}\n   Snippet: {c.snippet[:150]}..."
                for i, c in enumerate(quality_candidates[:15])  # Limit to top 15 for context
            ])
            
            prompt = f"""You are an agricultural research assistant. Given a search query and a list of article candidates, select the {max_articles} MOST RELEVANT articles that would provide the best information.

Query: {query}

Available Articles:
{candidates_text}

Return ONLY the numbers of the top {max_articles} most relevant articles, separated by commas (e.g., "1,3,5,7,9").
Consider:
1. Direct relevance to the query topic
2. Source credibility (government, research institutions, universities)
3. Specificity of information in the snippet
4. Indian agricultural context if applicable

Selected article numbers:"""

            response = requests.post(
                f'{self.ollama_host}/api/generate',
                json={
                    'model': 'llama3.2:latest',
                    'prompt': prompt,
                    'stream': False,
                    'options': {'temperature': 0.1, 'num_ctx': 4096}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                # Extract numbers from response
                selected_numbers = [int(n.strip()) for n in re.findall(r'\d+', answer)]
                
                # Mark selected articles
                for num in selected_numbers[:max_articles]:
                    if 0 < num <= len(quality_candidates):
                        quality_candidates[num-1].selected_for_extraction = True
                
                selected = [c for c in quality_candidates if c.selected_for_extraction]
                
                if selected:
                    self.logger.info(f"LLM selected {len(selected)} articles for extraction")
                    return selected
        
        except Exception as e:
            self.logger.warning(f"LLM selection failed: {e}, using fallback ranking")
        
        # Fallback: Use relevance score ranking
        quality_candidates.sort(key=lambda x: (x.relevance_score, x.estimated_quality), reverse=True)
        for candidate in quality_candidates[:max_articles]:
            candidate.selected_for_extraction = True
        
        return quality_candidates[:max_articles]


class IntelligentWebScraper:
    """Advanced web scraper with comprehensive content extraction and anti-bot handling"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Multiple user agents to rotate through
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15'
        ]
        self.current_ua_index = 0
        
        # Sites known to block bots - use snippet instead
        self.blocked_domains = [
            'researchgate.net',
            'sciencedirect.com',
            'springer.com',
            'wiley.com',
            'nature.com',
            'linkedin.com'
        ]
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        return url.lower().endswith('.pdf')
    
    def _extract_text_from_pdf(self, pdf_content: bytes, url: str) -> Optional[str]:
        """Extract text from PDF content using available libraries"""
        
        # Try PyMuPDF first (better text extraction)
        if HAS_PYMUPDF:
            try:
                self.logger.info(f"Extracting PDF text using PyMuPDF for {url}")
                pdf_stream = io.BytesIO(pdf_content)
                doc = fitz.open(stream=pdf_stream, filetype="pdf")
                
                text_parts = []
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        text_parts.append(text)
                
                doc.close()
                full_text = "\n\n".join(text_parts)
                
                if len(full_text.strip()) > 100:
                    self.logger.info(f"Successfully extracted {len(full_text)} chars from PDF using PyMuPDF")
                    return full_text
            except Exception as e:
                self.logger.warning(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to PyPDF2
        if HAS_PYPDF2:
            try:
                self.logger.info(f"Extracting PDF text using PyPDF2 for {url}")
                pdf_stream = io.BytesIO(pdf_content)
                reader = PyPDF2.PdfReader(pdf_stream)
                
                text_parts = []
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        text_parts.append(text)
                
                full_text = "\n\n".join(text_parts)
                
                if len(full_text.strip()) > 100:
                    self.logger.info(f"Successfully extracted {len(full_text)} chars from PDF using PyPDF2")
                    return full_text
            except Exception as e:
                self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        self.logger.error(f"Failed to extract text from PDF {url} - no PDF libraries available or extraction failed")
        return None
    
    def extract_full_article(self, url: str, title: str) -> Optional[ExtractedArticle]:
        """
        Extract complete article content without artificial limits
        Handles anti-bot protections gracefully and supports PDF extraction
        """
        try:
            # Check if domain is known to block bots
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            
            # Skip extraction for known blocking sites, return None to use snippet
            for blocked_domain in self.blocked_domains:
                if blocked_domain in domain:
                    self.logger.info(f"Skipping extraction for {domain} (known to block bots)")
                    return None
            
            self.logger.info(f"Extracting full content from: {url}")
            
            # Rotate user agent
            self.current_ua_index = (self.current_ua_index + 1) % len(self.user_agents)
            
            headers = {
                'User-Agent': self.user_agents[self.current_ua_index],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9,hi;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Cache-Control': 'max-age=0',
            }
            
            # Add small delay to be respectful
            time.sleep(0.5)
            
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            
            # Handle various HTTP errors
            if response.status_code == 403:
                self.logger.warning(f"Access forbidden (403) for {url} - likely bot detection")
                return None
            elif response.status_code == 429:
                self.logger.warning(f"Rate limited (429) for {url}")
                return None
            elif response.status_code != 200:
                self.logger.warning(f"Failed to fetch {url}: status {response.status_code}")
                return None
            
            # Check if response is PDF based on Content-Type or URL
            content_type = response.headers.get('Content-Type', '').lower()
            is_pdf = 'application/pdf' in content_type or self._is_pdf_url(url)
            
            if is_pdf:
                self.logger.info(f"Detected PDF content for {url}")
                
                # Extract text from PDF
                pdf_text = self._extract_text_from_pdf(response.content, url)
                
                if pdf_text and len(pdf_text.strip()) > 200:
                    word_count = len(pdf_text.split())
                    
                    return ExtractedArticle(
                        url=url,
                        title=title,
                        content=pdf_text,
                        metadata={
                            'source': 'pdf',
                            'content_type': content_type,
                            'file_size': len(response.content),
                            'extraction_method': 'pymupdf' if HAS_PYMUPDF else 'pypdf2'
                        },
                        word_count=word_count,
                        extraction_timestamp=time.time()
                    )
                else:
                    self.logger.warning(f"Failed to extract meaningful text from PDF: {url}")
                    return None
            
            # Regular HTML content extraction
            # Check if we got a CAPTCHA or bot check page
            if self._is_bot_check_page(response.text):
                self.logger.warning(f"Bot check detected for {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                                'iframe', 'noscript', 'form', 'button', 'input']):
                element.decompose()
            
            # Extract metadata
            metadata = self._extract_metadata(soup)
            
            # Find main content
            content = self._extract_main_content(soup)
            
            if not content or len(content.strip()) < 200:
                self.logger.warning(f"Insufficient content extracted from {url}")
                return None
            
            word_count = len(content.split())
            
            article = ExtractedArticle(
                url=url,
                title=title,
                content=content,
                metadata=metadata,
                word_count=word_count,
                extraction_timestamp=datetime.now().timestamp()
            )
            
            self.logger.info(f"Successfully extracted {word_count} words from {url}")
            return article
            
        except requests.exceptions.Timeout:
            self.logger.warning(f"Timeout extracting {url}")
        except requests.exceptions.TooManyRedirects:
            self.logger.warning(f"Too many redirects for {url}")
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Request failed for {url}: {e}")
        except Exception as e:
            self.logger.error(f"Extraction error for {url}: {e}")
        
        return None
    
    def _is_bot_check_page(self, html_content: str) -> bool:
        """Detect if the page is a CAPTCHA or bot verification page"""
        bot_check_indicators = [
            'captcha',
            'are you a robot',
            'verify you are human',
            'cloudflare',
            'security check',
            'access denied',
            'unusual traffic'
        ]
        
        html_lower = html_content.lower()
        
        # If page is very short, it might be a redirect/error page
        if len(html_content) < 500:
            return True
        
        # Check for bot detection indicators
        indicator_count = sum(1 for indicator in bot_check_indicators if indicator in html_lower)
        
        return indicator_count >= 2
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract article metadata"""
        metadata = {}
        
        # Try to find publication date
        date_patterns = [
            ('meta', {'property': 'article:published_time'}),
            ('meta', {'name': 'pubdate'}),
            ('meta', {'name': 'date'}),
            ('time', {'class': re.compile('date|publish|time', re.I)}),
        ]
        
        for tag, attrs in date_patterns:
            elem = soup.find(tag, attrs)
            if elem:
                date_val = elem.get('content') or elem.get('datetime') or elem.get_text()
                if date_val:
                    metadata['publication_date'] = date_val.strip()
                    break
        
        # Try to find author
        author_patterns = [
            ('meta', {'name': 'author'}),
            ('meta', {'property': 'article:author'}),
            ('span', {'class': re.compile('author', re.I)}),
            ('div', {'class': re.compile('author', re.I)}),
        ]
        
        for tag, attrs in author_patterns:
            elem = soup.find(tag, attrs)
            if elem:
                author_val = elem.get('content') or elem.get_text()
                if author_val:
                    metadata['author'] = author_val.strip()
                    break
        
        # Extract keywords if available
        keywords_meta = soup.find('meta', {'name': 'keywords'})
        if keywords_meta and keywords_meta.get('content'):
            metadata['keywords'] = keywords_meta['content']
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main article content using multiple strategies
        NO ARBITRARY CHARACTER LIMITS - extract full content
        """
        
        # Strategy 1: Look for article/main semantic tags
        for tag_name in ['article', 'main']:
            main_tag = soup.find(tag_name)
            if main_tag:
                content = self._extract_text_from_element(main_tag)
                if len(content.strip()) > 500:  # Minimum threshold
                    return content
        
        # Strategy 2: Look for common content container IDs/classes
        content_selectors = [
            {'id': re.compile('content|article|main|post', re.I)},
            {'class': re.compile('content|article|main|post|entry|text', re.I)},
        ]
        
        for selector in content_selectors:
            containers = soup.find_all(['div', 'section'], selector)
            for container in containers:
                content = self._extract_text_from_element(container)
                if len(content.strip()) > 500:
                    return content
        
        # Strategy 3: Find element with most paragraph tags
        all_containers = soup.find_all(['div', 'section', 'article'])
        best_container = None
        max_paragraph_count = 0
        
        for container in all_containers:
            paragraph_count = len(container.find_all('p'))
            if paragraph_count > max_paragraph_count:
                max_paragraph_count = paragraph_count
                best_container = container
        
        if best_container and max_paragraph_count >= 3:
            content = self._extract_text_from_element(best_container)
            if content.strip():
                return content
        
        # Strategy 4: Fallback - extract all paragraphs
        all_paragraphs = soup.find_all('p')
        if all_paragraphs:
            texts = []
            for p in all_paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 30:  # Filter very short paragraphs
                    texts.append(text)
            
            if texts:
                return '\n\n'.join(texts)
        
        # Last resort: body text
        body = soup.find('body')
        if body:
            return self._extract_text_from_element(body)
        
        return soup.get_text(strip=True)
    
    def _extract_text_from_element(self, element) -> str:
        """Extract clean text from an HTML element"""
        # Get all text-containing elements
        text_elements = element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'blockquote'])
        
        texts = []
        for elem in text_elements:
            text = elem.get_text(separator=' ', strip=True)
            # Filter out very short or navigation-like text
            if len(text) > 20 and not self._is_navigation_text(text):
                texts.append(text)
        
        # If we got good structured content, return it
        if texts:
            return '\n\n'.join(texts)
        
        # Otherwise return all text from element
        return element.get_text(separator=' ', strip=True)
    
    def _is_navigation_text(self, text: str) -> bool:
        """Check if text is likely navigation/menu content"""
        nav_indicators = [
            'copyright ©', 'all rights reserved', 'privacy policy',
            'terms of service', 'cookie policy', 'contact us',
            'follow us', 'subscribe', 'newsletter', 'sign up'
        ]
        
        text_lower = text.lower()
        
        # Too short
        if len(text) < 20:
            return True
        
        # Contains too many nav indicators
        nav_count = sum(1 for indicator in nav_indicators if indicator in text_lower)
        if nav_count >= 2:
            return True
        
        # Mostly links (high ratio of words to actual content)
        words = text.split()
        if len(words) < 5:
            return True
        
        return False
