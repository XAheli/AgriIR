#!/usr/bin/env python3
"""
Advanced Web Search System
Multi-engine search with intelligent ranking and quality filtering
"""

import time
import hashlib
import requests
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote_plus
from collections import defaultdict
import logging
import re

# Third-party imports
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import newspaper
from newspaper import Article

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Enhanced search result with quality metrics"""
    title: str
    url: str
    snippet: str
    content: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Quality metrics
    relevance_score: float = 0.0
    credibility_score: float = 0.0
    freshness_score: float = 0.0
    combined_score: float = 0.0
    
    # Metadata
    source_domain: str = ""
    author: Optional[str] = None
    publish_date: Optional[str] = None
    content_type: str = "article"  # article, research_paper, government_doc, news
    
    # Extraction metadata
    extraction_method: str = "basic"
    has_structured_data: bool = False
    word_count: int = 0
    
    def __post_init__(self):
        if not self.source_domain and self.url:
            self.source_domain = urlparse(self.url).netloc


@dataclass
class SearchEngineResult:
    """Result from a specific search engine"""
    engine: str
    results: List[SearchResult]
    query_time: float
    success: bool
    error: Optional[str] = None


class SearchCache:
    """Simple in-memory cache for search results"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, Tuple[List[SearchResult], float]] = {}
        self.ttl = ttl_seconds
    
    def get(self, query: str) -> Optional[List[SearchResult]]:
        """Get cached results if available and fresh"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if cache_key in self.cache:
            results, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Cache hit for query: {query[:50]}")
                return results
            else:
                # Expired, remove from cache
                del self.cache[cache_key]
        
        return None
    
    def set(self, query: str, results: List[SearchResult]):
        """Cache search results"""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self.cache[cache_key] = (results, time.time())
        logger.info(f"Cached {len(results)} results for query: {query[:50]}")


class DomainQualityChecker:
    """Check domain credibility and authority"""
    
    # Trusted agriculture domains
    TRUSTED_DOMAINS = {
        'gov': 10,  # Government sites
        'edu': 9,   # Educational institutions
        'org': 7,   # Organizations
        'ac.in': 9, # Academic institutions (India)
        'nic.in': 10, # Government (India)
    }
    
    # High authority agriculture domains
    AGRICULTURE_AUTHORITIES = {
        'fao.org': 10,
        'cgiar.org': 10,
        'icar.gov.in': 10,
        'agricoop.gov.in': 10,
        'tnau.ac.in': 9,
        'extension.org': 8,
        'agritech.tnau.ac.in': 9,
    }
    
    def check_domain(self, domain: str) -> float:
        """
        Check domain credibility score (0-10)
        
        Args:
            domain: Domain name (e.g., 'example.com')
            
        Returns:
            Credibility score from 0 to 10
        """
        domain_lower = domain.lower()
        
        # Check if in high authority list
        if domain_lower in self.AGRICULTURE_AUTHORITIES:
            return self.AGRICULTURE_AUTHORITIES[domain_lower]
        
        # Check TLD
        for tld, score in self.TRUSTED_DOMAINS.items():
            if domain_lower.endswith(tld):
                return score
        
        # Check for specific keywords
        if any(word in domain_lower for word in ['agriculture', 'farming', 'agri', 'crop']):
            return 6
        
        # Default score for unknown domains
        return 5


class SmartContentExtractor:
    """Intelligent content extraction from web pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract(self, url: str, timeout: int = 15) -> Dict[str, any]:
        """
        Extract content from URL using multiple methods
        
        Args:
            url: URL to extract from
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with extracted content and metadata
        """
        result = {
            'success': False,
            'content': '',
            'title': '',
            'author': None,
            'publish_date': None,
            'extraction_method': 'none',
            'word_count': 0,
            'has_structured_data': False,
            'structured_data': {}
        }
        
        try:
            # Method 1: Try newspaper3k (best for articles)
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                if article.text and len(article.text) > 200:
                    result['success'] = True
                    result['content'] = article.text
                    result['title'] = article.title
                    result['author'] = ', '.join(article.authors) if article.authors else None
                    result['publish_date'] = str(article.publish_date) if article.publish_date else None
                    result['extraction_method'] = 'newspaper3k'
                    result['word_count'] = len(article.text.split())
                    
                    logger.info(f"Extracted content using newspaper3k: {len(article.text)} chars")
                    return result
            except Exception as e:
                logger.debug(f"Newspaper3k extraction failed: {e}")
            
            # Method 2: Custom BeautifulSoup extraction
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe']):
                element.decompose()
            
            # Extract title
            title = soup.find('title')
            if title:
                result['title'] = title.get_text().strip()
            
            # Try to find main content
            main_content = self._extract_main_content(soup)
            
            if main_content:
                result['success'] = True
                result['content'] = main_content
                result['extraction_method'] = 'beautifulsoup_smart'
                result['word_count'] = len(main_content.split())
                
                # Extract structured data
                structured = self._extract_structured_data(soup)
                if structured:
                    result['has_structured_data'] = True
                    result['structured_data'] = structured
                
                logger.info(f"Extracted content using BeautifulSoup: {len(main_content)} chars")
                return result
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout extracting content from {url}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error extracting from {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error extracting from {url}: {e}")
        
        return result
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content using intelligent heuristics"""
        content_parts = []
        
        # Priority 1: Look for semantic HTML5 tags
        for tag_name in ['article', 'main']:
            elements = soup.find_all(tag_name)
            for elem in elements:
                text = elem.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    content_parts.append(text)
        
        if content_parts:
            return ' '.join(content_parts)
        
        # Priority 2: Look for divs with content-related classes
        content_classes = ['content', 'article', 'post', 'entry', 'main', 'body']
        for class_name in content_classes:
            elements = soup.find_all('div', class_=re.compile(class_name, re.I))
            for elem in elements:
                text = elem.get_text(separator=' ', strip=True)
                if len(text) > 200:
                    content_parts.append(text)
        
        if content_parts:
            return ' '.join(content_parts)
        
        # Priority 3: Extract all paragraphs
        paragraphs = soup.find_all('p')
        content_parts = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 50:  # Filter out short fragments
                content_parts.append(text)
        
        return ' '.join(content_parts)
    
    def _extract_structured_data(self, soup: BeautifulSoup) -> Dict:
        """Extract structured data like tables and lists"""
        structured = {}
        
        # Extract tables
        tables = soup.find_all('table')
        if tables:
            structured['tables'] = []
            for i, table in enumerate(tables[:3]):  # Limit to first 3 tables
                table_data = []
                rows = table.find_all('tr')
                for row in rows:
                    cols = row.find_all(['td', 'th'])
                    if cols:
                        table_data.append([col.get_text(strip=True) for col in cols])
                
                if table_data:
                    structured['tables'].append(table_data)
        
        # Extract lists
        lists = soup.find_all(['ul', 'ol'])
        if lists:
            structured['lists'] = []
            for lst in lists[:5]:  # Limit to first 5 lists
                items = [li.get_text(strip=True) for li in lst.find_all('li')]
                if items and len(items) > 2:
                    structured['lists'].append(items)
        
        return structured


class SearchResultRanker:
    """Rank and filter search results by quality"""
    
    def __init__(self):
        self.domain_checker = DomainQualityChecker()
    
    def rank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """
        Rank results by combined quality score
        
        Args:
            results: List of search results
            query: Original search query
            
        Returns:
            Ranked list of results
        """
        query_terms = set(query.lower().split())
        
        for result in results:
            # Calculate relevance score
            result.relevance_score = self._calculate_relevance(result, query_terms)
            
            # Calculate credibility score
            result.credibility_score = self._calculate_credibility(result)
            
            # Calculate freshness score
            result.freshness_score = self._calculate_freshness(result)
            
            # Combined score (weighted average)
            result.combined_score = (
                result.relevance_score * 0.5 +
                result.credibility_score * 0.3 +
                result.freshness_score * 0.2
            )
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        
        return results
    
    def _calculate_relevance(self, result: SearchResult, query_terms: Set[str]) -> float:
        """Calculate relevance score (0-10)"""
        text = (result.title + ' ' + result.snippet + ' ' + result.content).lower()
        words = set(text.split())
        
        # Term frequency
        matching_terms = query_terms.intersection(words)
        term_coverage = len(matching_terms) / len(query_terms) if query_terms else 0
        
        # Boost for agriculture keywords
        agriculture_keywords = ['agriculture', 'farming', 'crop', 'soil', 'irrigation', 
                               'cultivation', 'harvest', 'fertilizer', 'seed', 'plant']
        agri_matches = sum(1 for keyword in agriculture_keywords if keyword in text)
        agri_boost = min(agri_matches * 0.5, 3)  # Max 3 points boost
        
        # Combine
        relevance = term_coverage * 7 + agri_boost
        
        return min(relevance, 10)
    
    def _calculate_credibility(self, result: SearchResult) -> float:
        """Calculate credibility score (0-10)"""
        domain_score = self.domain_checker.check_domain(result.source_domain)
        
        # Boost for author information
        author_boost = 1 if result.author else 0
        
        # Boost for date information
        date_boost = 1 if result.publish_date else 0
        
        # Penalize very short content
        content_penalty = 0
        if result.word_count < 100:
            content_penalty = 2
        elif result.word_count < 300:
            content_penalty = 1
        
        credibility = domain_score + author_boost + date_boost - content_penalty
        
        return max(0, min(credibility, 10))
    
    def _calculate_freshness(self, result: SearchResult) -> float:
        """Calculate freshness score (0-10)"""
        if not result.publish_date:
            return 5  # Neutral score if no date
        
        try:
            # Parse date (simplified, would need more robust parsing)
            # For now, return neutral score
            return 7
        except:
            return 5
    
    def filter_low_quality(self, results: List[SearchResult], min_score: float = 3.0) -> List[SearchResult]:
        """Filter out low quality results"""
        filtered = [r for r in results if r.combined_score >= min_score]
        logger.info(f"Filtered {len(results) - len(filtered)} low quality results")
        return filtered


class AdvancedWebSearcher:
    """
    Advanced multi-engine web searcher with intelligent ranking and quality filtering
    """
    
    def __init__(self, cache_ttl: int = 3600, max_retries: int = 2):
        """
        Initialize advanced web searcher
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            max_retries: Maximum retry attempts for failed searches
        """
        self.cache = SearchCache(ttl_seconds=cache_ttl)
        self.content_extractor = SmartContentExtractor()
        self.result_ranker = SearchResultRanker()
        self.max_retries = max_retries
        
        logger.info("Advanced Web Searcher initialized")
    
    def search(self, query: str, num_results: int = 10, 
               extract_content: bool = True) -> List[SearchResult]:
        """
        Perform advanced multi-engine search
        
        Args:
            query: Search query
            num_results: Number of results to return
            extract_content: Whether to extract full content from URLs
            
        Returns:
            Ranked list of search results
        """
        # Check cache first
        cached_results = self.cache.get(query)
        if cached_results:
            return cached_results[:num_results]
        
        # Generate query variants
        query_variants = self._generate_query_variants(query)
        
        # Search with multiple engines
        all_results = []
        for variant in query_variants:
            # DuckDuckGo search
            ddg_results = self._search_duckduckgo(variant, num_results * 2)
            all_results.extend(ddg_results)
        
        # Deduplicate results
        unique_results = self._deduplicate_results(all_results)
        
        # Extract content if requested
        if extract_content:
            unique_results = self._extract_content_batch(unique_results)
        
        # Rank results
        ranked_results = self.result_ranker.rank_results(unique_results, query)
        
        # Filter low quality
        filtered_results = self.result_ranker.filter_low_quality(ranked_results)
        
        # Cache results
        self.cache.set(query, filtered_results)
        
        # Return top N results
        return filtered_results[:num_results]
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """Generate query variants for better coverage"""
        variants = [query]
        
        # Add agriculture context if not present
        if 'agriculture' not in query.lower() and 'farming' not in query.lower():
            variants.append(f"{query} agriculture")
            variants.append(f"{query} farming India")
        
        # Add specific contexts
        variants.append(f"{query} best practices")
        
        # Limit to avoid too many searches
        return variants[:2]
    
    def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                
                for result in search_results:
                    search_result = SearchResult(
                        title=result.get('title', ''),
                        url=result.get('href', ''),
                        snippet=result.get('body', ''),
                        source_domain=urlparse(result.get('href', '')).netloc
                    )
                    results.append(search_result)
            
            logger.info(f"DuckDuckGo returned {len(results)} results for: {query[:50]}")
        
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
        
        return results
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on URL"""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            # Normalize URL
            normalized_url = result.url.lower().rstrip('/')
            
            if normalized_url not in seen_urls:
                seen_urls.add(normalized_url)
                unique_results.append(result)
        
        logger.info(f"Deduplicated {len(results)} -> {len(unique_results)} results")
        return unique_results
    
    def _extract_content_batch(self, results: List[SearchResult]) -> List[SearchResult]:
        """Extract content from multiple URLs"""
        for result in results:
            try:
                extraction = self.content_extractor.extract(result.url)
                
                if extraction['success']:
                    result.content = extraction['content']
                    result.title = extraction['title'] or result.title
                    result.author = extraction['author']
                    result.publish_date = extraction['publish_date']
                    result.extraction_method = extraction['extraction_method']
                    result.word_count = extraction['word_count']
                    result.has_structured_data = extraction['has_structured_data']
                
            except Exception as e:
                logger.warning(f"Content extraction failed for {result.url}: {e}")
                # Keep result with snippet only
        
        return results


def main():
    """Example usage"""
    searcher = AdvancedWebSearcher()
    
    query = "best practices for wheat cultivation in India"
    results = searcher.search(query, num_results=5)
    
    print(f"\nSearch results for: {query}\n")
    print("=" * 80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   URL: {result.url}")
        print(f"   Domain: {result.source_domain}")
        print(f"   Scores: Relevance={result.relevance_score:.1f}, "
              f"Credibility={result.credibility_score:.1f}, "
              f"Combined={result.combined_score:.1f}")
        print(f"   Words: {result.word_count}")
        print(f"   Snippet: {result.snippet[:150]}...")
        if result.content:
            print(f"   Content preview: {result.content[:200]}...")


if __name__ == "__main__":
    main()
