"""
Advanced Search Tool with Enhanced Web Search Integration

This module provides comprehensive web search capabilities with:
- Improved search result processing and integration
- Search failure handling and fallback mechanisms
- Rate limiting and respectful crawling
- Search result caching and optimization
"""

import time
import json
import hashlib
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse, urljoin
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from ddgs import DDGS
import logging

from .base_tool import BaseTool


@dataclass
class SearchResult:
    """Enhanced search result with comprehensive metadata"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    relevance_score: float = 0.0
    timestamp: float = 0.0
    source_domain: str = ""
    content_length: int = 0
    has_detailed_content: bool = False
    extraction_success: bool = False
    cache_hit: bool = False
    processing_time: float = 0.0
    error_message: Optional[str] = None


@dataclass
class SearchMetrics:
    """Search operation metrics for monitoring and optimization"""
    query: str
    total_time: float
    search_time: float
    processing_time: float
    cache_hits: int
    cache_misses: int
    successful_extractions: int
    failed_extractions: int
    rate_limit_delays: int
    fallback_attempts: int
    results_count: int


class SearchCache:
    """Thread-safe search result cache with TTL support"""
    
    def __init__(self, cache_dir: str = ".cache/search", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self._lock = threading.RLock()
        self._memory_cache = {}
        self._max_memory_items = 100
    
    def _get_cache_key(self, query: str, max_results: int, focus_area: str) -> str:
        """Generate cache key for search parameters"""
        cache_data = f"{query}:{max_results}:{focus_area}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, query: str, max_results: int, focus_area: str) -> Optional[List[SearchResult]]:
        """Get cached search results if available and not expired"""
        cache_key = self._get_cache_key(query, max_results, focus_area)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self._memory_cache:
                cached_data, timestamp = self._memory_cache[cache_key]
                if datetime.now() - timestamp < self.ttl:
                    return [SearchResult(**result) for result in cached_data]
                else:
                    del self._memory_cache[cache_key]
            
            # Check disk cache
            cache_file = self._get_cache_file(cache_key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    
                    cache_time = datetime.fromisoformat(cached_data['timestamp'])
                    if datetime.now() - cache_time < self.ttl:
                        results = [SearchResult(**result) for result in cached_data['results']]
                        # Update memory cache
                        self._memory_cache[cache_key] = (cached_data['results'], cache_time)
                        return results
                    else:
                        cache_file.unlink()  # Remove expired cache
                        
                except (json.JSONDecodeError, KeyError, ValueError):
                    cache_file.unlink()  # Remove corrupted cache
        
        return None
    
    def set(self, query: str, max_results: int, focus_area: str, results: List[SearchResult]):
        """Cache search results"""
        cache_key = self._get_cache_key(query, max_results, focus_area)
        timestamp = datetime.now()
        
        # Mark results as cached
        for result in results:
            result.cache_hit = False  # These are fresh results
        
        results_data = [asdict(result) for result in results]
        cache_data = {
            'timestamp': timestamp.isoformat(),
            'results': results_data
        }
        
        with self._lock:
            # Update memory cache
            if len(self._memory_cache) >= self._max_memory_items:
                # Remove oldest item
                oldest_key = min(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k][1])
                del self._memory_cache[oldest_key]
            
            self._memory_cache[cache_key] = (results_data, timestamp)
            
            # Update disk cache
            cache_file = self._get_cache_file(cache_key)
            try:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logging.warning(f"Failed to write search cache: {e}")


class RateLimiter:
    """Thread-safe rate limiter for respectful crawling"""
    
    def __init__(self, requests_per_second: float = 2.0, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self) -> float:
        """Acquire permission to make a request, returns delay time"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst_size, 
                            self.tokens + elapsed * self.requests_per_second)
            self.last_update = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return 0.0  # No delay needed
            else:
                # Calculate delay needed
                delay = (1.0 - self.tokens) / self.requests_per_second
                return delay


class AdvancedSearchTool(BaseTool):
    """
    Advanced search tool with comprehensive web search capabilities
    
    Features:
    - Intelligent search result processing and integration
    - Robust error handling with fallback mechanisms
    - Rate limiting for respectful crawling
    - Search result caching for optimization
    - Content extraction with quality assessment
    - Search query enhancement and optimization
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.search_config = config.get('search', {})
        
        # Initialize components
        self.cache = SearchCache(
            cache_dir=self.search_config.get('cache_dir', '.cache/search'),
            ttl_hours=self.search_config.get('cache_ttl_hours', 24)
        )
        
        self.rate_limiter = RateLimiter(
            requests_per_second=self.search_config.get('rate_limit_rps', 2.0),
            burst_size=self.search_config.get('rate_limit_burst', 5)
        )
        
        # Setup HTTP session with retry strategy
        self.session = self._create_session()
        
        # Setup logging
        self.logger = logging.getLogger(f'{__name__}.AdvancedSearchTool')
        
        # Search configuration
        self.max_retries = self.search_config.get('max_retries', 3)
        self.content_timeout = self.search_config.get('content_timeout', 10)
        self.max_content_length = self.search_config.get('max_content_length', 5000)
        self.min_content_quality = self.search_config.get('min_content_quality', 100)
        
        # Fallback search engines (if needed)
        self.fallback_enabled = self.search_config.get('enable_fallback', True)
        
    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy and proper headers"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set headers
        session.headers.update({
            'User-Agent': self.search_config.get('user_agent', 
                'Mozilla/5.0 (compatible; AdvancedSearchTool/1.0)'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
        return session
    
    @property
    def name(self) -> str:
        return "advanced_search_web"
    
    @property
    def description(self) -> str:
        return """Advanced web search with intelligent processing, caching, and rate limiting.
        Provides comprehensive search results with content extraction, relevance scoring,
        and robust error handling with fallback mechanisms."""
    
    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find information on the web"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5,
                    "minimum": 1,
                    "maximum": 20
                },
                "focus_area": {
                    "type": "string",
                    "description": "Search focus area for query enhancement",
                    "enum": ["general", "technical", "news", "academic", "recent"],
                    "default": "general"
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to extract full content from web pages",
                    "default": True
                },
                "use_cache": {
                    "type": "boolean",
                    "description": "Whether to use cached results if available",
                    "default": True
                },
                "quality_threshold": {
                    "type": "number",
                    "description": "Minimum content quality threshold (0.0-1.0)",
                    "default": 0.3,
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["query"]
        }
    
    def execute(self, query: str, max_results: int = 5, focus_area: str = "general",
                include_content: bool = True, use_cache: bool = True,
                quality_threshold: float = 0.3) -> Dict[str, Any]:
        """Execute advanced web search with comprehensive processing"""
        
        start_time = time.time()
        metrics = SearchMetrics(
            query=query,
            total_time=0.0,
            search_time=0.0,
            processing_time=0.0,
            cache_hits=0,
            cache_misses=0,
            successful_extractions=0,
            failed_extractions=0,
            rate_limit_delays=0,
            fallback_attempts=0,
            results_count=0
        )
        
        try:
            self.logger.info(f"Starting advanced search for: '{query}' (focus: {focus_area})")
            
            # Check cache first
            cached_results = None
            if use_cache:
                cached_results = self.cache.get(query, max_results, focus_area)
                if cached_results:
                    metrics.cache_hits = len(cached_results)
                    metrics.total_time = time.time() - start_time
                    metrics.results_count = len(cached_results)
                    
                    # Mark results as cache hits
                    for result in cached_results:
                        result.cache_hit = True
                    
                    self.logger.info(f"Cache hit: returning {len(cached_results)} cached results")
                    return self._format_response(cached_results, metrics, from_cache=True)
            
            # Perform fresh search
            search_results = self._perform_search(query, max_results, focus_area, metrics)
            
            if not search_results:
                # Try fallback if enabled
                if self.fallback_enabled:
                    self.logger.info("Primary search failed, attempting fallback")
                    metrics.fallback_attempts += 1
                    search_results = self._fallback_search(query, max_results, metrics)
            
            if not search_results:
                return self._format_error_response("No search results found", metrics)
            
            # Process and enhance results
            processed_results = self._process_results(
                search_results, include_content, quality_threshold, metrics
            )
            
            # Cache results if successful
            if use_cache and processed_results:
                self.cache.set(query, max_results, focus_area, processed_results)
            
            metrics.total_time = time.time() - start_time
            metrics.results_count = len(processed_results)
            
            self.logger.info(f"Search completed: {len(processed_results)} results in {metrics.total_time:.2f}s")
            
            return self._format_response(processed_results, metrics)
            
        except Exception as e:
            metrics.total_time = time.time() - start_time
            error_msg = f"Advanced search failed: {str(e)}"
            self.logger.error(error_msg)
            return self._format_error_response(error_msg, metrics)
    
    def _perform_search(self, query: str, max_results: int, focus_area: str, 
                       metrics: SearchMetrics) -> Optional[List[Dict]]:
        """Perform the actual web search with enhanced query processing"""
        
        search_start = time.time()
        
        try:
            # Enhance query based on focus area
            enhanced_query = self._enhance_query(query, focus_area)
            self.logger.debug(f"Enhanced query: {enhanced_query}")
            
            # Apply rate limiting
            delay = self.rate_limiter.acquire()
            if delay > 0:
                metrics.rate_limit_delays += 1
                self.logger.debug(f"Rate limiting: waiting {delay:.2f}s")
                time.sleep(delay)
            
            # Perform search with DDGS
            ddgs = DDGS()
            raw_results = list(ddgs.text(enhanced_query, max_results=max_results * 2))  # Get extra for filtering
            
            metrics.search_time = time.time() - search_start
            
            if not raw_results:
                self.logger.warning("No raw search results returned")
                return None
            
            # Filter and score results
            filtered_results = self._filter_and_score_results(raw_results, query, max_results)
            
            self.logger.info(f"Search returned {len(filtered_results)} filtered results")
            return filtered_results
            
        except Exception as e:
            metrics.search_time = time.time() - search_start
            self.logger.error(f"Search failed: {e}")
            return None
    
    def _enhance_query(self, query: str, focus_area: str) -> str:
        """Enhance search query based on focus area and current context"""
        
        current_year = datetime.now().year
        
        enhancements = {
            "general": query,
            "technical": f"{query} documentation technical specifications guide",
            "news": f"{query} news {current_year} latest recent developments",
            "academic": f"{query} research study academic paper analysis",
            "recent": f"{query} {current_year} latest update recent"
        }
        
        enhanced = enhancements.get(focus_area, query)
        
        # Add temporal context for better relevance
        if focus_area in ["news", "recent"]:
            enhanced += f" {current_year}"
        
        return enhanced
    
    def _filter_and_score_results(self, raw_results: List[Dict], 
                                 original_query: str, max_results: int) -> List[Dict]:
        """Filter and score search results for relevance and quality"""
        
        scored_results = []
        query_words = set(original_query.lower().split())
        
        for result in raw_results:
            try:
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(
                    result, query_words, original_query
                )
                
                # Add domain reputation score
                domain_score = self._get_domain_reputation_score(result.get('href', ''))
                
                # Combined score
                combined_score = (relevance_score * 0.7) + (domain_score * 0.3)
                
                # Add score to result
                result['relevance_score'] = combined_score
                result['domain_score'] = domain_score
                
                scored_results.append(result)
                
            except Exception as e:
                self.logger.warning(f"Error scoring result: {e}")
                continue
        
        # Sort by combined score and return top results
        scored_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        return scored_results[:max_results]
    
    def _calculate_relevance_score(self, result: Dict, query_words: set, 
                                  original_query: str) -> float:
        """Calculate relevance score for a search result"""
        
        title = result.get('title', '').lower()
        snippet = result.get('body', '').lower()
        url = result.get('href', '').lower()
        
        # Word overlap scoring
        title_words = set(title.split())
        snippet_words = set(snippet.split())
        url_words = set(url.replace('/', ' ').replace('-', ' ').split())
        
        title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
        snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
        url_overlap = len(query_words.intersection(url_words)) / len(query_words) if query_words else 0
        
        # Weighted scoring (title is most important)
        relevance_score = (title_overlap * 0.5) + (snippet_overlap * 0.3) + (url_overlap * 0.2)
        
        # Boost for exact phrase matches
        if original_query.lower() in title:
            relevance_score += 0.3
        elif original_query.lower() in snippet:
            relevance_score += 0.2
        
        return min(relevance_score, 1.0)
    
    def _get_domain_reputation_score(self, url: str) -> float:
        """Get domain reputation score based on known quality indicators"""
        
        if not url:
            return 0.0
        
        try:
            domain = urlparse(url).netloc.lower()
            
            # High-quality domains
            high_quality_domains = {
                'wikipedia.org': 0.9,
                'github.com': 0.8,
                'stackoverflow.com': 0.8,
                'docs.python.org': 0.9,
                'mozilla.org': 0.8,
                'w3.org': 0.9,
                'ietf.org': 0.9,
                'arxiv.org': 0.8,
                'nature.com': 0.9,
                'science.org': 0.9,
                'ieee.org': 0.8,
                'acm.org': 0.8,
            }
            
            # Check for exact matches
            if domain in high_quality_domains:
                return high_quality_domains[domain]
            
            # Check for domain patterns
            if any(pattern in domain for pattern in ['.edu', '.gov', '.org']):
                return 0.7
            elif any(pattern in domain for pattern in ['docs.', 'documentation', 'manual']):
                return 0.6
            elif any(pattern in domain for pattern in ['blog', 'medium.com', 'dev.to']):
                return 0.5
            else:
                return 0.4
                
        except Exception:
            return 0.4
    
    def _process_results(self, search_results: List[Dict], include_content: bool,
                        quality_threshold: float, metrics: SearchMetrics) -> List[SearchResult]:
        """Process search results with content extraction and quality assessment"""
        
        processing_start = time.time()
        processed_results = []
        
        for result in search_results:
            try:
                # Create base SearchResult
                search_result = SearchResult(
                    title=result.get('title', 'No title'),
                    url=result.get('href', ''),
                    snippet=result.get('body', 'No snippet'),
                    relevance_score=result.get('relevance_score', 0.0),
                    timestamp=time.time(),
                    source_domain=urlparse(result.get('href', '')).netloc
                )
                
                # Extract content if requested
                if include_content and search_result.url:
                    content_start = time.time()
                    content_result = self._extract_content(search_result.url)
                    
                    if content_result['success']:
                        search_result.content = content_result['content']
                        search_result.content_length = len(content_result['content'])
                        search_result.has_detailed_content = search_result.content_length > self.min_content_quality
                        search_result.extraction_success = True
                        metrics.successful_extractions += 1
                    else:
                        search_result.error_message = content_result['error']
                        search_result.extraction_success = False
                        metrics.failed_extractions += 1
                    
                    search_result.processing_time = time.time() - content_start
                
                # Apply quality threshold
                if search_result.relevance_score >= quality_threshold:
                    processed_results.append(search_result)
                    metrics.cache_misses += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing result: {e}")
                metrics.failed_extractions += 1
                continue
        
        metrics.processing_time = time.time() - processing_start
        return processed_results
    
    def _extract_content(self, url: str) -> Dict[str, Any]:
        """Extract content from webpage with robust error handling"""
        
        try:
            # Apply rate limiting for content extraction
            delay = self.rate_limiter.acquire()
            if delay > 0:
                time.sleep(delay)
            
            response = self.session.get(url, timeout=self.content_timeout)
            response.raise_for_status()
            
            # Parse content with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'advertisement', 'ads']):
                element.decompose()
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            # Clean and limit content
            cleaned_content = self._clean_content(content)
            
            if len(cleaned_content) > self.max_content_length:
                cleaned_content = self._truncate_content(cleaned_content)
            
            return {
                'success': True,
                'content': cleaned_content,
                'length': len(cleaned_content)
            }
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': 'Content extraction timeout'}
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Request failed: {str(e)}'}
        except Exception as e:
            return {'success': False, 'error': f'Content extraction failed: {str(e)}'}
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from parsed HTML"""
        
        # Try different content selectors in order of preference
        content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '.main-content',
            '.page-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                return ' '.join([elem.get_text() for elem in elements])
        
        # Fallback to body
        body = soup.find('body')
        if body:
            return body.get_text()
        
        return soup.get_text()
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        
        if not content:
            return ""
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'Cookie.*?Accept',
            r'Subscribe.*?Newsletter',
            r'Sign up.*?free',
            r'Advertisement',
            r'Skip to.*?content',
            r'Menu.*?Navigation',
            r'Share.*?Facebook',
            r'Follow.*?Twitter',
            r'Privacy Policy',
            r'Terms of Service'
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _truncate_content(self, content: str) -> str:
        """Intelligently truncate content at sentence boundaries"""
        
        if len(content) <= self.max_content_length:
            return content
        
        # Try to cut at sentence boundary
        sentences = content.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated + sentence + '. ') > self.max_content_length - 50:
                break
            truncated += sentence + '. '
        
        return truncated + "..."
    
    def _fallback_search(self, query: str, max_results: int, 
                        metrics: SearchMetrics) -> Optional[List[Dict]]:
        """Fallback search mechanism with simplified query"""
        
        try:
            # Simplify query for fallback
            simplified_query = self._simplify_query(query)
            self.logger.info(f"Fallback search with simplified query: {simplified_query}")
            
            # Apply rate limiting
            delay = self.rate_limiter.acquire()
            if delay > 0:
                time.sleep(delay)
            
            ddgs = DDGS()
            fallback_results = list(ddgs.text(simplified_query, max_results=max_results))
            
            if fallback_results:
                self.logger.info(f"Fallback search successful: {len(fallback_results)} results")
                return fallback_results
            
        except Exception as e:
            self.logger.error(f"Fallback search failed: {e}")
        
        return None
    
    def _simplify_query(self, query: str) -> str:
        """Simplify query for fallback search"""
        
        # Remove common stop words and keep key terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in query.lower().split() if word not in stop_words]
        
        # Keep first 3-4 most important words
        return ' '.join(words[:4])
    
    def _format_response(self, results: List[SearchResult], metrics: SearchMetrics, 
                        from_cache: bool = False) -> Dict[str, Any]:
        """Format successful search response"""
        
        return {
            "success": True,
            "results": [asdict(result) for result in results],
            "metrics": asdict(metrics),
            "from_cache": from_cache,
            "summary": {
                "total_results": len(results),
                "successful_extractions": metrics.successful_extractions,
                "failed_extractions": metrics.failed_extractions,
                "cache_hits": metrics.cache_hits,
                "processing_time": metrics.total_time,
                "average_relevance": sum(r.relevance_score for r in results) / len(results) if results else 0
            }
        }
    
    def _format_error_response(self, error_message: str, metrics: SearchMetrics) -> Dict[str, Any]:
        """Format error response"""
        
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "metrics": asdict(metrics),
            "from_cache": False,
            "summary": {
                "total_results": 0,
                "successful_extractions": 0,
                "failed_extractions": metrics.failed_extractions,
                "cache_hits": 0,
                "processing_time": metrics.total_time,
                "average_relevance": 0
            }
        }