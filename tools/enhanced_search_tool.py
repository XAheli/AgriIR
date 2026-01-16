from .base_tool import BaseTool
from .advanced_search_tool import AdvancedSearchTool, SearchResult
from ddgs import DDGS
from bs4 import BeautifulSoup
import requests
import json
import time
from urllib.parse import urlparse, urljoin
import re
import logging

class EnhancedSearchTool(BaseTool):
    def __init__(self, config: dict):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.get('search', {}).get('user_agent', 'Mozilla/5.0 (compatible; MakeItHeavy Agent)')
        })
        
        # Initialize advanced search tool for enhanced capabilities
        self.advanced_search = AdvancedSearchTool(config)
        self.use_advanced_features = config.get('search', {}).get('use_advanced_features', True)
        self.logger = logging.getLogger(f'{__name__}.EnhancedSearchTool')
    
    @property
    def name(self) -> str:
        return "search_web_enhanced"
    
    @property
    def description(self) -> str:
        return "Enhanced web search with content extraction, multiple sources, and intelligent filtering for comprehensive research"
    
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
                    "default": 5
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Whether to extract full content from web pages",
                    "default": True
                },
                "focus_area": {
                    "type": "string",
                    "description": "Specific area to focus on (e.g., 'technical', 'news', 'academic', 'general')",
                    "default": "general"
                }
            },
            "required": ["query"]
        }
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        text = ' '.join(text.split())
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'Cookie.*?Accept',
            r'Subscribe.*?Newsletter',
            r'Sign up.*?free',
            r'Advertisement',
            r'Skip to.*?content',
            r'Menu.*?Navigation'
        ]
        
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_content(self, url, timeout=10):
        """Extract meaningful content from a webpage"""
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement']):
                element.decompose()
            
            # Try to find main content areas
            content_selectors = [
                'article',
                'main',
                '[role="main"]',
                '.content',
                '.post-content',
                '.entry-content',
                '.article-content',
                '#content',
                '.main-content'
            ]
            
            content_text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    content_text = ' '.join([elem.get_text() for elem in elements])
                    break
            
            # Fallback to body if no specific content area found
            if not content_text:
                body = soup.find('body')
                if body:
                    content_text = body.get_text()
            
            # Clean and limit content
            cleaned_content = self.clean_text(content_text)
            
            # Limit content length but keep it substantial
            if len(cleaned_content) > 2000:
                # Try to cut at sentence boundary
                sentences = cleaned_content.split('. ')
                truncated = ""
                for sentence in sentences:
                    if len(truncated + sentence) > 1800:
                        break
                    truncated += sentence + ". "
                cleaned_content = truncated + "..."
            
            return cleaned_content
            
        except Exception as e:
            return f"Content extraction failed: {str(e)}"
    
    def enhance_query(self, query, focus_area):
        """Enhance search query based on focus area"""
        enhancements = {
            'technical': f'{query} technical documentation specifications',
            'news': f'{query} latest news 2024 recent developments',
            'academic': f'{query} research paper academic study',
            'general': query
        }
        
        return enhancements.get(focus_area, query)
    
    def execute(self, query: str, max_results: int = 5, include_content: bool = True, focus_area: str = "general") -> list:
        """Execute enhanced web search with advanced features"""
        try:
            # Ensure parameters are correct types
            max_results = int(max_results) if isinstance(max_results, str) else max_results
            include_content = bool(include_content) if isinstance(include_content, str) else include_content
            
            # Use advanced search if enabled, otherwise fall back to legacy implementation
            if self.use_advanced_features:
                self.logger.info(f"Using advanced search for: {query}")
                
                # Use advanced search tool with enhanced capabilities
                advanced_result = self.advanced_search.execute(
                    query=query,
                    max_results=max_results,
                    focus_area=focus_area,
                    include_content=include_content,
                    use_cache=True,
                    quality_threshold=0.3
                )
                
                if advanced_result['success']:
                    # Convert advanced results to legacy format for backward compatibility
                    enhanced_results = []
                    for result_data in advanced_result['results']:
                        enhanced_result = {
                            "title": result_data['title'],
                            "url": result_data['url'],
                            "snippet": result_data['snippet'],
                            "source": result_data['source_domain'],
                            "relevance_score": result_data['relevance_score'],
                            "content": result_data.get('content', 'Content extraction disabled'),
                            "content_length": result_data.get('content_length', 0),
                            "has_detailed_content": result_data.get('has_detailed_content', False),
                            "cache_hit": result_data.get('cache_hit', False),
                            "processing_time": result_data.get('processing_time', 0.0)
                        }
                        enhanced_results.append(enhanced_result)
                    
                    print(f"✅ Advanced search completed: {len(enhanced_results)} results")
                    if advanced_result.get('from_cache'):
                        print(f"   📋 Results served from cache")
                    
                    return enhanced_results
                else:
                    # Advanced search failed, fall back to legacy
                    self.logger.warning(f"Advanced search failed: {advanced_result.get('error')}, falling back to legacy")
                    return self._legacy_search(query, max_results, include_content, focus_area)
            else:
                # Use legacy search implementation
                return self._legacy_search(query, max_results, include_content, focus_area)
                
        except Exception as e:
            self.logger.error(f"Enhanced search failed: {e}")
            error_result = [{
                "error": f"Enhanced search failed: {str(e)}",
                "query": query,
                "focus_area": focus_area
            }]
            print(f"❌ Search error: {str(e)}")
            return error_result
    
    def _legacy_search(self, query: str, max_results: int, include_content: bool, focus_area: str) -> list:
        """Legacy search implementation for backward compatibility"""
        try:
            # Enhance query based on focus area
            enhanced_query = self.enhance_query(query, focus_area)
            
            print(f"🔍 Enhanced search (legacy): {enhanced_query}")
            
            # Perform search
            ddgs = DDGS()
            search_results = ddgs.text(enhanced_query, max_results=max_results)
            
            enhanced_results = []
            
            for i, result in enumerate(search_results):
                enhanced_result = {
                    "title": result.get('title', 'No title'),
                    "url": result.get('href', ''),
                    "snippet": result.get('body', 'No snippet'),
                    "source": urlparse(result.get('href', '')).netloc,
                    "relevance_score": max_results - i,  # Simple relevance scoring
                }
                
                # Extract full content if requested
                if include_content and result.get('href'):
                    print(f"   📄 Extracting content from: {enhanced_result['source']}")
                    content = self.extract_content(result['href'])
                    enhanced_result['content'] = content
                    
                    # Add content quality indicators
                    if content and not content.startswith("Content extraction failed"):
                        enhanced_result['content_length'] = len(content)
                        enhanced_result['has_detailed_content'] = len(content) > 500
                    else:
                        enhanced_result['content_length'] = 0
                        enhanced_result['has_detailed_content'] = False
                else:
                    enhanced_result['content'] = "Content extraction disabled"
                    enhanced_result['content_length'] = 0
                    enhanced_result['has_detailed_content'] = False
                
                enhanced_results.append(enhanced_result)
                
                # Small delay to be respectful to servers
                time.sleep(0.5)
            
            # Sort by relevance and content quality
            enhanced_results.sort(key=lambda x: (x['relevance_score'], x['content_length']), reverse=True)
            
            print(f"✅ Enhanced search (legacy) completed: {len(enhanced_results)} results")
            return enhanced_results
            
        except Exception as e:
            raise e