#!/usr/bin/env python3
"""
Enhanced Agriculture Data Curator - Inspired by Heavy Ollama
A multi-agent system for curating Indian agriculture, plants, and farming data
Uses parallel Ollama agents with web search tools to create a comprehensive dataset in JSONL format

Enhanced Features:
- No content extraction limits for complete dataset
- PDF download and processing with OCR support
- Intelligent search expansion using LLM reasoning
- Comprehensive duplicate prevention across agents
- Domain assignment to prevent overlap between agents
"""

import asyncio
import json
import time
import threading
import logging
import hashlib
import os
import sys
import re
import subprocess
import signal
import atexit
import io
import difflib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs

try:
    import requests
except ImportError:
    print("Warning: requests not installed. Install with: pip install requests")
    requests = None

try:
    from ddgs import DDGS
except ImportError:
    print("Warning: duckduckgo-search not installed. Install with: pip install duckduckgo-search")
    DDGS = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    print("Warning: beautifulsoup4 not installed. Install with: pip install beautifulsoup4")
    BeautifulSoup = None

try:
    import PyPDF2
except ImportError:
    print("Warning: PyPDF2 not installed. Install with: pip install pypdf2")
    PyPDF2 = None

try:
    import fitz  # pymupdf
except ImportError:
    print("Warning: pymupdf not installed. Install with: pip install pymupdf")
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("Warning: OCR libraries not installed. Install with: pip install pytesseract pillow")
    pytesseract = None
    Image = None

try:
    import magic
except ImportError:
    print("Warning: python-magic not installed. Install with: pip install python-magic")
    magic = None

# Import shared utilities
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from jsonl_writer import ImmediateJSONLWriter
from duplicate_tracker import get_global_tracker


class AgentStatus(Enum):
    """Status enumeration for curator agents"""
    QUEUED = "queued"
    INITIALIZING = "initializing" 
    SEARCHING = "searching"
    EXTRACTING = "extracting"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class CurationResult:
    """Result from agriculture data curation"""
    agent_id: int
    search_query: str
    port: int
    success: bool
    execution_time: float
    data_entries: List[Dict] = field(default_factory=list)
    search_results_count: int = 0
    processed_count: int = 0
    pdfs_processed: int = 0  # Added for PDF tracking
    error_message: Optional[str] = None
    status: AgentStatus = AgentStatus.COMPLETED


@dataclass 
class AgricultureDataEntry:
    """Structured agriculture data entry for JSONL output"""
    title: str
    author: Optional[str]
    link: str
    text_extracted: str
    abstract: Optional[str]
    genre: str  # survey, dataset, pdf, book, report, article
    tags: List[str]
    indian_regions: List[str]
    crop_types: List[str] = field(default_factory=list)
    farming_methods: List[str] = field(default_factory=list)
    soil_types: List[str] = field(default_factory=list)  # Enhanced field
    climate_info: List[str] = field(default_factory=list)  # Enhanced field
    fertilizers: List[str] = field(default_factory=list)  # Enhanced field
    watering_schedule: Optional[str] = None  # Enhanced field
    plant_species: List[str] = field(default_factory=list)  # Enhanced field
    data_type: str = ""  # statistical, qualitative, mixed
    publication_year: Optional[int] = None
    source_domain: str = ""
    extraction_timestamp: str = ""
    relevance_score: float = 0.0
    content_length: int = 0
    content_hash: str = ""  # For duplicate detection
    url_hash: str = ""      # For URL duplicate detection
    pdf_path: Optional[str] = None  # Path to downloaded PDF
    is_pdf: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSONL export"""
        return asdict(self)


class DuplicateTracker:
    """Wrapper around global persistent duplicate tracker for backward compatibility"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.agent_url_assignments = {}  # Track which agent is handling which domains
        # Use global persistent tracker
        self._tracker = get_global_tracker("data/processed_urls.json")
        
    def assign_domains_to_agents(self, num_agents: int) -> Dict[int, List[str]]:
        """Assign different domains/patterns to different agents to prevent overlap"""
        domain_patterns = [
            # Government and research institutions
            ['gov.in', 'nic.in', 'icar.org', 'iari.res.in', 'icrisat.org'],
            # Academic institutions
            ['edu', 'ac.in', 'researchgate.net', 'academia.edu', 'ieee.org'],
            # International organizations
            ['fao.org', 'worldbank.org', 'un.org', 'cgiar.org', 'ifpri.org'],
            # Commercial and news
            ['com', 'org', 'net', 'business-standard.com', 'economictimes.com']
        ]
        
        assignments = {}
        for i in range(num_agents):
            assignments[i] = domain_patterns[i % len(domain_patterns)]
            
        self.agent_url_assignments = assignments
        return assignments
    
    def should_agent_process_url(self, agent_id: int, url: str) -> bool:
        """Check if an agent should process a specific URL based on domain assignment"""
        if agent_id not in self.agent_url_assignments:
            return True  # No assignment, process all
            
        assigned_patterns = self.agent_url_assignments[agent_id]
        url_lower = url.lower()
        
        return any(pattern in url_lower for pattern in assigned_patterns)
    
    def is_duplicate_url(self, url: str) -> bool:
        """Check if URL is already processed"""
        return self._tracker.is_duplicate_url(url)
    
    def is_duplicate_content(self, title: str, content: str) -> bool:
        """Check if content is duplicate using title and content hash"""
        return self._tracker.is_duplicate_content(title, content)
    
    def mark_processed(self, url: str, title: str = "", content: str = "") -> bool:
        """Mark URL/content as processed"""
        return self._tracker.mark_processed(url, title, content)


class ImprovedPDFProcessor:
    """Enhanced PDF processor with better text extraction, OCR, and immediate JSONL writing"""
    
    def __init__(self, storage_dir: str = "downloaded_pdfs", max_size_mb: int = 50,
                 jsonl_writer: Optional[ImmediateJSONLWriter] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True, parents=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.jsonl_writer = jsonl_writer
        
        # Agriculture-specific keywords for structured extraction
        self.soil_keywords = [
            'black soil', 'red soil', 'alluvial soil', 'laterite soil', 'sandy soil',
            'clay soil', 'loamy soil', 'saline soil', 'alkaline soil', 'acidic soil',
            'soil ph', 'soil fertility', 'soil organic matter', 'soil nutrients'
        ]
        
        self.climate_keywords = [
            'tropical climate', 'temperate climate', 'arid climate', 'semi-arid',
            'monsoon', 'rainfall', 'temperature', 'humidity', 'drought',
            'kharif season', 'rabi season', 'zaid season', 'irrigation'
        ]
        
        self.fertilizer_keywords = [
            'urea', 'dap', 'potash', 'nitrogen', 'phosphorus', 'potassium',
            'organic fertilizer', 'compost', 'manure', 'bio-fertilizer',
            'vermicompost', 'green manure', 'micronutrients'
        ]
        
        self.plant_species_keywords = [
            'varieties', 'cultivars', 'hybrid seeds', 'indigenous varieties',
            'high yielding varieties', 'drought resistant', 'pest resistant',
            'basmati rice', 'indica rice', 'japonica rice'
        ]
            
        content_snippet = content[:500] if content else ""
        
        for existing_snippet in self.content_snippets[-100:]:  # Check last 100 for efficiency
            similarity = self._calculate_similarity(content_snippet, existing_snippet)
            if similarity > self.similarity_threshold:
                return True
                
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()


class PDFProcessor:
    """PDF download and text extraction with OCR support"""
    
    def __init__(self, storage_dir: str = "downloaded_pdfs", max_size_mb: int = 50):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        
        if requests is None:
            logging.warning("Requests not available - PDF downloading disabled")
        
    def download_and_process_pdf(self, url: str, title: str = "") -> Optional[Dict]:
        """Download PDF and extract text with OCR fallback"""
        if requests is None:
            return None
            
        try:
            # Check if URL points to PDF
            if not self._is_pdf_url(url):
                return None
                
            # Generate safe filename
            safe_title = re.sub(r'[^\w\-_\.]', '_', title[:100])
            filename = f"{safe_title}_{int(time.time())}.pdf"
            filepath = self.storage_dir / filename
            
            # Download PDF
            if not self._download_file(url, filepath):
                return None
                
            # Extract text
            text_content = self._extract_text_from_pdf(filepath)
            if not text_content or len(text_content.strip()) < 100:
                # Try OCR if text extraction failed
                text_content = self._ocr_pdf(filepath)
                
            if not text_content:
                logging.warning(f"Failed to extract text from PDF: {filepath}")
                return None
                
            # Extract metadata
            metadata = self._extract_pdf_metadata(filepath)
            
            return {
                'filepath': str(filepath),
                'text_content': text_content,
                'metadata': metadata,
                'file_size': filepath.stat().st_size,
                'extraction_method': 'text' if text_content else 'ocr'
            }
            
        except Exception as e:
            logging.error(f"PDF processing failed for {url}: {e}")
            return None
    
    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF"""
        if url.lower().endswith('.pdf'):
            return True
            
        # Check content-type header
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            content_type = response.headers.get('content-type', '').lower()
            return 'pdf' in content_type
        except:
            return False
    
    def _download_file(self, url: str, filepath: Path) -> bool:
        """Download file with size limit"""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Check file size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_size_bytes:
                logging.warning(f"PDF too large: {content_length} bytes > {self.max_size_bytes}")
                return False
                
            downloaded_size = 0
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > self.max_size_bytes:
                            logging.warning(f"PDF download exceeded size limit: {filepath}")
                            filepath.unlink()  # Delete partial file
                            return False
                        f.write(chunk)
                        
            return True
            
        except Exception as e:
            logging.error(f"PDF download failed: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def _extract_text_from_pdf(self, filepath: Path) -> str:
        """Extract text from PDF using PyPDF2 and pymupdf"""
        text_content = ""
        
        # Try PyMuPDF first (better text extraction)
        if fitz:
            try:
                doc = fitz.open(str(filepath))
                for page in doc:
                    text_content += page.get_text()
                doc.close()
                
                if text_content.strip():
                    return text_content
            except Exception as e:
                logging.debug(f"PyMuPDF extraction failed: {e}")
        
        # Fallback to PyPDF2
        if PyPDF2:
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text_content += page.extract_text()
                        
                if text_content.strip():
                    return text_content
            except Exception as e:
                logging.debug(f"PyPDF2 extraction failed: {e}")
                
        return text_content
    
    def _ocr_pdf(self, filepath: Path) -> str:
        """Extract text using OCR for scanned PDFs"""
        if not pytesseract or not Image or not fitz:
            return ""
            
        try:
            doc = fitz.open(str(filepath))
            text_content = ""
            
            for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages for OCR
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("ppm")
                
                # Convert to PIL Image
                img = Image.open(io.BytesIO(img_data))
                
                # Perform OCR
                page_text = pytesseract.image_to_string(img, lang='eng')
                text_content += page_text + "\n"
                
            doc.close()
            return text_content
            
        except Exception as e:
            logging.error(f"OCR processing failed: {e}")
            return ""
    
    def _extract_pdf_metadata(self, filepath: Path) -> Dict:
        """Extract metadata from PDF"""
        metadata = {}
        
        if fitz:
            try:
                doc = fitz.open(str(filepath))
                pdf_metadata = doc.metadata
                metadata.update({
                    'title': pdf_metadata.get('title', ''),
                    'author': pdf_metadata.get('author', ''),
                    'subject': pdf_metadata.get('subject', ''),
                    'creator': pdf_metadata.get('creator', ''),
                    'producer': pdf_metadata.get('producer', ''),
                    'creation_date': pdf_metadata.get('creationDate', ''),
                    'modification_date': pdf_metadata.get('modDate', ''),
                    'page_count': doc.page_count
                })
                doc.close()
            except Exception as e:
                logging.debug(f"Metadata extraction failed: {e}")
                
        return metadata


class IntelligentSearchExpansion:
    """Generate additional search queries using LLM reasoning"""
    
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url
        
    def generate_expanded_queries(self, base_queries: List[str], max_new_queries: int = 30) -> List[str]:
        """Generate additional search queries using LLM"""
        expansion_prompt = self._create_expansion_prompt(base_queries)
        
        try:
            if requests is None:
                return []
                
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": expansion_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                },
                timeout=60
            )
            
            if response.status_code == 200:
                llm_output = response.json().get('response', '')
                return self._parse_query_suggestions(llm_output, max_new_queries)
                
        except Exception as e:
            logging.error(f"Query expansion failed: {e}")
            
        return []
    
    def _create_expansion_prompt(self, base_queries: List[str]) -> str:
        """Create prompt for LLM to generate additional search queries"""
        sample_queries = base_queries[:10]  # Show sample of existing queries
        
        return f"""
You are an agricultural research expert specializing in Indian farming and agriculture. 

I have been searching for comprehensive data on Indian agriculture using these search queries:

{chr(10).join(f"- {q}" for q in sample_queries)}

I need you to think of additional search queries that would help me find MORE comprehensive data on Indian agriculture, farming, and related topics that I might have missed. Focus on:

1. Emerging agricultural technologies in India
2. Traditional farming practices that are being documented
3. Agricultural challenges specific to different Indian regions
4. Economic aspects of farming (credit, insurance, marketing)
5. Environmental and climate issues affecting Indian agriculture
6. Government schemes and policies for farmers
7. Research studies on specific crops or farming methods
8. Agricultural education and extension services
9. Women's participation in agriculture
10. Agricultural cooperatives and self-help groups
11. Post-harvest management and food processing
12. Agricultural machinery and mechanization
13. Seed varieties and crop breeding research
14. Integrated pest management studies
15. Soil health and fertility management

Generate exactly 30 new search queries that are:
- Specific to Indian agriculture
- Different from the queries I already have
- Likely to find valuable research papers, reports, datasets, or articles
- Covering aspects that farmers, researchers, or policymakers would need

Format your response as a numbered list:
1. [query]
2. [query]
...
30. [query]

Each query should be 5-12 words long and include "India" or "Indian" for geographic specificity.
"""
    
    def _parse_query_suggestions(self, llm_output: str, max_queries: int) -> List[str]:
        """Parse LLM output to extract search queries"""
        queries = []
        lines = llm_output.split('\n')
        
        for line in lines:
            line = line.strip()
            # Look for numbered list items
            if re.match(r'^\d+\.', line):
                query = re.sub(r'^\d+\.\s*', '', line).strip()
                if query and len(query) > 10:  # Ensure meaningful queries
                    queries.append(query)
                    
            # Also look for bullet points
            elif line.startswith('- ') or line.startswith('* '):
                query = line[2:].strip()
                if query and len(query) > 10:
                    queries.append(query)
                    
        return queries[:max_queries]


class AgricultureSearchQueries:
    """Predefined search queries for Indian agriculture data"""
    
    BASE_QUERIES = [
        # Crop-specific searches
        "Indian rice cultivation surveys datasets research papers",
        "wheat farming India agricultural reports statistics",
        "cotton agriculture India production data analysis",
        "sugarcane cultivation Indian states research studies",
        "pulse crops India legumes farming data reports",
        "millets cultivation India traditional crops research",
        "spices agriculture India cardamom pepper turmeric data",
        "fruit cultivation India mango banana citrus reports",
        "vegetable farming India potato onion tomato surveys",
        "tea coffee plantation India cultivation data research",
        
        # Regional agriculture
        "Punjab agriculture Green Revolution data reports",
        "Maharashtra farming cotton sugarcane research",
        "Tamil Nadu agriculture rice cultivation studies",
        "Karnataka agriculture horticulture data reports",
        "Uttar Pradesh farming wheat rice surveys",
        "West Bengal agriculture rice jute research data",
        "Gujarat agriculture cotton groundnut reports",
        "Rajasthan agriculture desert farming data studies",
        "Kerala agriculture spices coconut research",
        "Andhra Pradesh agriculture rice cotton reports",
        
        # Farming techniques and methods
        "organic farming India research data reports",
        "precision agriculture India technology adoption",
        "sustainable farming practices India studies",
        "irrigation systems India water management data",
        "crop rotation India traditional practices research",
        "integrated pest management India studies",
        "soil health India agriculture research reports",
        "climate change agriculture India adaptation studies",
        "drought resistant crops India research data",
        "monsoon agriculture India rainfall correlation",
        
        # Economic and policy aspects
        "agricultural economics India farm income surveys",
        "crop insurance India farmers data reports",
        "agricultural subsidies India policy impact studies",
        "farm mechanization India adoption reports",
        "agricultural marketing India supply chain data",
        "food security India agriculture production",
        "farmer suicide India agriculture crisis studies",
        "women agriculture India participation research",
        "agricultural credit India rural banking data",
        "MSP minimum support price India crop data",
        
        # Research institutions and datasets
        "ICAR Indian Council Agricultural Research data",
        "ICRISAT agriculture research India reports",
        "IARI Indian Agricultural Research Institute studies",
        "agricultural statistics India government data",
        "crop production statistics India state wise",
        "livestock agriculture India dairy research",
        "fisheries aquaculture India coastal agriculture",
        "forestry agroforestry India research data",
        "horticulture India fruits vegetables reports",
        "agricultural census India land holdings data"
    ]
    
    @classmethod
    def get_search_queries(cls, num_queries: Optional[int] = None) -> List[str]:
        """Get search queries, optionally limited to a specific number"""
        if num_queries:
            return cls.BASE_QUERIES[:num_queries]
        return cls.BASE_QUERIES


class EnhancedWebSearch:
    """Enhanced web search with content extraction for agriculture data"""
    
    def __init__(self, max_results: int = 25, pdf_processor: Optional[PDFProcessor] = None):
        self.max_results = max_results
        self.pdf_processor = pdf_processor
        
        if requests is None:
            raise ImportError("requests library is required. Install with: pip install requests")
            
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Agriculture-specific domain priorities
        self.priority_domains = [
            'icar.org.in', 'iari.res.in', 'icrisat.org', 'agricoop.nic.in',
            'indiastat.com', 'mospi.gov.in', 'niti.gov.in', 'fao.org',
            'worldbank.org', 'researchgate.net', 'springer.com', 'sciencedirect.com',
            'jstor.org', 'plos.org', 'nature.com', 'ieee.org'
        ]
        
        # Indian state patterns for region extraction
        self.indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand',
            'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
            'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
            'uttar pradesh', 'uttarakhand', 'west bengal'
        ]
        
        # Crop type patterns
        self.crop_patterns = [
            'rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'bajra', 'jowar',
            'ragi', 'pulses', 'gram', 'tur', 'lentil', 'groundnut', 'soybean',
            'mustard', 'sunflower', 'tea', 'coffee', 'rubber', 'coconut',
            'cardamom', 'pepper', 'turmeric', 'ginger', 'mango', 'banana'
        ]
    
    def search_and_extract(self, query: str, duplicate_tracker: Optional[DuplicateTracker] = None, agent_id: int = 0) -> List[Dict]:
        """Search for agriculture data and extract relevant information"""
        results = []
        
        if DDGS is None:
            logging.error("DDGS not available. Install with: pip install duckduckgo-search")
            return results
        
        try:
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    query, 
                    max_results=self.max_results,
                    region='in'  # Focus on Indian results
                ))
                
                for result in search_results:
                    # Check domain assignment if duplicate tracker is available
                    if duplicate_tracker and not duplicate_tracker.should_agent_process_url(agent_id, result.get('href', '')):
                        continue
                        
                    # Check for URL duplicates
                    if duplicate_tracker and duplicate_tracker.is_duplicate_url(result.get('href', '')):
                        continue
                        
                    extracted_data = self._extract_content(result)
                    if extracted_data:
                        # Check for content duplicates
                        if duplicate_tracker and duplicate_tracker.is_duplicate_content(
                            extracted_data.get('title', ''), 
                            extracted_data.get('full_content', '')
                        ):
                            continue
                            
                        results.append(extracted_data)
                        
        except Exception as e:
            logging.error(f"Search failed for query '{query}': {e}")
            
        return results
    
    def _extract_content(self, search_result: Dict) -> Optional[Dict]:
        """Extract and structure content from search result"""
        try:
            url = search_result.get('href', '')
            title = search_result.get('title', '')
            snippet = search_result.get('body', '')
            
            # Determine content type/genre
            genre = self._classify_content_type(url, title, snippet)
            
            # Check if this is a PDF
            is_pdf = url.lower().endswith('.pdf') or 'pdf' in url.lower()
            pdf_data = None
            
            # Process PDF if available
            if is_pdf and self.pdf_processor:
                pdf_data = self.pdf_processor.download_and_process_pdf(url, title)
                
            # Extract full content (no limits for complete dataset)
            if pdf_data:
                full_content = pdf_data['text_content']
            else:
                full_content = self._fetch_full_content(url)
                
            if not full_content:
                full_content = snippet
            
            # Extract metadata
            regions = self._extract_regions(full_content + " " + title + " " + snippet)
            crops = self._extract_crops(full_content + " " + title + " " + snippet)
            tags = self._generate_tags(full_content, title, snippet, genre)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance(full_content, title, snippet)
            
            result = {
                'title': title,
                'url': url,
                'snippet': snippet,
                'full_content': full_content,
                'genre': genre,
                'indian_regions': regions,
                'crop_types': crops,
                'tags': tags,
                'relevance_score': relevance_score,
                'source_domain': self._extract_domain(url),
                'content_length': len(full_content),
                'is_pdf': is_pdf,
                'pdf_path': pdf_data['filepath'] if pdf_data else None,
                'content_hash': hashlib.md5(full_content.encode('utf-8')).hexdigest(),
                'url_hash': hashlib.md5(url.encode('utf-8')).hexdigest()
            }
            
            # Add PDF metadata if available
            if pdf_data:
                result['pdf_metadata'] = pdf_data['metadata']
                result['pdf_extraction_method'] = pdf_data['extraction_method']
            
            return result
            
        except Exception as e:
            logging.error(f"Content extraction failed for {search_result.get('href', 'unknown')}: {e}")
            return None
    
    def _classify_content_type(self, url: str, title: str, snippet: str) -> str:
        """Classify the type of content (survey, dataset, pdf, book, report, article)"""
        text = (title + " " + snippet + " " + url).lower()
        
        if any(term in text for term in ['survey', 'questionnaire', 'census']):
            return 'survey'
        elif any(term in text for term in ['dataset', 'data', 'statistics', 'csv', 'database']):
            return 'dataset'
        elif any(term in text for term in ['pdf', '.pdf', 'document']):
            return 'pdf'
        elif any(term in text for term in ['book', 'handbook', 'manual', 'guide']):
            return 'book'
        elif any(term in text for term in ['report', 'annual', 'study', 'analysis']):
            return 'report'
        else:
            return 'article'
    
    def _fetch_full_content(self, url: str) -> str:
        """Fetch and extract full content from URL - NO LENGTH LIMITS"""
        if BeautifulSoup is None:
            logging.debug("BeautifulSoup not available. Install with: pip install beautifulsoup4")
            return ""
            
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # NO LENGTH LIMIT - return full content for complete dataset
            return text if text else ""
            
        except Exception as e:
            logging.debug(f"Failed to fetch full content from {url}: {e}")
            return ""
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract Indian regions/states mentioned in text"""
        text_lower = text.lower()
        regions = []
        
        for state in self.indian_states:
            if state in text_lower:
                regions.append(state.title())
        
        return list(set(regions))
    
    def _extract_crops(self, text: str) -> List[str]:
        """Extract crop types mentioned in text"""
        text_lower = text.lower()
        crops = []
        
        for crop in self.crop_patterns:
            if crop in text_lower:
                crops.append(crop)
        
        return list(set(crops))
    
    def _generate_tags(self, content: str, title: str, snippet: str, genre: str) -> List[str]:
        """Generate relevant tags for the content"""
        text = (content + " " + title + " " + snippet).lower()
        tags = [genre]
        
        # Agriculture-specific tags
        if any(term in text for term in ['organic', 'sustainable']):
            tags.append('sustainable agriculture')
        if any(term in text for term in ['irrigation', 'water']):
            tags.append('irrigation')
        if any(term in text for term in ['climate', 'weather', 'monsoon']):
            tags.append('climate')
        if any(term in text for term in ['technology', 'precision', 'digital']):
            tags.append('agricultural technology')
        if any(term in text for term in ['policy', 'government', 'scheme']):
            tags.append('agricultural policy')
        if any(term in text for term in ['farmer', 'farming', 'cultivation']):
            tags.append('farming practices')
        if any(term in text for term in ['soil', 'fertility', 'nutrients']):
            tags.append('soil management')
        if any(term in text for term in ['pest', 'disease', 'protection']):
            tags.append('crop protection')
        
        return tags
    
    def _calculate_relevance(self, content: str, title: str, snippet: str) -> float:
        """Calculate relevance score for agriculture context"""
        text = (content + " " + title + " " + snippet).lower()
        score = 0.0
        
        # Base relevance keywords
        agriculture_terms = ['agriculture', 'farming', 'crop', 'cultivation', 'harvest']
        india_terms = ['india', 'indian', 'bharatiya']
        
        for term in agriculture_terms:
            score += text.count(term) * 0.1
        
        for term in india_terms:
            score += text.count(term) * 0.05
            
        # Bonus for priority domains
        domain = self._extract_domain(snippet)
        if any(priority in domain for priority in self.priority_domains):
            score += 0.5
            
        return min(score, 1.0)  # Cap at 1.0
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""


class OllamaServerManager:
    """Simplified Ollama server manager for agriculture data curation"""
    
    def __init__(self, max_instances: int = 4, model: str = "deepseek-r1:70b"):
        self.max_instances = max_instances
        self.model = model
        self.base_port = 11434
        self.ports = list(range(self.base_port, self.base_port + max_instances))
        self.running_instances = []
        self.processes = {}
        
        atexit.register(self.cleanup_all)
    
    def start_instance(self, port: int) -> bool:
        """Start an Ollama instance on specified port"""
        try:
            env = os.environ.copy()
            env['OLLAMA_HOST'] = f'localhost:{port}'
            
            process = subprocess.Popen(
                ['ollama', 'serve'],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.processes[port] = process
            
            # Wait for startup
            time.sleep(5)
            
            # Verify instance is running
            if self._health_check(port):
                self.running_instances.append(port)
                logging.info(f"Ollama instance started on port {port}")
                return True
            else:
                process.terminate()
                return False
                
        except Exception as e:
            logging.error(f"Failed to start Ollama instance on port {port}: {e}")
            return False
    
    def start_all_instances(self) -> List[int]:
        """Start all required Ollama instances"""
        for port in self.ports:
            self.start_instance(port)
        
        return self.running_instances
    
    def _health_check(self, port: int) -> bool:
        """Check if Ollama instance is healthy"""
        if requests is None:
            return False
            
        try:
            response = requests.get(f'http://localhost:{port}/api/tags', timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def cleanup_all(self):
        """Clean up all running instances"""
        for process in self.processes.values():
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                pass


class AgricultureDataAgent:
    """Individual agent for agriculture data curation with enhanced capabilities"""
    
    def __init__(self, agent_id: int, port: int, model: str, search_engine: EnhancedWebSearch, 
                 duplicate_tracker: DuplicateTracker, search_expansion: IntelligentSearchExpansion):
        self.agent_id = agent_id
        self.port = port
        self.model = model
        self.search_engine = search_engine
        self.duplicate_tracker = duplicate_tracker
        self.search_expansion = search_expansion
        self.base_url = f"http://localhost:{port}"
        
        # Agent specialization
        self.specializations = {
            0: "crop production and statistics",
            1: "sustainable farming practices", 
            2: "agricultural policy and economics",
            3: "agricultural technology and innovation"
        }
        
        self.specialization = self.specializations.get(agent_id % 4, "general agriculture")
    
    def curate_data(self, search_queries: List[str], expand_search: bool = True) -> CurationResult:
        """Curate agriculture data for assigned search queries with intelligent expansion"""
        start_time = time.time()
        data_entries = []
        total_search_results = 0
        all_queries = search_queries.copy()
        
        try:
            # Expand search queries using LLM if requested
            if expand_search and len(search_queries) > 0:
                logging.info(f"Agent {self.agent_id}: Expanding search queries...")
                expanded_queries = self.search_expansion.generate_expanded_queries(
                    search_queries, max_new_queries=10  # 10 additional queries per agent
                )
                all_queries.extend(expanded_queries)
                logging.info(f"Agent {self.agent_id}: Added {len(expanded_queries)} expanded queries")
            
            for query in all_queries:
                # Modify query based on agent specialization
                specialized_query = self._specialize_query(query)
                
                # Search and extract data with duplicate checking
                search_results = self.search_engine.search_and_extract(
                    specialized_query, 
                    self.duplicate_tracker, 
                    self.agent_id
                )
                total_search_results += len(search_results)
                
                # Process each search result
                for result in search_results:
                    processed_entry = self._process_search_result(result, specialized_query)
                    if processed_entry:
                        data_entries.append(processed_entry)
                
                # Delay between searches to be respectful
                time.sleep(2)
            
            execution_time = time.time() - start_time
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Processed {len(all_queries)} queries ({len(search_queries)} base + {len(all_queries) - len(search_queries)} expanded)",
                port=self.port,
                success=True,
                execution_time=execution_time,
                data_entries=data_entries,
                search_results_count=total_search_results,
                processed_count=len(data_entries),
                status=AgentStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Agent {self.agent_id} failed: {e}")
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Failed processing {len(all_queries)} queries",
                port=self.port,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                status=AgentStatus.FAILED
            )
    
    def _specialize_query(self, query: str) -> str:
        """Add specialization context to search query"""
        specialization_terms = {
            "crop production and statistics": "statistics data production yield",
            "sustainable farming practices": "organic sustainable eco-friendly practices",
            "agricultural policy and economics": "policy economics government schemes subsidy",
            "agricultural technology and innovation": "technology innovation precision digital"
        }
        
        terms = specialization_terms.get(self.specialization, "")
        return f"{query} {terms}"
    
    def _process_search_result(self, result: Dict, query: str) -> Optional[Dict]:
        """Process search result into structured data entry"""
        try:
            # Use LLM to extract structured information if available
            structured_data = self._extract_with_llm(result)
            if structured_data:
                return structured_data
            
            # Fallback to rule-based extraction
            return self._extract_with_rules(result, query)
            
        except Exception as e:
            logging.error(f"Failed to process search result: {e}")
            return None
    
    def _extract_with_llm(self, result: Dict) -> Optional[Dict]:
        """Use LLM to extract structured information (if Ollama is available)"""
        if requests is None:
            return None
            
        try:
            prompt = self._create_extraction_prompt(result)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                llm_output = response.json().get('response', '')
                return self._parse_llm_output(llm_output, result)
            
        except Exception as e:
            logging.debug(f"LLM extraction failed: {e}")
        
        return None
    
    def _create_extraction_prompt(self, result: Dict) -> str:
        """Create prompt for LLM to extract structured information"""
        content = result.get('full_content', result.get('snippet', ''))
        content_sample = content[:3000] if content else ""  # Use larger sample for better analysis
        
        return f"""
Extract structured information about Indian agriculture from the following content:

Title: {result.get('title', '')}
URL: {result.get('url', '')}
Content: {content_sample}

Extract the following information in JSON format:
- author: Author name if available
- abstract: Brief abstract or summary (2-3 sentences)
- publication_year: Year of publication if mentioned
- farming_methods: List of farming methods mentioned
- data_type: "statistical", "qualitative", or "mixed"

Respond only with valid JSON:
"""
    
    def _parse_llm_output(self, llm_output: str, result: Dict) -> Optional[Dict]:
        """Parse LLM output and combine with existing data"""
        try:
            # Try to extract JSON from LLM output
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                llm_data = json.loads(json_match.group())
                
                # Combine with existing result data
                return {
                    **result,
                    'author': llm_data.get('author'),
                    'abstract': llm_data.get('abstract'),
                    'publication_year': llm_data.get('publication_year'),
                    'farming_methods': llm_data.get('farming_methods', []),
                    'data_type': llm_data.get('data_type', 'mixed'),
                    'text_extracted': result.get('full_content', result.get('snippet', '')),
                    'extraction_timestamp': datetime.now().isoformat()
                }
        except:
            pass
        
        return None
    
    def _extract_with_rules(self, result: Dict, query: str) -> Dict:
        """Rule-based extraction as fallback"""
        return {
            'title': result.get('title', ''),
            'author': None,
            'link': result.get('url', ''),
            'text_extracted': result.get('full_content', result.get('snippet', '')),
            'abstract': result.get('snippet', '')[:500] if result.get('snippet') else None,
            'genre': result.get('genre', 'article'),
            'tags': result.get('tags', []),
            'indian_regions': result.get('indian_regions', []),
            'crop_types': result.get('crop_types', []),
            'farming_methods': [],
            'data_type': 'mixed',
            'publication_year': None,
            'source_domain': result.get('source_domain', ''),
            'extraction_timestamp': datetime.now().isoformat(),
            'relevance_score': result.get('relevance_score', 0.0),
            'content_length': result.get('content_length', 0),
            'content_hash': result.get('content_hash', ''),
            'url_hash': result.get('url_hash', ''),
            'is_pdf': result.get('is_pdf', False),
            'pdf_path': result.get('pdf_path')
        }


class AgricultureDataCurator:
    """Main agriculture data curation system with enhanced capabilities"""
    
    def __init__(self, 
                 num_agents: int = 4, 
                 model: str = "deepseek-r1:70b",
                 output_file: str = "indian_agriculture_data.jsonl",
                 max_search_results: int = 25,
                 pdf_storage_dir: str = "downloaded_pdfs",
                 enable_pdf_download: bool = True,
                 enable_intelligent_expansion: bool = True):
        
        self.num_agents = num_agents
        self.model = model
        self.output_file = output_file
        self.max_search_results = max_search_results
        self.enable_pdf_download = enable_pdf_download
        self.enable_intelligent_expansion = enable_intelligent_expansion
        
        # Initialize components
        self.server_manager = OllamaServerManager(num_agents, model)
        self.duplicate_tracker = DuplicateTracker()
        
        # Initialize PDF processor if enabled
        self.pdf_processor = PDFProcessor(pdf_storage_dir) if enable_pdf_download else None
        
        # Initialize search engine
        self.search_engine = EnhancedWebSearch(max_search_results, self.pdf_processor)
        
        # Initialize search expansion
        self.search_expansion = IntelligentSearchExpansion(model, f"http://localhost:11434")
        
        self.agents = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agriculture_curator.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create output directory
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF storage directory if needed
        if enable_pdf_download:
            Path(pdf_storage_dir).mkdir(parents=True, exist_ok=True)
    
    def start_curation(self, num_queries: Optional[int] = None) -> Dict:
        """Start the enhanced agriculture data curation process"""
        logging.info("Starting Enhanced Agriculture Data Curation System")
        
        # Start Ollama instances
        logging.info("Starting Ollama instances...")
        running_ports = self.server_manager.start_all_instances()
        
        if not running_ports:
            logging.error("No Ollama instances started successfully")
            return {"success": False, "error": "Failed to start Ollama instances"}
        
        logging.info(f"Started {len(running_ports)} Ollama instances on ports: {running_ports}")
        
        # Assign domains to agents to prevent overlap
        domain_assignments = self.duplicate_tracker.assign_domains_to_agents(len(running_ports))
        logging.info(f"Domain assignments: {domain_assignments}")
        
        # Initialize agents
        for i, port in enumerate(running_ports):
            agent = AgricultureDataAgent(
                i, port, self.model, self.search_engine, 
                self.duplicate_tracker, self.search_expansion
            )
            self.agents.append(agent)
        
        # Get search queries
        search_queries = AgricultureSearchQueries.get_search_queries(num_queries)
        logging.info(f"Processing {len(search_queries)} base search queries")
        
        # Distribute queries among agents
        queries_per_agent = len(search_queries) // len(self.agents)
        agent_queries = []
        
        for i in range(len(self.agents)):
            start_idx = i * queries_per_agent
            if i == len(self.agents) - 1:  # Last agent gets remaining queries
                end_idx = len(search_queries)
            else:
                end_idx = (i + 1) * queries_per_agent
            
            agent_queries.append(search_queries[start_idx:end_idx])
        
        # Execute curation in parallel
        logging.info("Starting parallel data curation with intelligent expansion...")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(agent.curate_data, queries, self.enable_intelligent_expansion): agent 
                for agent, queries in zip(self.agents, agent_queries)
            }
            
            results = []
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Agent {agent.agent_id} completed: {result.processed_count} entries")
                except Exception as e:
                    logging.error(f"Agent {agent.agent_id} failed: {e}")
        
        execution_time = time.time() - start_time
        
        # Consolidate and save results
        self._save_results(results)
        
        # Generate summary
        summary = self._generate_summary(results, execution_time)
        logging.info(f"Curation completed in {execution_time:.2f} seconds")
        logging.info(f"Total entries: {summary['total_entries']}")
        logging.info(f"PDF files downloaded: {summary.get('pdf_count', 0)}")
        
        return summary
    
    def _save_results(self, results: List[CurationResult]):
        """Save all curated data to JSONL file"""
        total_entries = 0
        pdf_count = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result.success:
                    for entry in result.data_entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                        total_entries += 1
                        
                        if entry.get('is_pdf'):
                            pdf_count += 1
        
        logging.info(f"Saved {total_entries} entries to {self.output_file}")
        if pdf_count > 0:
            logging.info(f"Downloaded and processed {pdf_count} PDF files")
    
    def _generate_summary(self, results: List[CurationResult], execution_time: float) -> Dict:
        """Generate summary of curation process"""
        total_entries = sum(r.processed_count for r in results if r.success)
        successful_agents = sum(1 for r in results if r.success)
        failed_agents = sum(1 for r in results if not r.success)
        total_search_results = sum(r.search_results_count for r in results if r.success)
        
        # Count PDFs
        pdf_count = 0
        for result in results:
            if result.success:
                for entry in result.data_entries:
                    if entry.get('is_pdf'):
                        pdf_count += 1
        
        return {
            "success": True,
            "execution_time": execution_time,
            "total_entries": total_entries,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "total_search_results": total_search_results,
            "pdf_count": pdf_count,
            "output_file": self.output_file,
            "agents_used": len(self.agents),
            "duplicate_prevention_enabled": True,
            "intelligent_expansion_enabled": self.enable_intelligent_expansion,
            "pdf_download_enabled": self.enable_pdf_download
        }


def main():
    """Main function to run the enhanced agriculture data curator"""
    # Enhanced configuration
    config = {
        "num_agents": 4,
        "model": "deepseek-r1:70b",  # Can also use "gemma3:27b"
        "output_file": "indian_agriculture_data_complete.jsonl",
        "max_search_results": 25,  # Increased for more comprehensive data
        "pdf_storage_dir": "downloaded_pdfs",
        "enable_pdf_download": True,  # Enable PDF downloading
        "enable_intelligent_expansion": True,  # Enable LLM-based query expansion
        "num_queries": None  # Use all queries for complete dataset
    }
    
    # Create and run enhanced curator
    curator = AgricultureDataCurator(
        num_agents=config["num_agents"],
        model=config["model"],
        output_file=config["output_file"],
        max_search_results=config["max_search_results"],
        pdf_storage_dir=config["pdf_storage_dir"],
        enable_pdf_download=config["enable_pdf_download"],
        enable_intelligent_expansion=config["enable_intelligent_expansion"]
    )
    
    try:
        summary = curator.start_curation(num_queries=config["num_queries"])
        
        print("\n" + "="*80)
        print("ENHANCED AGRICULTURE DATA CURATION COMPLETED")
        print("="*80)
        print(f"Total entries curated: {summary.get('total_entries', 0)}")
        print(f"PDF files processed: {summary.get('pdf_count', 0)}")
        print(f"Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"Output file: {summary.get('output_file', 'N/A')}")
        print(f"Successful agents: {summary.get('successful_agents', 0)}")
        print(f"Failed agents: {summary.get('failed_agents', 0)}")
        print(f"Duplicate prevention: {'Enabled' if summary.get('duplicate_prevention_enabled') else 'Disabled'}")
        print(f"Intelligent expansion: {'Enabled' if summary.get('intelligent_expansion_enabled') else 'Disabled'}")
        print(f"PDF download: {'Enabled' if summary.get('pdf_download_enabled') else 'Disabled'}")
        
    except KeyboardInterrupt:
        print("\nCuration interrupted by user")
    except Exception as e:
        print(f"Curation failed: {e}")
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
