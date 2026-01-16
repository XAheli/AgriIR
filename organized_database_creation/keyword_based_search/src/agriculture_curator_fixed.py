#!/usr/bin/env python3
"""
Fixed Agriculture Data Curator - Addresses search engine errors and improves PDF processing
Focuses on comprehensive Indian agriculture data including soil, climate, plant species, etc.
"""

import asyncio
import json
import time
import threading
import logging
import hashlib
import os
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
    print("Warning: ddgs not installed. Install with: pip install ddgs")
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

# Import shared utilities
import sys
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
    pdfs_processed: int = 0
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
    soil_types: List[str] = field(default_factory=list)  # New field
    climate_info: List[str] = field(default_factory=list)  # New field
    fertilizers: List[str] = field(default_factory=list)  # New field
    watering_schedule: Optional[str] = None  # New field
    plant_species: List[str] = field(default_factory=list)  # New field
    data_type: str = ""  # statistical, qualitative, mixed
    publication_year: Optional[int] = None
    source_domain: str = ""
    extraction_timestamp: str = ""
    relevance_score: float = 0.0
    content_length: int = 0
    content_hash: str = ""
    url_hash: str = ""
    pdf_path: Optional[str] = None
    is_pdf: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSONL export"""
        return asdict(self)


class ImprovedPDFProcessor:
    """Enhanced PDF processor with better text extraction and immediate JSONL writing"""
    
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
        
    def download_and_process_pdf(self, url: str, title: str = "", search_query: str = "") -> Optional[Dict]:
        """Download PDF and extract structured agriculture information with immediate JSONL save"""
        if requests is None:
            return None
        try:
            # Improved PDF detection: HEAD request, file extension, and stricter heuristics
            if not self._is_pdf_url_strict(url):
                logging.info(f"URL does not appear to be a PDF: {url}")
                return None

            # robots.txt and license check
            if not self._check_robots_and_license(url):
                logging.warning(f"Blocked by robots.txt or missing license: {url}")
                return None

            # HEAD request for content-type and length
            try:
                head_resp = requests.head(url, allow_redirects=True, timeout=10)
                content_type = head_resp.headers.get('Content-Type', '').lower()
                content_length = int(head_resp.headers.get('Content-Length', '0'))
                
                # RELAXED: If URL ends with .pdf, trust it even if content-type is wrong
                if url.lower().endswith('.pdf'):
                    logging.debug(f"Trusting .pdf extension despite content-type: {content_type}")
                elif 'pdf' not in content_type:
                    logging.info(f"HEAD Content-Type not PDF: {content_type} for {url}")
                    return None
                    
                if content_length > self.max_size_bytes and content_length > 0:
                    logging.warning(f"PDF too large ({content_length} bytes): {url}")
                    return None
            except Exception as e:
                logging.debug(f"HEAD request failed for {url}: {e}, trying direct download")
                # Continue anyway if HEAD fails

            # Generate safe filename
            safe_title = re.sub(r'[^\\w\-_\.]', '_', title[:100])
            filename = f"{safe_title}_{int(time.time())}.pdf"
            filepath = self.storage_dir / filename

            logging.info(f"📥 Attempting to download PDF: {url}")

            # Download PDF with improved error handling
            if not self._download_file(url, filepath):
                logging.warning(f"⚠️ PDF download failed, but continuing with other content: {url}")
                return None

            logging.info(f"🔄 Processing PDF immediately: {filepath.name}")

            # Extract text with multiple methods
            text_content = self._extract_text_comprehensive(filepath)

            if not text_content:
                logging.warning(f"⚠️ Failed to extract text from PDF: {filepath}")
                text_content = f"PDF file downloaded but text extraction failed. File: {filepath.name}"

            # Extract structured agriculture information
            structured_info = self._extract_agriculture_info(text_content + " " + title)

            # Extract metadata
            metadata = self._extract_pdf_metadata(filepath)

            # Create immediate JSONL entry
            jsonl_entry = {
                'title': title or metadata.get('title', filepath.stem),
                'author': metadata.get('author'),
                'link': url,
                'text_extracted': text_content,
                'abstract': self._create_abstract(text_content),
                'genre': 'pdf',
                'tags': self._generate_pdf_tags(text_content, title),
                'indian_regions': self._extract_regions(text_content + " " + title),
                'crop_types': self._extract_crops(text_content + " " + title),
                'farming_methods': structured_info.get('farming_methods', []),
                'soil_types': structured_info.get('soil_types', []),
                'climate_info': structured_info.get('climate_info', []),
                'fertilizers': structured_info.get('fertilizers', []),
                'watering_schedule': structured_info.get('watering_schedule'),
                'plant_species': structured_info.get('plant_species', []),
                'data_type': 'mixed',
                'publication_year': self._extract_year(metadata.get('creation_date', '')),
                'source_domain': self._extract_domain(url),
                'extraction_timestamp': datetime.now().isoformat(),
                'relevance_score': self._calculate_relevance(text_content, title, search_query),
                'content_length': len(text_content),
                'content_hash': hashlib.md5(text_content.encode('utf-8')).hexdigest(),
                'url_hash': hashlib.md5(url.encode('utf-8')).hexdigest(),
                'is_pdf': True,
                'pdf_path': str(filepath)
            }

            # IMMEDIATELY SAVE TO JSONL
            if self.jsonl_writer:
                success = self.jsonl_writer.write_entry(jsonl_entry)
                if success:
                    logging.info(f"✅ PDF IMMEDIATELY SAVED to JSONL: {title[:100]}")
                else:
                    logging.error(f"❌ Failed to save PDF to JSONL: {title[:100]}")

            result = {
                'filepath': str(filepath),
                'text_content': text_content,
                'structured_info': structured_info,
                'metadata': metadata,
                'file_size': filepath.stat().st_size,
                'extraction_method': 'comprehensive',
                'jsonl_entry': jsonl_entry,
                'saved_to_jsonl': success if self.jsonl_writer else False
            }

            logging.info(f"✅ Successfully processed PDF: {filepath.name} ({len(text_content)} chars)")
            return result

        except Exception as e:
            logging.warning(f"⚠️ PDF processing failed for {url}: {e} - Continuing with other content")
            return None

    def _is_pdf_url_strict(self, url: str) -> bool:
        """Stricter PDF detection: file extension, content-type, and URL pattern"""
        parsed = urlparse(url)
        if parsed.path.lower().endswith('.pdf'):
            return True
        # Heuristic: look for 'pdf' in query or fragment
        if 'pdf' in parsed.query.lower() or 'pdf' in parsed.fragment.lower():
            return True
        # Avoid naive 'pdf' in URL
        return False

    def _check_robots_and_license(self, url: str) -> bool:
        """Check robots.txt and look for license info before scraping/downloading"""
        try:
            parsed = urlparse(url)
            
            # RELAXED: Allow government and educational domains automatically
            trusted_domains = ['.gov.in', '.edu', '.ac.in', '.nic.in', 'tnau.ac.in', 
                             'icar.org.in', 'agricoop.nic.in', 'krishi.icar.gov.in',
                             'ciks.org', 'icrisat.org', 'cgiar.org']
            
            if any(domain in parsed.netloc.lower() for domain in trusted_domains):
                logging.debug(f"Trusted domain allowed: {parsed.netloc}")
                return True
            
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            try:
                resp = requests.get(robots_url, timeout=3)
                if resp.status_code == 200:
                    # Only block if explicitly disallows PDFs or the specific path
                    if 'Disallow: /pdf' in resp.text or f'Disallow: {parsed.path}' in resp.text:
                        logging.warning(f"robots.txt blocks PDF path for {parsed.netloc}")
                        return False
            except:
                pass  # If robots.txt not accessible, allow
                
            return True  # Default to allowing
        except Exception as e:
            logging.debug(f"robots.txt/license check bypassed for {url}: {e}")
            return True  # Default to allowing on error
    
    def _create_abstract(self, text: str, max_length: int = 500) -> Optional[str]:
        """Create abstract from text content"""
        if not text:
            return None
            
        # Clean text
        text = re.sub(r'\s+', ' ', text.strip())
        
        if len(text) <= max_length:
            return text
            
        # Find a good break point
        abstract = text[:max_length]
        last_period = abstract.rfind('.')
        if last_period > max_length * 0.7:  # If period is reasonably close to end
            abstract = abstract[:last_period + 1]
        else:
            abstract = abstract.rsplit(' ', 1)[0] + "..."
            
        return abstract
    
    def _generate_pdf_tags(self, content: str, title: str) -> List[str]:
        """Generate tags specifically for PDF content"""
        text = (content + " " + title).lower()
        tags = ['pdf', 'research', 'document']
        
        # PDF-specific tag mapping
        tag_mapping = {
            'soil': ['soil', 'fertility', 'nutrients', 'organic matter'],
            'irrigation': ['irrigation', 'water', 'drip', 'sprinkler'],
            'climate': ['climate', 'weather', 'monsoon', 'drought'],
            'technology': ['technology', 'precision', 'digital', 'smart'],
            'policy': ['policy', 'government', 'scheme', 'subsidy'],
            'organic': ['organic', 'sustainable', 'bio', 'natural'],
            'pest': ['pest', 'disease', 'protection', 'ipm'],
            'breeding': ['breeding', 'varieties', 'genetics', 'selection'],
            'economics': ['economics', 'cost', 'profit', 'income', 'market']
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        
        return list(set(tags))
    
    def _extract_year(self, date_string: str) -> Optional[int]:
        """Extract year from date string"""
        if not date_string:
            return None
            
        # Look for 4-digit year
        year_match = re.search(r'(19|20)\d{2}', str(date_string))
        if year_match:
            return int(year_match.group())
        return None
    
    def _calculate_relevance(self, content: str, title: str, query: str) -> float:
        """Calculate relevance score"""
        text = (content + " " + title + " " + query).lower()
        score = 0.0
        
        # High-value keywords
        agriculture_terms = ['agriculture', 'farming', 'crop', 'cultivation', 'harvest', 'soil', 'irrigation']
        india_terms = ['india', 'indian', 'bharatiya']
        
        for term in agriculture_terms:
            score += text.count(term) * 0.1
        
        for term in india_terms:
            score += text.count(term) * 0.05
            
        return min(score, 1.0)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""
    
    def _extract_regions(self, text: str) -> List[str]:
        """Extract Indian regions/states mentioned in text"""
        indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand',
            'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
            'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
            'uttar pradesh', 'uttarakhand', 'west bengal'
        ]
        
        text_lower = text.lower()
        regions = []
        
        for state in indian_states:
            if state in text_lower:
                regions.append(state.title())
        
        return list(set(regions))
    
    def _extract_crops(self, text: str) -> List[str]:
        """Extract crop types mentioned in text"""
        crop_patterns = [
            'rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'bajra', 'jowar',
            'ragi', 'pulses', 'gram', 'tur', 'lentil', 'groundnut', 'soybean',
            'mustard', 'sunflower', 'tea', 'coffee', 'rubber', 'coconut',
            'cardamom', 'pepper', 'turmeric', 'ginger', 'mango', 'banana',
            'potato', 'onion', 'tomato', 'chili', 'garlic', 'coriander'
        ]
        
        text_lower = text.lower()
        crops = []
        
        for crop in crop_patterns:
            if crop in text_lower:
                crops.append(crop)
        
        return list(set(crops))
    
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
        """Download file with improved error handling and retries"""
        max_retries = 2
        
        for attempt in range(max_retries):
            try:
                # Use different headers to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/pdf,application/octet-stream,*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1'
                }
                
                response = requests.get(url, stream=True, timeout=30, headers=headers, allow_redirects=True)
                
                # Handle different status codes more gracefully
                if response.status_code == 404:
                    logging.warning(f"⚠️ PDF not found (404): {url}")
                    return False
                elif response.status_code == 403:
                    logging.warning(f"⚠️ PDF access forbidden (403): {url}")
                    return False
                elif response.status_code == 503:
                    logging.warning(f"⚠️ PDF service unavailable (503): {url}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                        continue
                    return False
                
                response.raise_for_status()
                
                # Check if response is actually a PDF
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
                    logging.warning(f"⚠️ URL does not appear to be a PDF: {url} (content-type: {content_type})")
                    return False
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_size_bytes:
                    logging.warning(f"⚠️ PDF too large: {content_length} bytes > {self.max_size_bytes}")
                    return False
                    
                downloaded_size = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            downloaded_size += len(chunk)
                            if downloaded_size > self.max_size_bytes:
                                logging.warning(f"⚠️ PDF download exceeded size limit: {filepath}")
                                filepath.unlink()
                                return False
                            f.write(chunk)
                            
                logging.info(f"✅ PDF downloaded successfully: {filepath.name} ({downloaded_size} bytes)")
                return True
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"⚠️ PDF download attempt {attempt + 1}/{max_retries} failed: {e}")
                if filepath.exists():
                    filepath.unlink()
                
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                    
            except Exception as e:
                logging.error(f"❌ Unexpected error downloading PDF: {e}")
                if filepath.exists():
                    filepath.unlink()
                return False
                
        logging.warning(f"❌ Failed to download PDF after {max_retries} attempts: {url}")
        return False
    
    def _extract_text_comprehensive(self, filepath: Path) -> str:
        """Extract text using multiple methods with fallbacks"""
        text_content = ""
        
        # Method 1: Try PyMuPDF first (best for most PDFs)
        if fitz:
            try:
                doc = fitz.open(str(filepath))
                for page in doc:
                    text_content += page.get_text()
                doc.close()
                
                if text_content.strip():
                    logging.debug(f"PyMuPDF extraction successful: {len(text_content)} chars")
                    return text_content
            except Exception as e:
                logging.debug(f"PyMuPDF extraction failed: {e}")
        
        # Method 2: Try PyPDF2 as fallback
        if PyPDF2:
            try:
                with open(filepath, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page in reader.pages:
                        text_content += page.extract_text()
                        
                if text_content.strip():
                    logging.debug(f"PyPDF2 extraction successful: {len(text_content)} chars")
                    return text_content
            except Exception as e:
                logging.debug(f"PyPDF2 extraction failed: {e}")
        
        # Method 3: OCR as last resort for scanned PDFs
        if pytesseract and Image and fitz:
            try:
                logging.info(f"Attempting OCR for {filepath.name}")
                text_content = self._ocr_pdf(filepath)
                if text_content.strip():
                    logging.info(f"OCR extraction successful: {len(text_content)} chars")
                    return text_content
            except Exception as e:
                logging.error(f"OCR extraction failed: {e}")
                
        return text_content
    
    def _ocr_pdf(self, filepath: Path) -> str:
        """Extract text using OCR for scanned PDFs"""
        try:
            doc = fitz.open(str(filepath))
            text_content = ""
            
            # Limit OCR to first 5 pages for performance
            max_pages = min(doc.page_count, 5)
            
            for page_num in range(max_pages):
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
    
    def _extract_agriculture_info(self, text: str) -> Dict:
        """Extract structured agriculture information from text"""
        text_lower = text.lower()
        
        # Extract soil information
        soil_types = []
        for keyword in self.soil_keywords:
            if keyword in text_lower:
                soil_types.append(keyword)
        
        # Extract climate information
        climate_info = []
        for keyword in self.climate_keywords:
            if keyword in text_lower:
                climate_info.append(keyword)
        
        # Extract fertilizer information
        fertilizers = []
        for keyword in self.fertilizer_keywords:
            if keyword in text_lower:
                fertilizers.append(keyword)
        
        # Extract plant species information
        plant_species = []
        for keyword in self.plant_species_keywords:
            if keyword in text_lower:
                plant_species.append(keyword)
        
        # Extract watering schedule information
        watering_schedule = None
        watering_patterns = [
            r'water(?:ing)?\s+(?:schedule|frequency|interval)',
            r'irrigation\s+(?:schedule|timing|frequency)',
            r'(?:daily|weekly|monthly)\s+watering',
            r'water\s+(?:every|once)\s+\d+\s+(?:days|weeks|months)'
        ]
        
        for pattern in watering_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # Extract surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 100)
                watering_schedule = text[start:end].strip()
                break
        
        return {
            'soil_types': list(set(soil_types)),
            'climate_info': list(set(climate_info)),
            'fertilizers': list(set(fertilizers)),
            'plant_species': list(set(plant_species)),
            'watering_schedule': watering_schedule
        }
    
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


class ExpandedAgricultureQueries:
    """Comprehensive search queries covering all aspects of Indian agriculture"""
    
    BASE_QUERIES = [
        # Soil and Climate specific queries
        "Indian soil types classification agriculture research papers",
        "soil fertility management India organic matter studies",
        "soil pH management Indian agriculture practices",
        "climate change impact Indian agriculture adaptation",
        "monsoon agriculture India rainfall patterns research",
        "drought management techniques Indian farming",
        "irrigation systems India water management studies",
        "soil erosion control India conservation practices",
        
        # Plant species and varieties
        "Indian crop varieties indigenous seeds research",
        "high yielding varieties India crop breeding",
        "drought resistant crops India genetic studies",
        "pest resistant varieties Indian agriculture",
        "traditional plant varieties India conservation",
        "hybrid seeds India crop improvement programs",
        "wild relatives crops India genetic resources",
        
        # Fertilizers and nutrition
        "fertilizer management Indian agriculture NPK studies",
        "organic fertilizer India compost vermicompost",
        "bio-fertilizer India microbial agriculture",
        "micronutrient deficiency Indian soils research",
        "integrated nutrient management India studies",
        "green manure crops India soil fertility",
        "precision nutrient management Indian farming",
        
        # Watering and irrigation
        "irrigation scheduling Indian crops water management",
        "drip irrigation India water conservation studies",
        "sprinkler irrigation systems Indian agriculture",
        "water use efficiency crops India research",
        "deficit irrigation strategies Indian farming",
        "rainwater harvesting agriculture India",
        "groundwater irrigation India sustainability",
        
        # Crop-specific detailed queries
        "rice cultivation techniques India varieties studies",
        "wheat production India irrigation fertilizer",
        "cotton farming practices India pest management",
        "sugarcane cultivation India soil water requirements",
        "pulse crops India nitrogen fixation studies",
        "millet cultivation India drought tolerance",
        "spice crops India cultivation practices research",
        "fruit crops India horticulture management",
        "vegetable production India protected cultivation",
        
        # Regional specific agriculture
        "Punjab agriculture soil health studies",
        "Maharashtra farming cotton cultivation research",
        "Tamil Nadu agriculture rice ecosystem",
        "Karnataka horticulture crop management",
        "Uttar Pradesh agriculture crop diversification",
        "West Bengal agriculture rice intensification",
        "Gujarat agriculture groundwater irrigation",
        "Rajasthan agriculture arid farming techniques",
        "Kerala agriculture spice plantation management",
        "Andhra Pradesh agriculture irrigation studies",
        
        # Advanced agricultural practices
        "precision agriculture India GPS technology",
        "smart farming IoT sensors Indian agriculture",
        "sustainable agriculture practices India research",
        "integrated farming systems India studies",
        "conservation agriculture India tillage practices",
        "agroforestry systems India tree crop integration",
        "protected cultivation India greenhouse farming",
        "hydroponics agriculture India soilless cultivation",
        
        # Plant protection and pest management
        "integrated pest management India crop protection",
        "biological control agents Indian agriculture",
        "pesticide resistance management India studies",
        "disease forecasting systems Indian crops",
        "nematode management Indian agriculture",
        "weed management strategies India herbicides",
        "pollinator conservation Indian agriculture",
        
        # Post-harvest and processing
        "post-harvest management India crop storage",
        "food processing agriculture India value addition",
        "crop drying techniques India solar dryers",
        "storage pest management India grains",
        "cold storage facilities India horticulture",
        
        # Economic and policy aspects
        "agricultural economics India cost cultivation",
        "crop insurance schemes India risk management",
        "agricultural marketing India price discovery",
        "farm mechanization India labor productivity",
        "agricultural credit India institutional finance",
        "minimum support price India crop profitability",
        "agricultural subsidies India input costs",
        
        # Research and development
        "agricultural research institutions India ICAR",
        "crop breeding programs India varieties",
        "agricultural extension services India farmers",
        "farmer training programs India capacity building",
        "agricultural education India universities",
        "international cooperation agriculture India"
    ]
    
    @classmethod
    def get_search_queries(cls, num_queries: Optional[int] = None) -> List[str]:
        """Get comprehensive search queries"""
        if num_queries:
            return cls.BASE_QUERIES[:num_queries]
        return cls.BASE_QUERIES


class ImprovedWebSearch:
    """Improved web search with better error handling and immediate JSONL writing"""
    
    def __init__(self, max_results: int = 20, pdf_processor: Optional[ImprovedPDFProcessor] = None,
                 jsonl_writer: Optional[ImmediateJSONLWriter] = None):
        self.max_results = max_results
        self.pdf_processor = pdf_processor
        self.jsonl_writer = jsonl_writer
        
        if requests is None:
            raise ImportError("requests library is required")
            
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Extended patterns for agriculture information
        self.indian_states = [
            'andhra pradesh', 'arunachal pradesh', 'assam', 'bihar', 'chhattisgarh',
            'goa', 'gujarat', 'haryana', 'himachal pradesh', 'jharkhand',
            'karnataka', 'kerala', 'madhya pradesh', 'maharashtra', 'manipur',
            'meghalaya', 'mizoram', 'nagaland', 'odisha', 'punjab',
            'rajasthan', 'sikkim', 'tamil nadu', 'telangana', 'tripura',
            'uttar pradesh', 'uttarakhand', 'west bengal'
        ]
        
        self.crop_patterns = [
            'rice', 'wheat', 'cotton', 'sugarcane', 'maize', 'bajra', 'jowar',
            'ragi', 'pulses', 'gram', 'tur', 'lentil', 'groundnut', 'soybean',
            'mustard', 'sunflower', 'tea', 'coffee', 'rubber', 'coconut',
            'cardamom', 'pepper', 'turmeric', 'ginger', 'mango', 'banana',
            'potato', 'onion', 'tomato', 'chili', 'garlic', 'coriander'
        ]
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.close()
        return False
    
    def close(self):
        """Close and cleanup resources"""
        if self.session:
            self.session.close()
            logging.debug("Web search session closed")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        try:
            self.close()
        except:
            pass
    
    def search_and_extract(self, query: str, agent_id: int = 0) -> List[Dict]:
        """Search with improved error handling, fallback mechanism, and immediate JSONL save tracking"""
        results = []
        
        if DDGS is None:
            logging.error("DDGS not available")
            return results
        
        # Try multiple search strategies with fallback
        search_results = []
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Use simpler DDGS search to avoid engine errors
                with DDGS() as ddgs:
                    try:
                        # Use text search with correct ddgs syntax
                        for result in ddgs.text(
                            query,
                            region='wt-wt',  # Global region for better results
                            max_results=self.max_results
                        ):
                            search_results.append(result)
                            if len(search_results) >= self.max_results:
                                break
                                
                    except Exception as search_error:
                        logging.warning(f"DDGS search error for '{query}' (attempt {attempt + 1}/{max_attempts}): {search_error}")
                        
                        # If this isn't the last attempt, wait before retrying
                        if attempt < max_attempts - 1:
                            time.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s
                            continue
                        else:
                            # Last attempt failed - try alternative search strategy
                            logging.warning(f"All search attempts failed for: {query}")
                            break
                    
                # If we got results, break out of retry loop
                if search_results:
                    break
                    
            except Exception as e:
                logging.error(f"Search engine error for '{query}' (attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                    
        logging.info(f"Found {len(search_results)} search results for: {query}")
                
        for result in search_results:
            try:
                extracted_data = self._extract_content(result)
                if extracted_data:
                    results.append(extracted_data)
                    
                    # Log immediate save status
                    if extracted_data.get('saved_to_jsonl'):
                        logging.info(f"✅ Content IMMEDIATELY SAVED by Agent {agent_id}: {extracted_data.get('title', 'Unknown')[:80]}...")
                    else:
                        logging.warning(f"⚠️ Content not saved to JSONL by Agent {agent_id}: {extracted_data.get('title', 'Unknown')[:80]}...")
                    
            except Exception as e:
                logging.warning(f"Failed to extract content: {e}")
                continue
            
        return results
    
    def _extract_content(self, search_result: Dict) -> Optional[Dict]:
        """Extract and structure content from search result with immediate JSONL save"""
        try:
            url = search_result.get('href', '')
            title = search_result.get('title', '')
            snippet = search_result.get('body', '')
            
            if not url or not title:
                return None
            
            # Determine content type
            genre = self._classify_content_type(url, title, snippet)
            
            # Check if this is a PDF
            is_pdf = url.lower().endswith('.pdf') or 'pdf' in url.lower()
            pdf_data = None
            
            # Process PDF if available and processor is enabled
            if is_pdf and self.pdf_processor:
                logging.info(f"📄 Processing PDF: {title[:100]}")
                pdf_data = self.pdf_processor.download_and_process_pdf(url, title, "")
                
                # If PDF was successfully processed and saved, return it
                if pdf_data:
                    if pdf_data.get('saved_to_jsonl'):
                        logging.info(f"✅ PDF already saved to JSONL: {title[:100]}")
                        return pdf_data  # PDF was already saved by processor
                    else:
                        # PDF was processed but not saved - continue to process
                        logging.info(f"ℹ️ PDF processed but not saved, will process as web content: {title[:100]}")
            
            # Process as web content (either non-PDF, PDF processing disabled, or PDF processing failed)
            # Only process if we don't already have successfully saved PDF data
            if not (pdf_data and pdf_data.get('saved_to_jsonl')):
                # Extract full content
                full_content = self._fetch_full_content(url)
                if not full_content:
                    full_content = snippet
                
                # Extract structured information
                structured_info = self._extract_agriculture_info_from_text(full_content + " " + title + " " + snippet)
                
                # Extract metadata
                regions = self._extract_regions(full_content + " " + title + " " + snippet)
                crops = self._extract_crops(full_content + " " + title + " " + snippet)
                tags = self._generate_tags(full_content, title, snippet, genre)
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance(full_content, title, snippet)
                
                # Create JSONL entry
                jsonl_entry = {
                    'title': title,
                    'author': None,
                    'link': url,
                    'text_extracted': full_content,
                    'abstract': snippet[:500] if snippet else None,
                    'genre': genre,
                    'tags': tags,
                    'indian_regions': regions,
                    'crop_types': crops,
                    'farming_methods': structured_info.get('farming_methods', []),
                    'soil_types': structured_info.get('soil_types', []),
                    'climate_info': structured_info.get('climate_info', []),
                    'fertilizers': structured_info.get('fertilizers', []),
                    'watering_schedule': structured_info.get('watering_schedule'),
                    'plant_species': structured_info.get('plant_species', []),
                    'data_type': 'mixed',
                    'publication_year': None,
                    'source_domain': self._extract_domain(url),
                    'extraction_timestamp': datetime.now().isoformat(),
                    'relevance_score': relevance_score,
                    'content_length': len(full_content),
                    'content_hash': hashlib.md5(full_content.encode('utf-8')).hexdigest(),
                    'url_hash': hashlib.md5(url.encode('utf-8')).hexdigest(),
                    'is_pdf': is_pdf,
                    'pdf_path': None
                }
                
                # IMMEDIATELY SAVE TO JSONL
                saved_to_jsonl = False
                if self.jsonl_writer:
                    saved_to_jsonl = self.jsonl_writer.write_entry(jsonl_entry)
                    if saved_to_jsonl:
                        logging.info(f"✅ Web content IMMEDIATELY SAVED to JSONL: {title[:100]}")
                    else:
                        logging.error(f"❌ Failed to save web content to JSONL: {title[:100]}")
            
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
                    'pdf_path': None,
                    'content_hash': hashlib.md5(full_content.encode('utf-8')).hexdigest(),
                    'url_hash': hashlib.md5(url.encode('utf-8')).hexdigest(),
                    'jsonl_entry': jsonl_entry,
                    'saved_to_jsonl': saved_to_jsonl,
                    **structured_info  # Add soil, climate, fertilizer info
                }
                
                return result
            
            return pdf_data
            
        except Exception as e:
            logging.error(f"❌ Content extraction and save failed: {e}")
            return None
    
    def _extract_agriculture_info_from_text(self, text: str) -> Dict:
        """Extract agriculture-specific information from text"""
        text_lower = text.lower()
        
        # Soil types
        soil_keywords = ['black soil', 'red soil', 'alluvial soil', 'laterite soil', 'sandy soil', 'clay soil', 'loamy soil']
        soil_types = [keyword for keyword in soil_keywords if keyword in text_lower]
        
        # Climate info
        climate_keywords = ['tropical', 'temperate', 'arid', 'semi-arid', 'monsoon', 'rainfall', 'drought']
        climate_info = [keyword for keyword in climate_keywords if keyword in text_lower]
        
        # Fertilizers
        fertilizer_keywords = ['urea', 'dap', 'potash', 'nitrogen', 'phosphorus', 'potassium', 'compost', 'manure']
        fertilizers = [keyword for keyword in fertilizer_keywords if keyword in text_lower]
        
        # Plant species
        species_keywords = ['varieties', 'cultivars', 'hybrid', 'indigenous', 'high yielding', 'drought resistant']
        plant_species = [keyword for keyword in species_keywords if keyword in text_lower]
        
        # Farming methods
        method_keywords = ['organic', 'precision', 'sustainable', 'integrated', 'conservation', 'drip irrigation']
        farming_methods = [keyword for keyword in method_keywords if keyword in text_lower]
        
        return {
            'soil_types': list(set(soil_types)),
            'climate_info': list(set(climate_info)),
            'fertilizers': list(set(fertilizers)),
            'plant_species': list(set(plant_species)),
            'farming_methods': list(set(farming_methods)),
            'watering_schedule': None
        }
    
    def _classify_content_type(self, url: str, title: str, snippet: str) -> str:
        """Classify the type of content"""
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
        """Fetch full content from URL"""
        if BeautifulSoup is None:
            return ""
            
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text if text else ""
            
        except Exception as e:
            logging.debug(f"Failed to fetch content from {url}: {e}")
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
        tag_mapping = {
            'soil': ['soil', 'fertility', 'nutrients', 'organic matter'],
            'irrigation': ['irrigation', 'water', 'drip', 'sprinkler'],
            'climate': ['climate', 'weather', 'monsoon', 'drought'],
            'technology': ['technology', 'precision', 'digital', 'smart'],
            'policy': ['policy', 'government', 'scheme', 'subsidy'],
            'organic': ['organic', 'sustainable', 'bio', 'natural'],
            'pest': ['pest', 'disease', 'protection', 'ipm']
        }
        
        for tag, keywords in tag_mapping.items():
            if any(keyword in text for keyword in keywords):
                tags.append(tag)
        
        return list(set(tags))
    
    def _calculate_relevance(self, content: str, title: str, snippet: str) -> float:
        """Calculate relevance score"""
        text = (content + " " + title + " " + snippet).lower()
        score = 0.0
        
        # High-value keywords
        agriculture_terms = ['agriculture', 'farming', 'crop', 'cultivation', 'harvest', 'soil', 'irrigation']
        india_terms = ['india', 'indian', 'bharatiya']
        
        for term in agriculture_terms:
            score += text.count(term) * 0.1
        
        for term in india_terms:
            score += text.count(term) * 0.05
            
        return min(score, 1.0)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc
        except:
            return ""


class SimpleAgricultureAgent:
    """Simplified agent with improved error handling"""
    
    def __init__(self, agent_id: int, search_engine: ImprovedWebSearch):
        self.agent_id = agent_id
        self.search_engine = search_engine
        
        # Agent specialization
        self.specializations = {
            0: "soil and climate management",
            1: "plant varieties and breeding", 
            2: "fertilizers and nutrition",
            3: "irrigation and water management"
        }
        
        self.specialization = self.specializations.get(agent_id % 4, "general agriculture")
    
    def curate_data(self, search_queries: List[str]) -> CurationResult:
        """Curate agriculture data with comprehensive information extraction"""
        start_time = time.time()
        data_entries = []
        total_search_results = 0
        pdfs_processed = 0
        
        try:
            logging.info(f"Agent {self.agent_id} ({self.specialization}): Processing {len(search_queries)} queries")
            
            for i, query in enumerate(search_queries):
                # Modify query based on agent specialization
                specialized_query = self._specialize_query(query)
                
                logging.info(f"Agent {self.agent_id}: Query {i+1}/{len(search_queries)}: {specialized_query}")
                
                # Search and extract data
                search_results = self.search_engine.search_and_extract(specialized_query, self.agent_id)
                total_search_results += len(search_results)
                
                # Process each search result
                for result in search_results:
                    processed_entry = self._process_search_result(result, specialized_query)
                    if processed_entry:
                        data_entries.append(processed_entry)
                        if processed_entry.get('is_pdf'):
                            pdfs_processed += 1
                
                # Delay between searches
                time.sleep(2)
            
            execution_time = time.time() - start_time
            
            logging.info(f"Agent {self.agent_id} completed: {len(data_entries)} entries, {pdfs_processed} PDFs")
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Processed {len(search_queries)} queries ({self.specialization})",
                port=0,  # Not using port-based agents
                success=True,
                execution_time=execution_time,
                data_entries=data_entries,
                search_results_count=total_search_results,
                processed_count=len(data_entries),
                pdfs_processed=pdfs_processed,
                status=AgentStatus.COMPLETED
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logging.error(f"Agent {self.agent_id} failed: {e}")
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Failed processing {len(search_queries)} queries",
                port=0,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                status=AgentStatus.FAILED
            )
    
    def _specialize_query(self, query: str) -> str:
        """Add specialization context to search query"""
        specialization_terms = {
            "soil and climate management": "soil climate weather irrigation",
            "plant varieties and breeding": "varieties cultivars seeds breeding genetics",
            "fertilizers and nutrition": "fertilizer nutrients NPK organic compost",
            "irrigation and water management": "irrigation water drip sprinkler scheduling"
        }
        
        terms = specialization_terms.get(self.specialization, "")
        return f"{query} {terms}"
    
    def _process_search_result(self, result: Dict, query: str) -> Dict:
        """Process search result into structured data entry"""
        try:
            return {
                'title': result.get('title', ''),
                'author': result.get('pdf_metadata', {}).get('author') if result.get('is_pdf') else None,
                'link': result.get('url', ''),
                'text_extracted': result.get('full_content', result.get('snippet', '')),
                'abstract': result.get('snippet', '')[:500] if result.get('snippet') else None,
                'genre': result.get('genre', 'article'),
                'tags': result.get('tags', []),
                'indian_regions': result.get('indian_regions', []),
                'crop_types': result.get('crop_types', []),
                'farming_methods': result.get('farming_methods', []),
                'soil_types': result.get('soil_types', []),
                'climate_info': result.get('climate_info', []),
                'fertilizers': result.get('fertilizers', []),
                'watering_schedule': result.get('watering_schedule'),
                'plant_species': result.get('plant_species', []),
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
            
        except Exception as e:
            logging.error(f"Failed to process search result: {e}")
            return None


class FixedAgricultureCurator:
    """Main curator with improved error handling and comprehensive data extraction"""
    
    def __init__(self, 
                 num_agents: int = 4, 
                 output_file: str = "indian_agriculture_complete.jsonl",
                 max_search_results: int = 20,
                 pdf_storage_dir: str = "downloaded_pdfs",
                 enable_pdf_download: bool = True):
        
        self.num_agents = num_agents
        self.output_file = output_file
        self.max_search_results = max_search_results
        self.enable_pdf_download = enable_pdf_download
        
        # Initialize IMMEDIATE JSONL writer with duplicate tracking
        persistence_file = str(Path(output_file).parent / "processed_urls.json")
        self.jsonl_writer = ImmediateJSONLWriter(
            output_file, 
            use_duplicate_tracker=True,
            duplicate_tracker_file=persistence_file,
            clear_file=True
        )
        
        # Initialize PDF processor if enabled
        self.pdf_processor = ImprovedPDFProcessor(
            storage_dir=pdf_storage_dir, 
            jsonl_writer=self.jsonl_writer
        ) if enable_pdf_download else None
        
        # Initialize search engine with JSONL writer
        self.search_engine = ImprovedWebSearch(max_search_results, self.pdf_processor, self.jsonl_writer)
        
        self.agents = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('agriculture_curator_fixed.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create directories
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        if enable_pdf_download:
            Path(pdf_storage_dir).mkdir(parents=True, exist_ok=True)
        
        logging.info(f"✅ IMMEDIATE JSONL WRITER initialized for: {output_file}")
        logging.info(f"✅ PDF download enabled: {enable_pdf_download}")
        logging.info(f"✅ All content will be IMMEDIATELY SAVED upon processing")
    
    def _distribute_queries_by_specialization(self, queries: List[str]) -> List[List[str]]:
        """
        Distribute queries to agents based on specialization relevance.
        Uses keyword matching to route queries to most relevant specialized agent.
        """
        # Define keywords for each specialization (same as SimpleAgricultureAgent)
        specialization_map = {
            0: "soil and climate management",
            1: "plant varieties and breeding", 
            2: "fertilizers and nutrition",
            3: "irrigation and water management"
        }
        
        specialization_keywords = {
            0: ['soil', 'climate', 'weather', 'temperature', 'rainfall', 'land', 'erosion', 'fertility'],
            1: ['variety', 'varieties', 'seed', 'breeding', 'hybrid', 'cultivar', 'species', 'plant'],
            2: ['fertilizer', 'nutrient', 'nitrogen', 'phosphorus', 'potassium', 'urea', 'compost', 'manure'],
            3: ['irrigation', 'water', 'drip', 'sprinkler', 'canal', 'rainfall', 'drainage', 'moisture']
        }
        
        # Initialize query lists for each agent
        agent_queries = [[] for _ in range(len(self.agents))]
        
        # Score each query for each specialization
        for query in queries:
            query_lower = query.lower()
            scores = []
            
            for agent_id in range(len(self.agents)):
                spec_id = agent_id % 4
                keywords = specialization_keywords.get(spec_id, [])
                
                # Count keyword matches
                score = sum(1 for keyword in keywords if keyword in query_lower)
                scores.append(score)
            
            # Assign to agent with highest score (or round-robin if tie)
            max_score = max(scores)
            if max_score > 0:
                # Find agents with max score
                best_agents = [i for i, s in enumerate(scores) if s == max_score]
                # Assign to least loaded agent among best matches
                best_agent = min(best_agents, key=lambda x: len(agent_queries[x]))
                agent_queries[best_agent].append(query)
            else:
                # No keyword match - use round-robin
                min_loaded = min(range(len(agent_queries)), key=lambda x: len(agent_queries[x]))
                agent_queries[min_loaded].append(query)
        
        # Log distribution
        for i, queries in enumerate(agent_queries):
            agent_spec = specialization_map.get(i % 4, "general agriculture")
            logging.info(f"Agent {i} ({agent_spec}): Assigned {len(queries)} queries")
        
        return agent_queries
    
    def start_curation(self, num_queries: Optional[int] = None) -> Dict:
        """Start the fixed agriculture data curation process"""
        logging.info("Starting Fixed Agriculture Data Curation System")
        
        # Initialize agents (no Ollama needed for basic version)
        for i in range(self.num_agents):
            agent = SimpleAgricultureAgent(i, self.search_engine)
            self.agents.append(agent)
        
        # Get comprehensive search queries
        search_queries = ExpandedAgricultureQueries.get_search_queries(num_queries)
        logging.info(f"Processing {len(search_queries)} comprehensive search queries")
        
        # Distribute queries among agents based on specialization
        agent_queries = self._distribute_queries_by_specialization(search_queries)
        
        # Execute curation in parallel
        logging.info("Starting parallel data curation...")
        start_time = time.time()
        
        results = []
        failed_tasks = []
        
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            future_to_agent = {
                executor.submit(agent.curate_data, queries): (agent, queries)
                for agent, queries in zip(self.agents, agent_queries)
            }
            
            for future in as_completed(future_to_agent):
                agent, queries = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    logging.info(f"Agent {agent.agent_id} completed: {result.processed_count} entries, {result.pdfs_processed} PDFs")
                except Exception as e:
                    logging.error(f"Agent {agent.agent_id} failed: {e}")
                    failed_tasks.append((agent, queries, str(e)))
        
        # Retry failed tasks with exponential backoff
        max_retries = 2
        for retry_attempt in range(max_retries):
            if not failed_tasks:
                break
                
            logging.info(f"Retrying {len(failed_tasks)} failed tasks (attempt {retry_attempt + 1}/{max_retries})...")
            time.sleep(2 ** retry_attempt)  # Exponential backoff: 1s, 2s
            
            retry_tasks = failed_tasks.copy()
            failed_tasks.clear()
            
            with ThreadPoolExecutor(max_workers=min(len(retry_tasks), self.num_agents)) as executor:
                future_to_retry = {
                    executor.submit(agent.curate_data, queries): (agent, queries)
                    for agent, queries, _ in retry_tasks
                }
                
                for future in as_completed(future_to_retry):
                    agent, queries = future_to_retry[future]
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(f"✅ Agent {agent.agent_id} retry succeeded: {result.processed_count} entries")
                    except Exception as e:
                        logging.error(f"❌ Agent {agent.agent_id} retry failed: {e}")
                        failed_tasks.append((agent, queries, str(e)))
        
        # Log final failed tasks
        if failed_tasks:
            logging.warning(f"⚠️ {len(failed_tasks)} tasks failed after all retries")
            for agent, queries, error in failed_tasks:
                logging.warning(f"   Agent {agent.agent_id}: {len(queries)} queries - {error}")
        
        execution_time = time.time() - start_time
        
        # Save results with duplicate tracking (already done immediately)
        # self._save_results(results)
        
        # Generate summary
        summary = self._generate_summary(results, execution_time)
        logging.info(f"Curation completed in {execution_time:.2f} seconds")
        logging.info(f"Total entries: {summary['total_entries']}")
        logging.info(f"Total PDFs processed: {summary['total_pdfs']}")
        
        return summary
    
    def _save_results(self, results: List[CurationResult]):
        """Save all curated data to JSONL file"""
        total_entries = 0
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                if result.success:
                    for entry in result.data_entries:
                        json.dump(entry, f, ensure_ascii=False)
                        f.write('\n')
                        total_entries += 1
        
        logging.info(f"Saved {total_entries} entries to {self.output_file}")
    
    def _generate_summary(self, results: List[CurationResult], execution_time: float) -> Dict:
        """Generate summary of curation process"""
        total_entries = self.jsonl_writer.get_entries_count()
        total_pdfs = sum(r.pdfs_processed for r in results if r.success)
        successful_agents = sum(1 for r in results if r.success)
        failed_agents = sum(1 for r in results if not r.success)
        total_search_results = sum(r.search_results_count for r in results if r.success)
        
        return {
            "success": True,
            "execution_time": execution_time,
            "total_entries": total_entries,
            "total_pdfs": total_pdfs,
            "successful_agents": successful_agents,
            "failed_agents": failed_agents,
            "total_search_results": total_search_results,
            "output_file": self.output_file,
            "agents_used": len(self.agents),
            "pdf_download_enabled": self.enable_pdf_download
        }


def main():
    """Main function to run the fixed agriculture data curator"""
    config = {
        "num_agents": 4,
        "output_file": "indian_agriculture_complete.jsonl",
        "max_search_results": 20,
        "pdf_storage_dir": "downloaded_pdfs",
        "enable_pdf_download": True,
        "num_queries": None  # Use all queries for complete dataset
    }
    
    # Create and run fixed curator
    curator = FixedAgricultureCurator(
        num_agents=config["num_agents"],
        output_file=config["output_file"],
        max_search_results=config["max_search_results"],
        pdf_storage_dir=config["pdf_storage_dir"],
        enable_pdf_download=config["enable_pdf_download"]
    )
    
    try:
        summary = curator.start_curation(num_queries=config["num_queries"])
        
        print("\n" + "="*80)
        print("FIXED AGRICULTURE DATA CURATION COMPLETED")
        print("="*80)
        print(f"Total entries curated: {summary.get('total_entries', 0)}")
        print(f"PDF files processed: {summary.get('total_pdfs', 0)}")
        print(f"Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"Output file: {summary.get('output_file', 'N/A')}")
        print(f"Successful agents: {summary.get('successful_agents', 0)}")
        print(f"Failed agents: {summary.get('failed_agents', 0)}")
        
    except KeyboardInterrupt:
        print("\nCuration interrupted by user")
    except Exception as e:
        print(f"Curation failed: {e}")
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
