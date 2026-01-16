#!/usr/bin/env python3
"""
Improved Content Extraction with Robust Heuristics
Fixes brittle extraction logic and improves relevance detection
"""

import re
import logging
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter


class ImprovedContentExtractor:
    """
    Robust content extractor with improved agriculture relevance detection.
    """
    
    # Expanded agriculture keywords
    AGRICULTURE_KEYWORDS = {
        # Core terms
        'agriculture', 'farming', 'agricultural', 'farm', 'crop', 'crops',
        'soil', 'irrigation', 'fertilizer', 'pesticide', 'herbicide',
        'harvest', 'cultivation', 'livestock', 'cattle', 'poultry',
        
        # Indian agriculture specific
        'kharif', 'rabi', 'zaid', 'monsoon', 'paddy', 'wheat', 'rice',
        'millet', 'pulses', 'oilseeds', 'sugarcane', 'cotton',
        'kisan', 'farmer', 'grameen', 'krishi',
        
        # Technical terms
        'agronomy', 'horticulture', 'sericulture', 'apiculture',
        'aquaculture', 'pisciculture', 'floriculture',
        'organic farming', 'sustainable agriculture', 'precision agriculture',
        
        # Related terms
        'yield', 'production', 'productivity', 'land', 'arable',
        'agri-tech', 'agritech', 'food security', 'rural',
        'plantation', 'orchard', 'greenhouse', 'nursery'
    }
    
    FARMING_KEYWORDS = {
        'farming', 'tillage', 'plowing', 'ploughing', 'sowing',
        'transplanting', 'harvesting', 'threshing', 'winnowing',
        'tractor', 'combine', 'plough', 'harrow', 'seed drill'
    }
    
    # Indian regions and states
    INDIAN_REGIONS = {
        'punjab', 'haryana', 'uttar pradesh', 'up', 'madhya pradesh', 'mp',
        'rajasthan', 'gujarat', 'maharashtra', 'karnataka', 'tamil nadu',
        'andhra pradesh', 'telangana', 'kerala', 'odisha', 'west bengal',
        'bihar', 'jharkhand', 'chhattisgarh', 'assam', 'himachal pradesh',
        'uttarakhand', 'punjab', 'haryana', 'delhi', 'goa',
        'indo-gangetic', 'deccan', 'western ghats', 'eastern ghats'
    }
    
    def __init__(self, min_relevance_score: float = 0.3):
        """
        Initialize content extractor.
        
        Args:
            min_relevance_score: Minimum relevance score for content (0.0-1.0)
        """
        self.min_relevance_score = min_relevance_score
        
        # Compile regex patterns
        self.agriculture_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.AGRICULTURE_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        
        self.farming_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(kw) for kw in self.FARMING_KEYWORDS) + r')\b',
            re.IGNORECASE
        )
        
        self.region_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(region) for region in self.INDIAN_REGIONS) + r')\b',
            re.IGNORECASE
        )
        
        logging.info(f"🌾 Improved Content Extractor initialized")
    
    def is_agriculture_relevant(self, text: str, title: str = "") -> Tuple[bool, float]:
        """
        Check if content is agriculture-relevant with confidence score.
        
        Args:
            text: Content text
            title: Content title (weighted more heavily)
            
        Returns:
            Tuple of (is_relevant, relevance_score)
        """
        if not text:
            return False, 0.0
        
        combined_text = f"{title} {title} {text}"  # Weight title 2x
        combined_text_lower = combined_text.lower()
        
        # Count keyword matches
        agri_matches = len(self.agriculture_pattern.findall(combined_text))
        farming_matches = len(self.farming_pattern.findall(combined_text))
        region_matches = len(self.region_pattern.findall(combined_text))
        
        # Calculate word count
        word_count = len(combined_text.split())
        if word_count == 0:
            return False, 0.0
        
        # Calculate density scores
        agri_density = agri_matches / word_count
        farming_density = farming_matches / word_count
        region_density = region_matches / word_count
        
        # Weighted relevance score
        relevance_score = (
            agri_density * 0.5 +
            farming_density * 0.3 +
            region_density * 0.2
        )
        
        # Bonus for title matches
        title_lower = title.lower()
        if any(kw in title_lower for kw in ['agriculture', 'farming', 'crop', 'soil', 'kisan']):
            relevance_score += 0.2
        
        # Cap at 1.0
        relevance_score = min(1.0, relevance_score)
        
        is_relevant = relevance_score >= self.min_relevance_score
        
        logging.debug(f"Relevance: {relevance_score:.3f} ({'✅' if is_relevant else '❌'}) - {title[:50]}")
        
        return is_relevant, relevance_score
    
    def extract_agriculture_paragraphs(self, text: str, min_paragraph_length: int = 50) -> List[str]:
        """
        Extract paragraphs containing agriculture-related content.
        Fixes brittle extraction logic with proper precedence.
        
        Args:
            text: Full text content
            min_paragraph_length: Minimum paragraph length to consider
            
        Returns:
            List of relevant paragraphs
        """
        if not text:
            return []
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        relevant_paragraphs = []
        
        for para in paragraphs:
            para = para.strip()
            
            # Skip short paragraphs
            if len(para) < min_paragraph_length:
                continue
            
            # Check relevance with proper logical precedence
            para_lower = para.lower()
            
            # Use explicit parentheses for clarity
            has_agriculture = any(kw in para_lower for kw in ['agriculture', 'agricultural'])
            has_farming = any(kw in para_lower for kw in ['farming', 'farm', 'crop'])
            has_indian_context = any(region in para_lower for region in self.INDIAN_REGIONS)
            
            # Improved logic: agriculture OR farming, with bonus for Indian context
            if (has_agriculture or has_farming) or (has_indian_context and len(para) > 100):
                relevant_paragraphs.append(para)
        
        return relevant_paragraphs
    
    def extract_key_terms(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract key agriculture-related terms from text.
        
        Args:
            text: Text content
            top_n: Number of top terms to return
            
        Returns:
            List of key terms
        """
        if not text:
            return []
        
        # Find all agriculture-related terms
        agri_terms = self.agriculture_pattern.findall(text.lower())
        farming_terms = self.farming_pattern.findall(text.lower())
        
        # Combine and count
        all_terms = agri_terms + farming_terms
        term_counts = Counter(all_terms)
        
        # Return top N most common
        return [term for term, count in term_counts.most_common(top_n)]
    
    def extract_structured_info(self, text: str, title: str = "") -> Dict:
        """
        Extract structured information from agriculture content.
        
        Args:
            text: Content text
            title: Content title
            
        Returns:
            Dictionary of extracted structured information
        """
        if not text:
            return {
                'is_relevant': False,
                'relevance_score': 0.0,
                'key_terms': [],
                'paragraphs': [],
                'regions_mentioned': [],
                'word_count': 0
            }
        
        # Check relevance
        is_relevant, relevance_score = self.is_agriculture_relevant(text, title)
        
        # Extract paragraphs
        paragraphs = self.extract_agriculture_paragraphs(text)
        
        # Extract key terms
        key_terms = self.extract_key_terms(text)
        
        # Extract Indian regions mentioned
        regions = list(set(self.region_pattern.findall(text.lower())))
        
        # Word count
        word_count = len(text.split())
        
        return {
            'is_relevant': is_relevant,
            'relevance_score': relevance_score,
            'key_terms': key_terms,
            'paragraphs': paragraphs[:10],  # Limit to top 10
            'regions_mentioned': regions,
            'word_count': word_count,
            'paragraph_count': len(paragraphs)
        }
    
    def create_enhanced_abstract(self, text: str, title: str = "", max_length: int = 500) -> str:
        """
        Create enhanced abstract focusing on agriculture-relevant content.
        
        Args:
            text: Full text content
            title: Content title
            max_length: Maximum abstract length
            
        Returns:
            Enhanced abstract
        """
        # Extract relevant paragraphs
        paragraphs = self.extract_agriculture_paragraphs(text)
        
        if not paragraphs:
            # Fallback to first paragraph
            paragraphs = re.split(r'\n\s*\n', text)[:1]
        
        # Combine paragraphs until max length
        abstract_parts = []
        current_length = 0
        
        for para in paragraphs:
            if current_length + len(para) > max_length:
                # Add partial paragraph
                remaining = max_length - current_length
                if remaining > 50:
                    abstract_parts.append(para[:remaining] + "...")
                break
            
            abstract_parts.append(para)
            current_length += len(para)
        
        abstract = " ".join(abstract_parts).strip()
        
        # Ensure abstract is not empty
        if not abstract and text:
            abstract = text[:max_length] + ("..." if len(text) > max_length else "")
        
        return abstract
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\-\(\)\[\]\:\;]', '', text)
        
        # Trim
        text = text.strip()
        
        return text


# Convenience function
def extract_agriculture_content(text: str, title: str = "", 
                                min_relevance: float = 0.3) -> Dict:
    """
    Convenience function to extract agriculture content with one call.
    
    Args:
        text: Content text
        title: Content title
        min_relevance: Minimum relevance score
        
    Returns:
        Extracted content dictionary
    """
    extractor = ImprovedContentExtractor(min_relevance)
    return extractor.extract_structured_info(text, title)
