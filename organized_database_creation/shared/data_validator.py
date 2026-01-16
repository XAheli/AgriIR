#!/usr/bin/env python3
"""
Data Validator - Shared utility for validating agriculture data entries
"""

import re
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime


class AgricultureDataValidator:
    """Validator for agriculture data entries"""
    
    def __init__(self):
        self.required_fields = [
            'title', 'link', 'text_extracted', 'abstract', 'genre',
            'tags', 'indian_regions', 'crop_types', 'farming_methods',
            'soil_types', 'climate_info', 'fertilizers', 'plant_species',
            'data_type', 'source_domain', 'extraction_timestamp',
            'relevance_score', 'content_length', 'content_hash',
            'url_hash', 'is_pdf'
        ]
        
        self.optional_fields = [
            'author', 'watering_schedule', 'publication_year', 'pdf_path'
        ]
        
        self.valid_genres = [
            'survey', 'dataset', 'pdf', 'book', 'report', 'article',
            'news', 'thesis', 'manual', 'policy'
        ]
        
        self.valid_data_types = [
            'statistical', 'qualitative', 'mixed', 'technical', 'policy'
        ]
    
    def validate_entry(self, entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a single data entry"""
        errors = []
        
        # Check required fields
        for field in self.required_fields:
            if field not in entry:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors
        
        # Validate data types
        type_errors = self._validate_data_types(entry)
        errors.extend(type_errors)
        
        # Validate field values
        value_errors = self._validate_field_values(entry)
        errors.extend(value_errors)
        
        # Validate content quality
        quality_errors = self._validate_content_quality(entry)
        errors.extend(quality_errors)
        
        return len(errors) == 0, errors
    
    def _validate_data_types(self, entry: Dict[str, Any]) -> List[str]:
        """Validate data types of fields"""
        errors = []
        
        type_checks = [
            ('title', str),
            ('link', str),
            ('text_extracted', str),
            ('abstract', str),
            ('genre', str),
            ('tags', list),
            ('indian_regions', list),
            ('crop_types', list),
            ('farming_methods', list),
            ('soil_types', list),
            ('climate_info', list),
            ('fertilizers', list),
            ('plant_species', list),
            ('data_type', str),
            ('source_domain', str),
            ('extraction_timestamp', str),
            ('relevance_score', (int, float)),
            ('content_length', int),
            ('content_hash', str),
            ('url_hash', str),
            ('is_pdf', bool)
        ]
        
        for field, expected_type in type_checks:
            if field in entry and not isinstance(entry[field], expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type}, got {type(entry[field])}")
        
        return errors
    
    def _validate_field_values(self, entry: Dict[str, Any]) -> List[str]:
        """Validate specific field values"""
        errors = []
        
        # Validate genre
        if entry.get('genre') not in self.valid_genres:
            errors.append(f"Invalid genre: {entry.get('genre')}. Must be one of {self.valid_genres}")
        
        # Validate data_type
        if entry.get('data_type') not in self.valid_data_types:
            errors.append(f"Invalid data_type: {entry.get('data_type')}. Must be one of {self.valid_data_types}")
        
        # Validate URL format
        if not self._is_valid_url(entry.get('link', '')):
            errors.append(f"Invalid URL format: {entry.get('link')}")
        
        # Validate relevance score range
        relevance = entry.get('relevance_score', 0)
        if not (0.0 <= relevance <= 1.0):
            errors.append(f"Relevance score must be between 0.0 and 1.0, got {relevance}")
        
        # Validate timestamp format
        if not self._is_valid_timestamp(entry.get('extraction_timestamp', '')):
            errors.append(f"Invalid timestamp format: {entry.get('extraction_timestamp')}")
        
        # Validate publication year if present
        pub_year = entry.get('publication_year')
        if pub_year is not None and not (1900 <= pub_year <= 2030):
            errors.append(f"Invalid publication year: {pub_year}")
        
        return errors
    
    def _validate_content_quality(self, entry: Dict[str, Any]) -> List[str]:
        """Validate content quality metrics"""
        errors = []
        
        # Minimum content length
        content_length = entry.get('content_length', 0)
        if content_length < 50:
            errors.append(f"Content too short: {content_length} characters (minimum 50)")
        
        # Check for empty required text fields
        text_fields = ['title', 'text_extracted', 'abstract']
        for field in text_fields:
            if not entry.get(field, '').strip():
                errors.append(f"Empty or whitespace-only {field}")
        
        # Validate hash lengths
        if len(entry.get('content_hash', '')) != 32:
            errors.append("Invalid content_hash length (should be 32 characters)")
        
        if len(entry.get('url_hash', '')) != 32:
            errors.append("Invalid url_hash length (should be 32 characters)")
        
        return errors
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL format is valid"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None
    
    def _is_valid_timestamp(self, timestamp: str) -> bool:
        """Check if timestamp format is valid ISO 8601"""
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return True
        except ValueError:
            return False
    
    def get_validation_summary(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get validation summary for a list of entries"""
        total_entries = len(entries)
        valid_entries = 0
        all_errors = []
        
        for i, entry in enumerate(entries):
            is_valid, errors = self.validate_entry(entry)
            if is_valid:
                valid_entries += 1
            else:
                all_errors.extend([f"Entry {i}: {error}" for error in errors])
        
        return {
            'total_entries': total_entries,
            'valid_entries': valid_entries,
            'invalid_entries': total_entries - valid_entries,
            'validation_rate': valid_entries / total_entries if total_entries > 0 else 0,
            'errors': all_errors[:50]  # Limit to first 50 errors
        }


class ContentQualityAssessor:
    """Assess content quality for agriculture data"""
    
    def __init__(self):
        self.agriculture_keywords = [
            'agriculture', 'farming', 'crop', 'soil', 'irrigation', 'fertilizer',
            'pest', 'disease', 'yield', 'harvest', 'cultivation', 'organic',
            'sustainable', 'precision', 'technology', 'research', 'study'
        ]
        
        self.indian_keywords = [
            'India', 'Indian', 'Punjab', 'Maharashtra', 'Tamil Nadu', 'Karnataka',
            'Uttar Pradesh', 'West Bengal', 'Gujarat', 'Rajasthan', 'Kerala',
            'Andhra Pradesh', 'ICAR', 'IARI', 'government', 'ministry'
        ]
    
    def assess_quality(self, entry: Dict[str, Any]) -> Dict[str, float]:
        """Assess overall quality of a data entry"""
        content = entry.get('text_extracted', '').lower()
        title = entry.get('title', '').lower()
        
        # Agriculture relevance (0-1)
        agriculture_score = self._calculate_keyword_score(content + ' ' + title, self.agriculture_keywords)
        
        # Indian context relevance (0-1)
        indian_score = self._calculate_keyword_score(content + ' ' + title, self.indian_keywords)
        
        # Content completeness (0-1)
        completeness_score = self._calculate_completeness_score(entry)
        
        # Source credibility (0-1)
        credibility_score = self._calculate_credibility_score(entry.get('source_domain', ''))
        
        # Overall quality score
        overall_score = (agriculture_score * 0.3 + indian_score * 0.25 + 
                        completeness_score * 0.25 + credibility_score * 0.2)
        
        return {
            'agriculture_relevance': agriculture_score,
            'indian_context': indian_score,
            'content_completeness': completeness_score,
            'source_credibility': credibility_score,
            'overall_quality': overall_score
        }
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        if not text:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(matches / len(keywords), 1.0)
    
    def _calculate_completeness_score(self, entry: Dict[str, Any]) -> float:
        """Calculate content completeness score"""
        score = 0.0
        
        # Check text length
        content_length = entry.get('content_length', 0)
        if content_length > 1000:
            score += 0.3
        elif content_length > 500:
            score += 0.2
        elif content_length > 100:
            score += 0.1
        
        # Check metadata completeness
        if entry.get('author'):
            score += 0.1
        if entry.get('publication_year'):
            score += 0.1
        if entry.get('tags') and len(entry['tags']) > 0:
            score += 0.1
        if entry.get('crop_types') and len(entry['crop_types']) > 0:
            score += 0.1
        if entry.get('indian_regions') and len(entry['indian_regions']) > 0:
            score += 0.1
        if entry.get('farming_methods') and len(entry['farming_methods']) > 0:
            score += 0.1
        if entry.get('abstract') and len(entry['abstract']) > 50:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_credibility_score(self, domain: str) -> float:
        """Calculate source credibility score"""
        if not domain:
            return 0.3
        
        # High credibility domains
        high_credibility = [
            'icar.org.in', 'iari.res.in', 'icrisat.org', 'agricoop.nic.in',
            'gov.in', 'edu', 'ac.in', 'res.in'
        ]
        
        # Medium credibility domains
        medium_credibility = [
            'researchgate.net', 'springer.com', 'sciencedirect.com',
            'jstor.org', 'ieee.org', 'nature.com', 'science.org'
        ]
        
        domain_lower = domain.lower()
        
        if any(cred in domain_lower for cred in high_credibility):
            return 1.0
        elif any(cred in domain_lower for cred in medium_credibility):
            return 0.7
        elif any(ext in domain_lower for ext in ['.edu', '.gov', '.org']):
            return 0.6
        else:
            return 0.4