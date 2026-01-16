#!/usr/bin/env python3
"""
LLM Output Parser with Schema Validation and JSON Repair
Handles fragile LLM JSON outputs with robust parsing and validation
"""

import json
import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime


try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    logging.warning("jsonschema not installed. Install with: pip install jsonschema")
    JSONSCHEMA_AVAILABLE = False


class LLMOutputParser:
    """
    Robust parser for LLM outputs with JSON repair and schema validation.
    """
    
    # Default schema for agriculture content analysis
    AGRICULTURE_ANALYSIS_SCHEMA = {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "key_insights": {"type": "array", "items": {"type": "string"}},
            "indian_context": {"type": "string"},
            "actionable_info": {"type": "string"},
            "data_points": {"type": "array", "items": {"type": "string"}},
            "geographic_relevance": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["domain", "relevance_score"]
    }
    
    def __init__(self, schema: Optional[Dict] = None, max_repair_attempts: int = 3):
        """
        Initialize LLM output parser.
        
        Args:
            schema: JSON schema for validation (defaults to agriculture analysis schema)
            max_repair_attempts: Maximum number of JSON repair attempts
        """
        self.schema = schema or self.AGRICULTURE_ANALYSIS_SCHEMA
        self.max_repair_attempts = max_repair_attempts
        self.use_jsonschema = JSONSCHEMA_AVAILABLE
        
        logging.info(f"🔧 LLM Output Parser initialized with schema validation: {self.use_jsonschema}")
    
    def parse_and_validate(self, llm_output: str, fallback_data: Optional[Dict] = None) -> Dict:
        """
        Parse and validate LLM output with automatic repair attempts.
        
        Args:
            llm_output: Raw LLM output string
            fallback_data: Fallback data if all parsing attempts fail
            
        Returns:
            Parsed and validated dictionary
        """
        if not llm_output or not llm_output.strip():
            logging.warning("Empty LLM output received")
            return fallback_data or self._get_default_fallback()
        
        # Attempt 1: Direct JSON parse
        parsed = self._try_direct_parse(llm_output)
        if parsed and self._validate(parsed):
            return parsed
        
        # Attempt 2: Extract JSON from markdown code blocks
        parsed = self._extract_from_markdown(llm_output)
        if parsed and self._validate(parsed):
            return parsed
        
        # Attempt 3: Repair common JSON errors
        for attempt in range(self.max_repair_attempts):
            repaired = self._repair_json(llm_output, attempt)
            if repaired and self._validate(repaired):
                logging.info(f"✅ Successfully repaired JSON on attempt {attempt + 1}")
                return repaired
        
        # Attempt 4: Extract key-value pairs with regex
        parsed = self._extract_key_values(llm_output)
        if parsed and self._validate(parsed):
            return parsed
        
        # All attempts failed - return fallback
        logging.error("❌ All parsing attempts failed, returning fallback data")
        return fallback_data or self._get_default_fallback()
    
    def _try_direct_parse(self, text: str) -> Optional[Dict]:
        """Try direct JSON parsing"""
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None
    
    def _extract_from_markdown(self, text: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks"""
        # Look for ```json ... ``` or ``` ... ```
        patterns = [
            r'```json\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```',
            r'`(.*?)`'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                parsed = self._try_direct_parse(match)
                if parsed:
                    return parsed
        
        return None
    
    def _repair_json(self, text: str, attempt: int) -> Optional[Dict]:
        """
        Attempt to repair malformed JSON.
        
        Different strategies based on attempt number:
        - Attempt 0: Fix common quote/escape issues
        - Attempt 1: Remove trailing commas
        - Attempt 2: Add missing brackets/braces
        """
        try:
            if attempt == 0:
                # Fix single quotes to double quotes
                repaired = text.replace("'", '"')
                # Fix unescaped newlines in strings
                repaired = re.sub(r'(?<!\\)\\n', r'\\\\n', repaired)
                return self._try_direct_parse(repaired)
            
            elif attempt == 1:
                # Remove trailing commas before } or ]
                repaired = re.sub(r',\s*([}\]])', r'\1', text)
                return self._try_direct_parse(repaired)
            
            elif attempt == 2:
                # Try to extract just the JSON object
                # Look for first { and last }
                start = text.find('{')
                end = text.rfind('}')
                if start != -1 and end != -1 and end > start:
                    repaired = text[start:end+1]
                    return self._try_direct_parse(repaired)
            
        except Exception as e:
            logging.debug(f"Repair attempt {attempt} failed: {e}")
        
        return None
    
    def _extract_key_values(self, text: str) -> Optional[Dict]:
        """
        Extract key-value pairs using regex when JSON parsing fails.
        Looks for patterns like "key": "value" or key: value
        """
        try:
            result = {}
            
            # Pattern for "key": "value" or "key": value or "key": [...]
            patterns = [
                r'"([^"]+)":\s*"([^"]*)"',  # "key": "value"
                r'"([^"]+)":\s*(\d+\.?\d*)',  # "key": number
                r'"([^"]+)":\s*\[(.*?)\]',  # "key": [array]
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                for key, value in matches:
                    if pattern.endswith(r'\]'):  # Array pattern
                        # Parse array items
                        items = re.findall(r'"([^"]+)"', value)
                        result[key] = items
                    elif re.match(r'\d+\.?\d*$', value):  # Number
                        result[key] = float(value)
                    else:
                        result[key] = value
            
            return result if result else None
            
        except Exception as e:
            logging.debug(f"Key-value extraction failed: {e}")
            return None
    
    def _validate(self, data: Dict) -> bool:
        """Validate parsed data against schema"""
        if not data:
            return False
        
        if not self.use_jsonschema:
            # Basic validation without jsonschema
            return self._basic_validate(data)
        
        try:
            jsonschema.validate(instance=data, schema=self.schema)
            return True
        except jsonschema.ValidationError as e:
            logging.debug(f"Schema validation failed: {e.message}")
            return False
        except Exception as e:
            logging.debug(f"Validation error: {e}")
            return False
    
    def _basic_validate(self, data: Dict) -> bool:
        """Basic validation without jsonschema library"""
        if not isinstance(data, dict):
            return False
        
        # Check required fields from schema
        required = self.schema.get("required", [])
        for field in required:
            if field not in data:
                return False
        
        # Check type constraints for key fields
        properties = self.schema.get("properties", {})
        for key, constraints in properties.items():
            if key in data:
                expected_type = constraints.get("type")
                value = data[key]
                
                if expected_type == "string" and not isinstance(value, str):
                    return False
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    return False
                elif expected_type == "array" and not isinstance(value, list):
                    return False
                
                # Check number ranges
                if expected_type == "number":
                    if "minimum" in constraints and value < constraints["minimum"]:
                        return False
                    if "maximum" in constraints and value > constraints["maximum"]:
                        return False
        
        return True
    
    def _get_default_fallback(self) -> Dict:
        """Get default fallback data matching the schema"""
        return {
            "domain": "general",
            "relevance_score": 0.5,
            "key_insights": [],
            "indian_context": "",
            "actionable_info": "",
            "data_points": [],
            "geographic_relevance": [],
            "llm_processed": False,
            "parse_error": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def sanitize_output(self, data: Dict) -> Dict:
        """
        Sanitize and clean validated data.
        Ensures data types, limits array sizes, etc.
        """
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                # Limit array sizes to prevent bloat
                sanitized[key] = value[:20]
            elif isinstance(value, str):
                # Trim long strings
                sanitized[key] = value[:5000] if len(value) > 5000 else value
            elif isinstance(value, (int, float)):
                # Keep numbers as-is
                sanitized[key] = value
            else:
                # Convert other types to string
                sanitized[key] = str(value)
        
        # Add metadata
        sanitized['parsed_at'] = datetime.now().isoformat()
        
        return sanitized


# Query generation schema
QUERY_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 10
        },
        "specialization": {"type": "string"},
        "rationale": {"type": "string"}
    },
    "required": ["queries"]
}


def create_agriculture_parser() -> LLMOutputParser:
    """Create parser for agriculture content analysis"""
    return LLMOutputParser(schema=LLMOutputParser.AGRICULTURE_ANALYSIS_SCHEMA)


def create_query_parser() -> LLMOutputParser:
    """Create parser for query generation"""
    return LLMOutputParser(schema=QUERY_GENERATION_SCHEMA)
