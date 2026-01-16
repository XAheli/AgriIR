#!/usr/bin/env python3
"""
Autonomous Agriculture Data Curator with 12+ Intelligent Agents
Each agent can autonomously generate search queries and scour the internet for Indian agriculture data
"""

import asyncio
import json
import time
import threading
import logging
import hashlib
import os
import re
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import itertools

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

# Import shared utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))
from jsonl_writer import ImmediateJSONLWriter
from duplicate_tracker import get_global_tracker

# Import from keyword-based search
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "keyword_based_search" / "src"))
from agriculture_curator_fixed import (
    ImprovedPDFProcessor, ImprovedWebSearch,
    AgentStatus, CurationResult, ExpandedAgricultureQueries
)


class AutonomousSearchAgent:
    """Intelligent agent that can autonomously generate and execute agriculture searches with advanced features"""
    
    def __init__(self, agent_id: int, specialization: str, search_engine: ImprovedWebSearch, 
                 jsonl_writer: ImmediateJSONLWriter, duplicate_tracker=None):
        self.agent_id = agent_id
        self.specialization = specialization
        self.search_engine = search_engine
        self.jsonl_writer = jsonl_writer
        
        # Use global persistent duplicate tracker instead of session-only set
        self.duplicate_tracker = duplicate_tracker or get_global_tracker()
        self.processed_urls = set()  # Session tracking for analytics only
        
        self.search_history = set()  # Track search queries to avoid repetition
        self.success_patterns = []  # Track successful search patterns for learning
        self.failure_patterns = []  # Track failed patterns to avoid
        self.domain_preferences = {}  # Track which domains yield good results
        self.content_quality_scores = []  # Track content quality over time
        
        # Define specialized knowledge areas (expanded)
        self.knowledge_areas = self._define_knowledge_areas()
        self.indian_contexts = self._define_indian_contexts()
        self.search_modifiers = self._define_search_modifiers()
        self.advanced_patterns = self._define_advanced_patterns()
        self.research_institutions = self._define_research_institutions()
        self.government_schemes = self._define_government_schemes()
        
        # Advanced search strategies
        self.search_strategies = [
            "multi_domain_fusion", "temporal_analysis", "comparative_study",
            "technology_integration", "policy_impact", "regional_specificity",
            "value_chain_analysis", "sustainability_focus", "innovation_tracking",
            "problem_diagnosis", "solution_implementation", "best_practices"
        ]
        
    def _define_knowledge_areas(self) -> Dict[str, List[str]]:
        """Define comprehensive knowledge areas for autonomous search generation"""
        return {
            "crops": [
                # Cereals
                "rice", "wheat", "maize", "barley", "oats", "rye", "millets", "sorghum", "bajra", "jowar", "ragi",
                # Pulses
                "lentils", "chickpeas", "pigeon peas", "black gram", "green gram", "kidney beans", "field peas",
                # Cash crops
                "cotton", "sugarcane", "tobacco", "jute", "flax", "hemp",
                # Oilseeds
                "groundnut", "mustard", "sesame", "sunflower", "safflower", "linseed", "castor", "coconut",
                # Spices
                "turmeric", "coriander", "cumin", "fennel", "fenugreek", "cardamom", "pepper", "cinnamon", "cloves",
                # Fruits
                "mango", "banana", "apple", "grapes", "orange", "pomegranate", "guava", "papaya", "litchi",
                # Vegetables
                "potato", "onion", "tomato", "brinjal", "okra", "cabbage", "cauliflower", "peas", "beans",
                # Plantation crops
                "tea", "coffee", "rubber", "arecanut", "cocoa"
            ],
            
            "soil_science": [
                "soil fertility", "soil health", "soil organic matter", "soil pH", "soil nutrients",
                "black soil", "red soil", "alluvial soil", "laterite soil", "desert soil", "mountain soil",
                "soil erosion", "soil conservation", "soil testing", "soil amendments", "soil microbiome",
                "vermicomposting", "green manure", "biofertilizers", "soil salinity", "soil degradation"
            ],
            
            "climate_agriculture": [
                "monsoon agriculture", "drought management", "climate change adaptation", "weather forecasting",
                "rainfall patterns", "temperature effects", "humidity control", "frost protection",
                "heat stress", "water stress", "climate resilient varieties", "seasonal cropping",
                "kharif crops", "rabi crops", "zaid crops", "agroforestry", "carbon sequestration"
            ],
            
            "water_management": [
                "irrigation systems", "drip irrigation", "sprinkler irrigation", "flood irrigation",
                "micro irrigation", "precision irrigation", "water harvesting", "watershed management",
                "groundwater management", "surface water", "water quality", "saline water irrigation",
                "fertigation", "irrigation scheduling", "water use efficiency", "canal irrigation"
            ],
            
            "plant_science": [
                "plant breeding", "genetics", "biotechnology", "tissue culture", "seed technology",
                "plant pathology", "plant physiology", "plant nutrition", "photosynthesis",
                "plant hormones", "growth regulators", "flowering", "fruit development",
                "post harvest technology", "seed production", "hybrid development", "GMO crops"
            ],
            
            "pest_management": [
                "integrated pest management", "biological control", "pesticides", "insecticides",
                "fungicides", "herbicides", "nematicides", "pest resistance", "beneficial insects",
                "natural enemies", "pheromone traps", "organic pest control", "biopesticides",
                "pest monitoring", "economic threshold", "pest forecasting"
            ],
            
            "agricultural_technology": [
                "precision agriculture", "smart farming", "IoT agriculture", "drones", "sensors",
                "GPS technology", "variable rate application", "yield monitoring", "satellite imagery",
                "machine learning", "artificial intelligence", "robotics", "automation",
                "digital agriculture", "farm management software", "decision support systems"
            ],
            
            "economics_policy": [
                "agricultural economics", "farm income", "crop insurance", "price policy", "subsidies",
                "credit systems", "marketing", "supply chain", "food security", "nutrition security",
                "government schemes", "minimum support price", "agricultural reforms", "trade policy",
                "export promotion", "import regulations", "value addition", "agribusiness"
            ],
            
            "sustainable_agriculture": [
                "organic farming", "natural farming", "zero budget farming", "sustainable intensification",
                "conservation agriculture", "agroecology", "permaculture", "biodynamic farming",
                "regenerative agriculture", "circular agriculture", "low external input agriculture",
                "integrated farming systems", "crop diversification", "sustainable livestock"
            ],
            
            "research_institutions": [
                "ICAR", "IARI", "ICRISAT", "IRRI", "CIMMYT", "agricultural universities",
                "research stations", "KVK", "extension services", "farmer field schools",
                "demonstration plots", "technology transfer", "capacity building", "training programs"
            ]
        }
    
    def _define_indian_contexts(self) -> List[str]:
        """Define Indian geographical and cultural contexts"""
        return [
            # States and regions
            "Punjab", "Haryana", "Uttar Pradesh", "Bihar", "West Bengal", "Odisha", "Jharkhand",
            "Madhya Pradesh", "Chhattisgarh", "Maharashtra", "Gujarat", "Rajasthan",
            "Karnataka", "Andhra Pradesh", "Telangana", "Tamil Nadu", "Kerala",
            "Himachal Pradesh", "Uttarakhand", "Jammu Kashmir", "Sikkim", "Assam",
            "Meghalaya", "Manipur", "Tripura", "Mizoram", "Nagaland", "Arunachal Pradesh", "Goa",
            
            # Agro-climatic zones
            "Western Himalayas", "Eastern Himalayas", "Lower Gangetic Plains", "Middle Gangetic Plains",
            "Upper Gangetic Plains", "Trans Gangetic Plains", "Eastern Plateau", "Central Plateau",
            "Western Plateau", "Southern Plateau", "East Coast Plains", "West Coast Plains",
            "Gujarat Plains", "Western Dry Region", "Island Region",
            
            # Cultural and linguistic contexts
            "Bharatiya", "Hindustani", "Desi", "traditional knowledge", "indigenous practices",
            "tribal agriculture", "women farmers", "small farmers", "marginal farmers",
            "farmer producer organizations", "self help groups", "cooperative farming"
        ]
    
    def _define_search_modifiers(self) -> List[str]:
        """Define search modifiers for comprehensive coverage"""
        return [
            "research", "study", "analysis", "survey", "report", "data", "statistics",
            "technology", "innovation", "best practices", "case study", "success story",
            "challenges", "problems", "solutions", "improvement", "development",
            "training", "capacity building", "extension", "demonstration",
            "policy", "scheme", "program", "initiative", "project",
            "yield", "productivity", "efficiency", "sustainability", "profitability",
            "modern", "traditional", "contemporary", "future", "trends", "outlook",
            "impact", "assessment", "evaluation", "monitoring", "implementation",
            "optimization", "enhancement", "transformation", "adaptation", "resilience"
        ]
    
    def _define_advanced_patterns(self) -> Dict[str, List[str]]:
        """Define advanced search patterns for sophisticated queries"""
        return {
            "comparative_analysis": [
                "comparison between {term1} and {term2} in {location}",
                "{term1} vs {term2} {location} agriculture study",
                "comparative analysis {term1} {term2} Indian farming"
            ],
            "temporal_trends": [
                "{term} trends over time India agriculture",
                "historical analysis {term} {location} farming",
                "evolution of {term} in Indian agriculture",
                "future prospects {term} India {year}"
            ],
            "impact_assessment": [
                "impact of {term} on {crop} yield {location}",
                "{term} effect on farmer income India",
                "economic impact {term} agriculture {state}",
                "environmental impact {term} farming practices"
            ],
            "technology_adoption": [
                "adoption of {technology} in {location} agriculture",
                "{technology} implementation challenges India",
                "barriers to {technology} adoption farmers",
                "success factors {technology} agriculture India"
            ],
            "policy_analysis": [
                "policy impact on {term} agriculture India",
                "government schemes for {term} {location}",
                "regulatory framework {term} farming India",
                "subsidy analysis {term} agriculture policy"
            ]
        }
    
    def _define_research_institutions(self) -> List[str]:
        """Define major research institutions for targeted searches"""
        return [
            "ICAR", "IARI", "ICRISAT", "IRRI", "CIMMYT", "IFPRI", "CGIAR",
            "Indian Agricultural Research Institute", "International Crops Research Institute",
            "Indian Council of Agricultural Research", "National Academy of Agricultural Sciences",
            "Central Research Institute", "State Agricultural Universities",
            "Krishi Vigyan Kendra", "ICAR Research Complex", "National Research Centre",
            "Agricultural Technology Application Research Institute",
            "Indian Institute of Technology", "Indian Institute of Science",
            "Central University", "Agricultural College", "Veterinary University"
        ]
    
    def _define_government_schemes(self) -> List[str]:
        """Define government schemes and programs for policy-focused searches"""
        return [
            "Pradhan Mantri Fasal Bima Yojana", "PM-KISAN", "Kisan Credit Card",
            "National Food Security Mission", "Rashtriya Krishi Vikas Yojana",
            "Soil Health Card Scheme", "Paramparagat Krishi Vikas Yojana",
            "National Mission on Sustainable Agriculture", "National Horticulture Mission",
            "National Livestock Mission", "Blue Revolution", "White Revolution",
            "Krishi Sinchai Yojana", "National Agricultural Market", "e-NAM",
            "Minimum Support Price", "Agricultural Marketing", "Crop Insurance",
            "Farmer Producer Organizations", "Self Help Groups", "Cooperative Societies",
            "Direct Benefit Transfer", "Input Subsidy", "Credit Subsidy"
        ]
    
    def generate_autonomous_searches(self, num_searches: int = 50) -> List[str]:
        """Generate autonomous search queries with advanced strategies and learning"""
        search_queries = []
        attempts = 0
        max_attempts = num_searches * 4  # Increased for more variety
        
        # Include comprehensive base queries from fixed version
        base_queries = ExpandedAgricultureQueries.get_search_queries(20)  # Get 20 base queries
        
        while len(search_queries) < num_searches and attempts < max_attempts:
            attempts += 1
            
            # Choose search generation strategy
            if len(search_queries) < 10:
                # Start with some base queries for foundation
                if len(search_queries) < len(base_queries):
                    query = self._enhance_base_query(base_queries[len(search_queries)])
                else:
                    query = self._generate_advanced_query()
            else:
                # Use advanced generation strategies
                strategy = random.choice([
                    "advanced_pattern", "institution_focused", "scheme_focused",
                    "technology_fusion", "comparative_analysis", "temporal_analysis",
                    "regional_deep_dive", "value_chain", "sustainability_focus"
                ])
                query = self._generate_query_by_strategy(strategy)
            
            # Ensure uniqueness, relevance, and quality
            if query and query not in self.search_history and len(query.split()) >= 3:
                # Add specialization context
                enhanced_query = self._add_specialization_context(query)
                search_queries.append(enhanced_query)
                self.search_history.add(enhanced_query)
        
        logging.info(f"🧠 Agent {self.agent_id} ({self.specialization}): Generated {len(search_queries)} advanced autonomous searches")
        return search_queries
    
    def _enhance_base_query(self, base_query: str) -> str:
        """Enhance base query with agent specialization and context"""
        # Add specialization terms
        specialization_terms = {
            "Crop Science & Plant Breeding": "varieties genetics breeding improvement",
            "Soil Science & Fertility Management": "soil health fertility nutrients management",
            "Water Resources & Irrigation": "irrigation water management efficiency",
            "Plant Protection & Pest Management": "pest disease management protection IPM",
            "Agricultural Technology & Precision Farming": "technology precision smart farming",
            "Sustainable & Organic Farming": "sustainable organic natural farming",
            "Agricultural Economics & Policy": "economics policy finance market",
            "Climate Change & Adaptation": "climate adaptation resilience mitigation"
        }
        
        terms = specialization_terms.get(self.specialization, "agriculture")
        location = random.choice(self.indian_contexts)
        
        return f"{base_query} {terms} {location}"
    
    def _generate_advanced_query(self) -> str:
        """Generate advanced queries using sophisticated patterns"""
        strategy = random.choice(list(self.advanced_patterns.keys()))
        pattern = random.choice(self.advanced_patterns[strategy])
        
        # Fill pattern with relevant terms
        if "{term1}" in pattern and "{term2}" in pattern:
            area1 = random.choice(list(self.knowledge_areas.keys()))
            area2 = random.choice(list(self.knowledge_areas.keys()))
            term1 = random.choice(self.knowledge_areas[area1])
            term2 = random.choice(self.knowledge_areas[area2])
            pattern = pattern.replace("{term1}", term1).replace("{term2}", term2)
        
        if "{term}" in pattern:
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            pattern = pattern.replace("{term}", term)
        
        if "{location}" in pattern:
            location = random.choice(self.indian_contexts)
            pattern = pattern.replace("{location}", location)
        
        if "{crop}" in pattern:
            crop = random.choice(self.knowledge_areas.get("crops", ["rice"]))
            pattern = pattern.replace("{crop}", crop)
        
        if "{technology}" in pattern:
            tech = random.choice(self.knowledge_areas.get("agricultural_technology", ["IoT"]))
            pattern = pattern.replace("{technology}", tech)
        
        if "{state}" in pattern:
            state = random.choice([s for s in self.indian_contexts if len(s.split()) <= 2])
            pattern = pattern.replace("{state}", state)
        
        if "{year}" in pattern:
            year = random.choice(["2023", "2024", "2025", "future"])
            pattern = pattern.replace("{year}", year)
        
        return pattern
    
    def _generate_query_by_strategy(self, strategy: str) -> str:
        """Generate query based on advanced strategy"""
        
        if strategy == "institution_focused":
            institution = random.choice(self.research_institutions)
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            location = random.choice(self.indian_contexts)
            return f"{institution} {term} research {location} agriculture"
            
        elif strategy == "scheme_focused":
            scheme = random.choice(self.government_schemes)
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            return f"{scheme} {term} implementation impact India"
            
        elif strategy == "technology_fusion":
            tech1 = random.choice(self.knowledge_areas.get("agricultural_technology", ["IoT"]))
            tech2 = random.choice(self.knowledge_areas.get("agricultural_technology", ["AI"]))
            crop = random.choice(self.knowledge_areas.get("crops", ["rice"]))
            location = random.choice(self.indian_contexts)
            return f"{tech1} {tech2} integration {crop} farming {location}"
            
        elif strategy == "comparative_analysis":
            method1 = random.choice(["traditional", "organic", "precision", "sustainable"])
            method2 = random.choice(["modern", "conventional", "intensive", "commercial"])
            crop = random.choice(self.knowledge_areas.get("crops", ["wheat"]))
            location = random.choice(self.indian_contexts)
            return f"{method1} vs {method2} {crop} cultivation {location} comparison"
            
        elif strategy == "temporal_analysis":
            aspect = random.choice(["productivity", "yield", "adoption", "implementation"])
            term = random.choice(self.knowledge_areas.get("crops", ["cotton"]))
            location = random.choice(self.indian_contexts)
            return f"{term} {aspect} trends over time {location} agriculture"
            
        elif strategy == "regional_deep_dive":
            location = random.choice(self.indian_contexts)
            area1 = random.choice(list(self.knowledge_areas.keys()))
            area2 = random.choice(list(self.knowledge_areas.keys()))
            term1 = random.choice(self.knowledge_areas[area1])
            term2 = random.choice(self.knowledge_areas[area2])
            return f"{location} {term1} {term2} comprehensive analysis agriculture"
            
        elif strategy == "value_chain":
            crop = random.choice(self.knowledge_areas.get("crops", ["sugarcane"]))
            aspects = ["production", "processing", "marketing", "distribution", "consumption"]
            aspect = random.choice(aspects)
            location = random.choice(self.indian_contexts)
            return f"{crop} value chain {aspect} {location} India"
            
        elif strategy == "sustainability_focus":
            sustainability_terms = ["carbon footprint", "environmental impact", "resource efficiency", "biodiversity"]
            sustain_term = random.choice(sustainability_terms)
            crop = random.choice(self.knowledge_areas.get("crops", ["rice"]))
            location = random.choice(self.indian_contexts)
            return f"{crop} farming {sustain_term} {location} sustainable agriculture"
        
        # Default fallback
        return self._generate_query_by_type("combination")
    
    def _add_specialization_context(self, query: str) -> str:
        """Add specialization-specific context to any query"""
        specialization_keywords = {
            "Crop Science & Plant Breeding": ["varieties", "genetics", "breeding", "yield"],
            "Soil Science & Fertility Management": ["soil", "fertility", "nutrients", "organic matter"],
            "Water Resources & Irrigation": ["irrigation", "water", "efficiency", "conservation"],
            "Plant Protection & Pest Management": ["pest", "disease", "protection", "management"],
            "Agricultural Technology & Precision Farming": ["technology", "precision", "smart", "digital"],
            "Sustainable & Organic Farming": ["sustainable", "organic", "natural", "eco-friendly"],
            "Agricultural Economics & Policy": ["economics", "policy", "market", "finance"],
            "Climate Change & Adaptation": ["climate", "adaptation", "resilience", "weather"],
            "Horticulture & Plantation Crops": ["horticulture", "fruits", "vegetables", "plantation"],
            "Livestock & Animal Husbandry": ["livestock", "animal", "dairy", "cattle"],
            "Food Processing & Post-Harvest": ["processing", "storage", "post-harvest", "value-addition"],
            "Rural Development & Extension": ["extension", "training", "rural", "development"]
        }
        
        keywords = specialization_keywords.get(self.specialization, ["agriculture"])
        keyword = random.choice(keywords)
        
        # Avoid adding if already present
        if keyword.lower() not in query.lower():
            return f"{query} {keyword}"
        return query
    
    def _generate_query_by_type(self, query_type: str) -> str:
        """Generate search query based on type"""
        
        if query_type == "combination":
            # Combine knowledge areas for comprehensive searches
            primary_area = random.choice(list(self.knowledge_areas.keys()))
            secondary_area = random.choice(list(self.knowledge_areas.keys()))
            
            primary_term = random.choice(self.knowledge_areas[primary_area])
            secondary_term = random.choice(self.knowledge_areas[secondary_area])
            location = random.choice(self.indian_contexts)
            
            return f"{primary_term} {secondary_term} {location} India"
            
        elif query_type == "specific_location":
            # Location-specific searches
            location = random.choice(self.indian_contexts)
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            modifier = random.choice(self.search_modifiers)
            
            return f"{term} {modifier} {location} agriculture"
            
        elif query_type == "technology_focus":
            # Technology and innovation focus
            tech_terms = self.knowledge_areas.get("agricultural_technology", [])
            crop_terms = self.knowledge_areas.get("crops", [])
            
            if tech_terms and crop_terms:
                tech = random.choice(tech_terms)
                crop = random.choice(crop_terms)
                location = random.choice(self.indian_contexts)
                
                return f"{tech} {crop} farming {location} India"
                
        elif query_type == "problem_solution":
            # Problem-solution oriented searches
            problems = ["drought", "pest", "disease", "low yield", "soil degradation", "water scarcity"]
            solutions = ["management", "control", "resistance", "adaptation", "conservation"]
            
            problem = random.choice(problems)
            solution = random.choice(solutions)
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            
            return f"{problem} {solution} {term} India agriculture"
            
        elif query_type == "research_focus":
            # Research and scientific focus
            research_terms = ["research", "development", "innovation", "breeding", "improvement"]
            area = random.choice(list(self.knowledge_areas.keys()))
            term = random.choice(self.knowledge_areas[area])
            location = random.choice(self.indian_contexts)
            
            research = random.choice(research_terms)
            return f"{term} {research} {location} agricultural science"
            
        elif query_type == "policy_focus":
            # Policy and economics focus
            policy_terms = self.knowledge_areas.get("economics_policy", [])
            if policy_terms:
                policy = random.choice(policy_terms)
                location = random.choice(self.indian_contexts)
                area = random.choice(list(self.knowledge_areas.keys()))
                term = random.choice(self.knowledge_areas[area])
                
                return f"{policy} {term} {location} government scheme"
        
        return ""
    
    def autonomous_search_and_curate(self, max_searches: int = 50) -> CurationResult:
        """Perform autonomous search and curation with advanced learning and adaptation"""
        start_time = time.time()
        
        try:
            logging.info(f"🚀 Agent {self.agent_id} ({self.specialization}): Starting advanced autonomous curation")
            
            # Generate autonomous search queries with advanced strategies
            search_queries = self.generate_autonomous_searches(max_searches)
            
            data_entries = []
            total_search_results = 0
            pdfs_processed = 0
            successful_searches = 0
            failed_searches = 0
            quality_scores = []
            
            for i, query in enumerate(search_queries):
                logging.info(f"🔍 Agent {self.agent_id}: Advanced search {i+1}/{len(search_queries)}: {query}")
                
                try:
                    # Perform search with quality tracking
                    search_results = self.search_engine.search_and_extract(query, self.agent_id)
                    total_search_results += len(search_results)
                    
                    if search_results:
                        successful_searches += 1
                        # Track successful pattern for learning
                        self.success_patterns.append(query)
                    else:
                        failed_searches += 1
                        # Track failed pattern to avoid in future
                        self.failure_patterns.append(query)
                    
                    # Process results with quality assessment and persistent duplicate checking
                    batch_quality_scores = []
                    for result in search_results:
                        url = result.get('url', '')
                        title = result.get('title', '')
                        
                        # CRITICAL FIX #1: Check persistent global duplicate tracker first
                        if self.duplicate_tracker.is_duplicate_url(url):
                            logging.debug(f"🔄 Agent {self.agent_id}: Skipping duplicate URL: {url}")
                            continue
                        
                        # Check content hash for deeper duplicate detection
                        content = result.get('full_content', result.get('text_extracted', ''))
                        if content and self.duplicate_tracker.is_duplicate_content(title, content):
                            logging.debug(f"🔄 Agent {self.agent_id}: Skipping duplicate content from: {url}")
                            continue
                        
                        # CRITICAL FIX #2: Check if this is a PDF and process with OCR
                        is_pdf = url.lower().endswith('.pdf') or 'pdf' in url.lower()
                        if is_pdf and self.search_engine.pdf_processor:
                            logging.info(f"📄 Agent {self.agent_id}: Processing PDF with OCR: {title[:100]}")
                            pdf_data = self.search_engine.pdf_processor.download_and_process_pdf(url, title, query)
                            
                            if pdf_data and pdf_data.get('saved_to_jsonl'):
                                # PDF was successfully processed with OCR and saved
                                logging.info(f"✅ Agent {self.agent_id}: PDF with OCR saved: {title[:100]}")
                                pdfs_processed += 1
                                
                                # CRITICAL FIX #3: Mark as processed in duplicate tracker
                                self.duplicate_tracker.mark_processed(url, title, pdf_data.get('text_extracted', ''), success=True)
                                self.processed_urls.add(url)  # Session analytics
                                
                                data_entries.append(pdf_data)
                                
                                # Track quality metrics
                                quality_score = self._calculate_content_quality(pdf_data)
                                batch_quality_scores.append(quality_score)
                                quality_scores.append(quality_score)
                                
                                # Update domain preferences
                                domain = pdf_data.get('source_domain', '')
                                if domain:
                                    self._update_domain_preference(domain, quality_score)
                                
                                # CRITICAL FIX: Update learning patterns for PDFs
                                if quality_score > 0.7:
                                    self.success_patterns.append(query)
                                    logging.debug(f"📚 Agent {self.agent_id}: PDF success pattern learned: {query[:50]}")
                                else:
                                    self.failure_patterns.append(query)
                                    logging.debug(f"⚠️ Agent {self.agent_id}: PDF quality low, adjusting strategy")
                                
                                continue  # Skip to next result
                        
                        # Mark as processed in session tracker
                        self.processed_urls.add(url)  # Session analytics
                        
                        # Process with quality tracking (web content or non-PDF)
                        processed_entry = self._process_autonomous_result_with_quality(result, query)
                        if processed_entry:
                            # CRITICAL FIX #4: Mark as processed in duplicate tracker
                            entry_content = processed_entry.get('text_extracted', processed_entry.get('full_content', ''))
                            entry_title = processed_entry.get('title', title)
                            self.duplicate_tracker.mark_processed(url, entry_title, entry_content, success=True)
                            
                            data_entries.append(processed_entry)
                            if processed_entry.get('is_pdf'):
                                pdfs_processed += 1
                            
                            # Track quality metrics
                            quality_score = self._calculate_content_quality(processed_entry)
                            batch_quality_scores.append(quality_score)
                            quality_scores.append(quality_score)
                            
                            # Update domain preferences based on quality
                            domain = processed_entry.get('source_domain', '')
                            if domain:
                                self._update_domain_preference(domain, quality_score)
                        else:
                            # CRITICAL FIX: Mark failed URLs to prevent infinite retries
                            logging.warning(f"⚠️ Agent {self.agent_id}: Failed to process {url}, marking as failed")
                            self.duplicate_tracker.mark_processed(url, title, "", success=False)
                    
                    # Log batch quality
                    if batch_quality_scores:
                        avg_batch_quality = sum(batch_quality_scores) / len(batch_quality_scores)
                        logging.info(f"📊 Agent {self.agent_id}: Batch quality score: {avg_batch_quality:.3f}")
                    
                except Exception as e:
                    logging.warning(f"⚠️ Agent {self.agent_id}: Search failed for '{query}': {e}")
                    failed_searches += 1
                    self.failure_patterns.append(query)
                
                # Progressive logging and adaptation
                if (i + 1) % 10 == 0:
                    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
                    logging.info(f"📊 Agent {self.agent_id}: Progress {i+1}/{len(search_queries)}")
                    logging.info(f"    📝 Entries: {len(data_entries)}, Quality: {avg_quality:.3f}")
                    logging.info(f"    ✅ Success: {successful_searches}, ❌ Failed: {failed_searches}")
                    
                    # OPTIMIZED: Reduced adaptive delay for faster processing
                    if successful_searches > 0:
                        success_rate = successful_searches / (successful_searches + failed_searches)
                        if success_rate > 0.8:
                            time.sleep(0.1)  # Minimal delay when successful (was 0.3)
                        elif success_rate > 0.5:
                            time.sleep(0.2)  # Normal speed (was 0.5)
                        else:
                            time.sleep(0.4)  # Slower when failing (was 0.8)
                    else:
                        time.sleep(0.2)  # Reduced from 0.5
                else:
                    # OPTIMIZED: Minimal respectful delay for continuous operation
                    time.sleep(0.1)  # Reduced from 0.3 for maximum throughput
            
            execution_time = time.time() - start_time
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            logging.info(f"✅ Agent {self.agent_id} advanced autonomous curation completed!")
            logging.info(f"📈 Final stats: {len(data_entries)} entries, {pdfs_processed} PDFs")
            logging.info(f"🎯 Quality score: {avg_quality:.3f}, Success rate: {successful_searches}/{len(search_queries)}")
            logging.info(f"🌐 Unique URLs: {len(self.processed_urls)}, Top domains: {self._get_top_domains()}")
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Advanced autonomous curation: {len(search_queries)} queries ({self.specialization})",
                port=0,
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
            logging.error(f"❌ Agent {self.agent_id} advanced autonomous curation failed: {e}")
            
            return CurationResult(
                agent_id=self.agent_id,
                search_query=f"Failed advanced autonomous curation",
                port=0,
                success=False,
                execution_time=execution_time,
                error_message=str(e),
                status=AgentStatus.FAILED
            )
    
    def _process_autonomous_result_with_quality(self, result: Dict, query: str) -> Optional[Dict]:
        """Process autonomous search result with enhanced metadata and quality tracking"""
        try:
            # Add comprehensive autonomous search metadata
            enhanced_result = result.copy()
            enhanced_result.update({
                'autonomous_search': True,
                'agent_specialization': self.specialization,
                'search_query_generated': query,
                'discovery_timestamp': datetime.now().isoformat(),
                'agent_id': self.agent_id,
                'search_strategy': self._identify_search_strategy(query),
                'content_category': self._categorize_content(result),
                'data_richness_score': self._calculate_data_richness(result),
                'agriculture_relevance': self._calculate_agriculture_relevance(result),
                'indian_context_score': self._calculate_indian_context(result)
            })
            
            return enhanced_result
            
        except Exception as e:
            logging.warning(f"⚠️ Agent {self.agent_id}: Failed to process autonomous result: {e}")
            return None
    
    def _calculate_content_quality(self, entry: Dict) -> float:
        """Calculate comprehensive content quality score"""
        score = 0.0
        
        # Content length score (0-0.2)
        content_length = entry.get('content_length', 0)
        if content_length > 5000:
            score += 0.2
        elif content_length > 2000:
            score += 0.15
        elif content_length > 500:
            score += 0.1
        elif content_length > 100:
            score += 0.05
        
        # Agriculture relevance (0-0.3)
        relevance = entry.get('agriculture_relevance', 0)
        score += min(relevance * 0.3, 0.3)
        
        # Indian context (0-0.2)
        indian_score = entry.get('indian_context_score', 0)
        score += min(indian_score * 0.2, 0.2)
        
        # Data richness (0-0.2)
        richness = entry.get('data_richness_score', 0)
        score += min(richness * 0.2, 0.2)
        
        # PDF bonus (0-0.1)
        if entry.get('is_pdf', False):
            score += 0.1
        
        return min(score, 1.0)
    
    def _identify_search_strategy(self, query: str) -> str:
        """Identify which search strategy was used"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['vs', 'comparison', 'compare']):
            return 'comparative_analysis'
        elif any(word in query_lower for word in ['trend', 'over time', 'historical']):
            return 'temporal_analysis'
        elif any(word in query_lower for word in ['impact', 'effect', 'influence']):
            return 'impact_assessment'
        elif any(inst in query_lower for inst in ['icar', 'iari', 'research']):
            return 'institution_focused'
        elif any(tech in query_lower for tech in ['iot', 'ai', 'precision', 'smart']):
            return 'technology_integration'
        else:
            return 'general_exploration'
    
    def _categorize_content(self, result: Dict) -> str:
        """Categorize content type with detailed classification"""
        title = result.get('title', '').lower()
        content = result.get('full_content', '').lower()
        url = result.get('url', '').lower()
        
        if result.get('is_pdf', False):
            if any(word in title for word in ['research', 'study', 'analysis']):
                return 'research_paper'
            elif any(word in title for word in ['report', 'survey']):
                return 'technical_report'
            else:
                return 'pdf_document'
        elif any(domain in url for domain in ['gov.in', 'nic.in']):
            return 'government_content'
        elif any(domain in url for domain in ['edu', 'ac.in', 'university']):
            return 'academic_content'
        elif any(word in title for word in ['news', 'article']):
            return 'news_article'
        else:
            return 'web_content'
    
    def _calculate_data_richness(self, result: Dict) -> float:
        """Calculate how data-rich the content is"""
        score = 0.0
        
        # Check for structured data
        if result.get('tags'):
            score += 0.2
        if result.get('indian_regions'):
            score += 0.2
        if result.get('crop_types'):
            score += 0.2
        if result.get('soil_types'):
            score += 0.1
        if result.get('climate_info'):
            score += 0.1
        if result.get('fertilizers'):
            score += 0.1
        if result.get('farming_methods'):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_agriculture_relevance(self, result: Dict) -> float:
        """Calculate agriculture relevance score"""
        text = (result.get('title', '') + ' ' + result.get('full_content', '')).lower()
        
        agriculture_keywords = [
            'agriculture', 'farming', 'crop', 'plant', 'soil', 'irrigation',
            'fertilizer', 'pest', 'disease', 'yield', 'cultivation', 'harvest'
        ]
        
        score = 0.0
        for keyword in agriculture_keywords:
            count = text.count(keyword)
            score += min(count * 0.1, 0.2)  # Max 0.2 per keyword
        
        return min(score, 1.0)
    
    def _calculate_indian_context(self, result: Dict) -> float:
        """Calculate Indian context relevance"""
        text = (result.get('title', '') + ' ' + result.get('full_content', '')).lower()
        
        indian_keywords = ['india', 'indian', 'bharatiya', 'hindustan']
        states = self.indian_contexts
        
        score = 0.0
        
        # Indian keywords
        for keyword in indian_keywords:
            if keyword in text:
                score += 0.3
        
        # State/region mentions
        for state in states[:10]:  # Check top 10 states
            if state.lower() in text:
                score += 0.1
        
        return min(score, 1.0)
    
    def _update_domain_preference(self, domain: str, quality_score: float):
        """Update domain preferences based on quality"""
        if domain not in self.domain_preferences:
            self.domain_preferences[domain] = []
        
        self.domain_preferences[domain].append(quality_score)
        
        # Keep only recent scores (last 10)
        if len(self.domain_preferences[domain]) > 10:
            self.domain_preferences[domain] = self.domain_preferences[domain][-10:]
    
    def _get_top_domains(self) -> List[str]:
        """Get top performing domains"""
        domain_scores = {}
        for domain, scores in self.domain_preferences.items():
            if scores:
                domain_scores[domain] = sum(scores) / len(scores)
        
        sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1], reverse=True)
        return [domain for domain, score in sorted_domains[:5]]


class AutonomousAgricultureCurator:
    """Main curator with 12+ autonomous agents for comprehensive Indian agriculture data collection"""
    
    def __init__(self, 
                 num_agents: int = 12,
                 output_file: str = "autonomous_indian_agriculture.jsonl",
                 max_search_results: int = 30,
                 pdf_storage_dir: str = "autonomous_pdfs",
                 enable_pdf_download: bool = True,
                 searches_per_agent: int = 50):
        
        self.num_agents = num_agents
        self.output_file = output_file
        self.max_search_results = max_search_results
        self.enable_pdf_download = enable_pdf_download
        self.searches_per_agent = searches_per_agent
        
        # Advanced monitoring and analytics
        self.start_timestamp = datetime.now()
        self.analytics = {
            'total_searches': 0,
            'successful_searches': 0,
            'failed_searches': 0,
            'total_urls_processed': 0,
            'total_pdfs_processed': 0,
            'average_quality_score': 0.0,
            'domain_distribution': {},
            'content_type_distribution': {},
            'state_coverage': {},
            'crop_coverage': {},
            'technology_coverage': {}
        }
        
        # Initialize IMMEDIATE JSONL writer with enhanced features
        self.jsonl_writer = ImmediateJSONLWriter(output_file)
        
        # Initialize PDF processor with comprehensive capabilities
        self.pdf_processor = ImprovedPDFProcessor(
            storage_dir=pdf_storage_dir, 
            jsonl_writer=self.jsonl_writer
        ) if enable_pdf_download else None
        
        # Initialize search engine with all advanced features
        self.search_engine = ImprovedWebSearch(max_search_results, self.pdf_processor, self.jsonl_writer)
        
        # Define comprehensive agent specializations (expanded)
        self.agent_specializations = [
            "Crop Science & Plant Breeding",
            "Soil Science & Fertility Management", 
            "Water Resources & Irrigation",
            "Plant Protection & Pest Management",
            "Agricultural Technology & Precision Farming",
            "Sustainable & Organic Farming",
            "Agricultural Economics & Policy",
            "Climate Change & Adaptation",
            "Horticulture & Plantation Crops",
            "Livestock & Animal Husbandry",
            "Food Processing & Post-Harvest",
            "Rural Development & Extension",
            "Agricultural Research & Innovation",
            "Traditional Knowledge & Indigenous Practices",
            "Agricultural Biotechnology & Genetics",
            "Farm Mechanization & Engineering",
            "Agribusiness & Supply Chain",
            "Agricultural Statistics & Data Science",
            "Agroforestry & Silviculture",
            "Fisheries & Aquaculture"
        ]
        
        self.agents = []
        
        # Setup comprehensive logging with analytics
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - [%(funcName)s] %(message)s',
            handlers=[
                logging.FileHandler('autonomous_agriculture_curator_advanced.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create directories with proper structure
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        if enable_pdf_download:
            Path(pdf_storage_dir).mkdir(parents=True, exist_ok=True)
            
        # Create analytics directory
        Path("analytics").mkdir(exist_ok=True)
        
        logging.info(f"🚀 Initialized Advanced Autonomous Agriculture Curator")
        logging.info(f"📊 Configuration: {num_agents} agents, {searches_per_agent} searches each")
        logging.info(f"🎯 Target: {num_agents * searches_per_agent} total autonomous searches")
    
    def start_autonomous_curation(self) -> Dict:
        """Start advanced autonomous agriculture data curation with comprehensive monitoring"""
        logging.info("🚀 Starting ADVANCED AUTONOMOUS Indian Agriculture Data Curation")
        logging.info(f"🤖 Deploying {self.num_agents} intelligent autonomous agents")
        logging.info(f"🔍 Each agent will perform {self.searches_per_agent} advanced autonomous searches")
        logging.info(f"📝 Data will be immediately written to: {self.output_file}")
        logging.info(f"📊 Advanced analytics and monitoring enabled")
        
        # Initialize global persistent duplicate tracker for cross-agent deduplication
        self.global_duplicate_tracker = get_global_tracker()
        logging.info(f"🔒 Global duplicate tracker initialized")
        
        # Initialize autonomous agents with specializations and shared duplicate tracker
        for i in range(self.num_agents):
            specialization = self.agent_specializations[i % len(self.agent_specializations)]
            agent = AutonomousSearchAgent(i, specialization, self.search_engine, 
                                         self.jsonl_writer, self.global_duplicate_tracker)
            self.agents.append(agent)
            logging.info(f"🤖 Agent {i}: {specialization}")
        
        # Execute autonomous curation with advanced monitoring
        logging.info("🔄 Starting parallel autonomous data curation with real-time analytics...")
        start_time = time.time()
        
        # Create analytics tracking thread
        analytics_thread = threading.Thread(target=self._monitor_progress, daemon=True)
        analytics_thread.start()
        
        with ThreadPoolExecutor(max_workers=min(self.num_agents, 6)) as executor:  # Optimal concurrency
            future_to_agent = {
                executor.submit(agent.autonomous_search_and_curate, self.searches_per_agent): agent 
                for agent in self.agents
            }
            
            results = []
            completed_agents = 0
            
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed_agents += 1
                    
                    # Update analytics
                    self._update_analytics(agent, result)
                    
                    logging.info(f"✅ Agent {agent.agent_id} ({agent.specialization}) completed!")
                    logging.info(f"📊 Progress: {completed_agents}/{self.num_agents} agents completed")
                    logging.info(f"📝 Agent {agent.agent_id} collected: {result.get('processed_count', 0)} entries")
                    logging.info(f"🎯 Agent {agent.agent_id} quality: {self._calculate_agent_quality(agent):.3f}")
                    
                    # Real-time analytics update
                    self._log_current_analytics()
                    
                except Exception as e:
                    logging.error(f"❌ Agent {agent.agent_id} failed: {e}")
                    completed_agents += 1
        
        execution_time = time.time() - start_time
        
        # Generate comprehensive analytics and summary
        summary = self._generate_comprehensive_summary(results, execution_time)
        
        # Save detailed analytics
        self._save_detailed_analytics(summary)
        
        logging.info("🎉 ADVANCED AUTONOMOUS CURATION COMPLETED!")
        logging.info(f"⏱️ Total execution time: {execution_time:.2f} seconds")
        logging.info(f"📊 Total unique entries collected: {summary['total_entries']}")
        logging.info(f"📁 Output file: {self.output_file}")
        logging.info(f"🧠 Total autonomous searches: {summary['total_autonomous_searches']}")
        logging.info(f"🏆 Average quality score: {summary['average_quality_score']:.3f}")
        
        return summary
    
    def _monitor_progress(self):
        """Monitor progress in real-time"""
        while True:
            time.sleep(30)  # Update every 30 seconds
            
            # Check if all agents are still running
            active_agents = sum(1 for agent in self.agents if hasattr(agent, 'processed_urls'))
            if active_agents == 0:
                break
                
            # Log progress summary
            total_processed = sum(len(agent.processed_urls) for agent in self.agents if hasattr(agent, 'processed_urls'))
            total_entries = self.jsonl_writer.get_entries_count()
            
            logging.info(f"📊 Real-time update: {total_entries} entries saved, {total_processed} URLs processed")
    
    def _update_analytics(self, agent: AutonomousSearchAgent, result: CurationResult):
        """Update comprehensive analytics"""
        if result.success:
            self.analytics['successful_searches'] += len(agent.search_history)
            self.analytics['total_urls_processed'] += len(agent.processed_urls)
            self.analytics['total_pdfs_processed'] += result.pdfs_processed
            
            # Update domain distribution
            for domain, scores in agent.domain_preferences.items():
                if domain not in self.analytics['domain_distribution']:
                    self.analytics['domain_distribution'][domain] = 0
                self.analytics['domain_distribution'][domain] += len(scores)
        else:
            self.analytics['failed_searches'] += self.searches_per_agent
        
        self.analytics['total_searches'] += len(agent.search_history)
    
    def _calculate_agent_quality(self, agent: AutonomousSearchAgent) -> float:
        """Calculate agent quality score"""
        if not hasattr(agent, 'content_quality_scores') or not agent.content_quality_scores:
            return 0.0
        return sum(agent.content_quality_scores) / len(agent.content_quality_scores)
    
    def _log_current_analytics(self):
        """Log current analytics state"""
        total_entries = self.jsonl_writer.get_entries_count()
        success_rate = (self.analytics['successful_searches'] / max(self.analytics['total_searches'], 1)) * 100
        
        logging.info(f"📈 Current analytics: {total_entries} entries, {success_rate:.1f}% search success rate")
    
    def _generate_comprehensive_summary(self, results: List[CurationResult], execution_time: float) -> Dict:
        """Generate comprehensive summary with advanced analytics"""
        successful_results = [r for r in results if r.success]
        
        total_entries = sum(r.processed_count for r in successful_results)
        total_pdfs = sum(r.pdfs_processed for r in successful_results)
        total_searches = sum(len(agent.search_history) for agent in self.agents)
        total_unique_urls = sum(len(agent.processed_urls) for agent in self.agents)
        
        # Calculate average quality score
        all_quality_scores = []
        for agent in self.agents:
            if hasattr(agent, 'content_quality_scores') and agent.content_quality_scores:
                all_quality_scores.extend(agent.content_quality_scores)
        
        average_quality_score = sum(all_quality_scores) / len(all_quality_scores) if all_quality_scores else 0.0
        
        # Get actual count from JSONL writer
        actual_entries_written = self.jsonl_writer.get_entries_count()
        
        # Agent performance analysis with quality metrics
        agent_performance = []
        for result in successful_results:
            agent_info = next((agent for agent in self.agents if agent.agent_id == result.agent_id), None)
            if agent_info:
                agent_performance.append({
                    'agent_id': result.agent_id,
                    'specialization': agent_info.specialization,
                    'entries_collected': result.processed_count,
                    'pdfs_processed': result.pdfs_processed,
                    'autonomous_searches': len(agent_info.search_history),
                    'unique_urls': len(agent_info.processed_urls),
                    'execution_time': result.execution_time,
                    'quality_score': self._calculate_agent_quality(agent_info),
                    'success_patterns': len(agent_info.success_patterns),
                    'failure_patterns': len(agent_info.failure_patterns),
                    'top_domains': agent_info._get_top_domains()
                })
        
        return {
            "success": True,
            "autonomous_curation": True,
            "execution_time": execution_time,
            "total_entries": total_entries,
            "actual_entries_written": actual_entries_written,
            "total_pdfs_processed": total_pdfs,
            "total_autonomous_searches": total_searches,
            "total_unique_urls": total_unique_urls,
            "average_quality_score": average_quality_score,
            "successful_agents": len(successful_results),
            "failed_agents": len(results) - len(successful_results),
            "agents_deployed": self.num_agents,
            "searches_per_agent": self.searches_per_agent,
            "output_file": self.output_file,
            "pdf_download_enabled": self.enable_pdf_download,
            "agent_performance": agent_performance,
            "coverage_areas": [agent.specialization for agent in self.agents],
            "analytics": self.analytics
        }
    
    def _save_detailed_analytics(self, summary: Dict):
        """Save detailed analytics to file"""
        analytics_file = f"analytics/curation_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        detailed_analytics = {
            'summary': summary,
            'system_analytics': self.analytics,
            'agent_performance': [
                {
                    'agent_id': agent.agent_id,
                    'specialization': agent.specialization,
                    'searches_performed': len(agent.search_history),
                    'urls_processed': len(agent.processed_urls),
                    'success_patterns': len(agent.success_patterns),
                    'failure_patterns': len(agent.failure_patterns),
                    'top_domains': agent._get_top_domains(),
                    'quality_score': self._calculate_agent_quality(agent)
                }
                for agent in self.agents
            ],
            'configuration': {
                'num_agents': self.num_agents,
                'searches_per_agent': self.searches_per_agent,
                'max_search_results': self.max_search_results,
                'pdf_download_enabled': self.enable_pdf_download
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_analytics, f, indent=2, ensure_ascii=False)
        
        logging.info(f"📊 Detailed analytics saved to: {analytics_file}")
    
    def _generate_autonomous_summary(self, results: List[CurationResult], execution_time: float) -> Dict:
        """Generate comprehensive summary of autonomous curation"""
        successful_results = [r for r in results if r.success]
        
        total_entries = sum(r.processed_count for r in successful_results)
        total_pdfs = sum(r.pdfs_processed for r in successful_results)
        total_searches = sum(len(agent.search_history) for agent in self.agents)
        total_unique_urls = sum(len(agent.processed_urls) for agent in self.agents)
        
        # Get actual count from JSONL writer
        actual_entries_written = self.jsonl_writer.get_entries_count()
        
        # Agent performance analysis
        agent_performance = []
        for result in successful_results:
            agent_info = next((agent for agent in self.agents if agent.agent_id == result.agent_id), None)
            if agent_info:
                agent_performance.append({
                    'agent_id': result.agent_id,
                    'specialization': agent_info.specialization,
                    'entries_collected': result.processed_count,
                    'pdfs_processed': result.pdfs_processed,
                    'autonomous_searches': len(agent_info.search_history),
                    'unique_urls': len(agent_info.processed_urls),
                    'execution_time': result.execution_time
                })
        
        return {
            "success": True,
            "autonomous_curation": True,
            "execution_time": execution_time,
            "total_entries": total_entries,
            "actual_entries_written": actual_entries_written,
            "total_pdfs_processed": total_pdfs,
            "total_autonomous_searches": total_searches,
            "total_unique_urls": total_unique_urls,
            "successful_agents": len(successful_results),
            "failed_agents": len(results) - len(successful_results),
            "agents_deployed": self.num_agents,
            "searches_per_agent": self.searches_per_agent,
            "output_file": self.output_file,
            "pdf_download_enabled": self.enable_pdf_download,
            "agent_performance": agent_performance,
            "coverage_areas": [agent.specialization for agent in self.agents]
        }


def main():
    """Main function to run autonomous agriculture data curation"""
    config = {
        "num_agents": 12,  # Deploy 12 autonomous agents
        "output_file": "autonomous_indian_agriculture_complete.jsonl",
        "max_search_results": 25,
        "pdf_storage_dir": "autonomous_agriculture_pdfs",
        "enable_pdf_download": True,
        "searches_per_agent": 50  # Each agent performs 50 autonomous searches
    }
    
    print("🌾 AUTONOMOUS INDIAN AGRICULTURE DATA CURATOR 🌾")
    print("=" * 60)
    print(f"🤖 Deploying {config['num_agents']} intelligent agents")
    print(f"🔍 Total autonomous searches: {config['num_agents'] * config['searches_per_agent']}")
    print(f"📊 Expected data coverage: Comprehensive Indian Agriculture")
    print("=" * 60)
    
    # Create and run autonomous curator
    curator = AutonomousAgricultureCurator(
        num_agents=config["num_agents"],
        output_file=config["output_file"],
        max_search_results=config["max_search_results"],
        pdf_storage_dir=config["pdf_storage_dir"],
        enable_pdf_download=config["enable_pdf_download"],
        searches_per_agent=config["searches_per_agent"]
    )
    
    try:
        summary = curator.start_autonomous_curation()
        
        print("\n" + "="*80)
        print("🎉 AUTONOMOUS INDIAN AGRICULTURE CURATION COMPLETED 🎉")
        print("="*80)
        print(f"✅ Total entries collected: {summary.get('total_entries', 0)}")
        print(f"📝 Actual entries written: {summary.get('actual_entries_written', 0)}")
        print(f"📄 PDFs processed: {summary.get('total_pdfs_processed', 0)}")
        print(f"🔍 Autonomous searches: {summary.get('total_autonomous_searches', 0)}")
        print(f"🌐 Unique URLs processed: {summary.get('total_unique_urls', 0)}")
        print(f"⏱️ Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"📁 Output file: {summary.get('output_file', 'N/A')}")
        print(f"🤖 Successful agents: {summary.get('successful_agents', 0)}/{summary.get('agents_deployed', 0)}")
        
        print("\n📊 AGENT PERFORMANCE:")
        for agent_perf in summary.get('agent_performance', []):
            print(f"  Agent {agent_perf['agent_id']}: {agent_perf['specialization']}")
            print(f"    📝 Entries: {agent_perf['entries_collected']}")
            print(f"    🔍 Searches: {agent_perf['autonomous_searches']}")
            print(f"    🌐 URLs: {agent_perf['unique_urls']}")
        
        print("\n🎯 COVERAGE AREAS:")
        for area in summary.get('coverage_areas', []):
            print(f"  ✓ {area}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Autonomous curation interrupted by user")
    except Exception as e:
        print(f"❌ Autonomous curation failed: {e}")
        logging.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    main()
