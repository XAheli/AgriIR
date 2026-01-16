#!/usr/bin/env python3
"""
Enhanced RAG System with Web Search Integration
Combines embeddings-based retrieval with web search for comprehensive answers
"""

import os
import json
import numpy as np
import warnings
# Suppress FAISS warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='faiss')
import faiss
import pickle
import requests
import logging
import asyncio
import tempfile
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Import DuckDuckGo search
try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False


def check_offline_model(model_name: str) -> Optional[str]:
    """
    Check if a HuggingFace model exists offline and return the local path
    """
    try:
        import os
        from pathlib import Path
        
        # Check HuggingFace cache directory
        hf_cache_dir = os.path.expanduser("~/.cache/huggingface/hub/")
        
        # Convert model name to cache directory format
        cache_model_name = f"models--{model_name.replace('/', '--')}"
        model_cache_path = os.path.join(hf_cache_dir, cache_model_name)
        
        if os.path.exists(model_cache_path):
            print(f"✅ Found offline model: {model_name} at {model_cache_path}")
            return model_name  # Return original name for SentenceTransformer
        else:
            print(f"❌ Offline model not found: {model_name}")
            return None
            
    except Exception as e:
        print(f"⚠️ Error checking offline model {model_name}: {e}")
        return None


def get_available_embedding_models() -> List[str]:
    """
    Get list of available embedding models (offline first, then fallbacks)
    """
    # Preferred models in order of preference
    preferred_models = [
        "Qwen/Qwen3-Embedding-8B",
        "Qwen/Qwen3-Embedding-4B", 
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-small-en-v1.5"
    ]
    
    available_models = []
    
    # Check which models are available offline
    for model in preferred_models:
        if check_offline_model(model):
            available_models.append(model)
    
    # If no offline models found, add fallback models that can be downloaded
    if not available_models:
        available_models = ["sentence-transformers/all-MiniLM-L6-v2"]
        print("⚠️ No offline embedding models found, will attempt to download fallback model")
    
    return available_models


@dataclass
class Citation:
    """Structured citation information"""
    citation_id: str  # e.g., "DB-1-1" or "WEB-2-3"
    citation_type: str  # "database" or "web"
    title: str
    url: str
    source: str
    content_snippet: str
    full_content: str
    relevance_score: float
    timestamp: str
    sub_query_idx: int
    result_idx: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_markdown(self) -> str:
        """Convert citation to markdown format"""
        return f"[{self.citation_id}] {self.title} - {self.url}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'citation_id': self.citation_id,
            'citation_type': self.citation_type,
            'title': self.title,
            'url': self.url,
            'source': self.source,
            'content_snippet': self.content_snippet,
            'relevance_score': self.relevance_score,
            'timestamp': self.timestamp,
            'metadata': self.metadata
        }


@dataclass
class SearchResult:
    """Web search result with metadata"""
    title: str
    url: str
    snippet: str
    content: Optional[str] = None
    source_type: str = "web"
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    relevance_score: float = 0.0
    publication_date: Optional[str] = None
    author: Optional[str] = None
    citation: Optional[Citation] = None


@dataclass
class DatabaseChunk:
    """Database chunk result with metadata"""
    chunk_text: str
    source: str
    title: str = ""
    chunk_id: str = ""
    source_domain: str = ""
    similarity_score: float = 0.0
    source_type: str = "database"
    metadata: Dict[str, Any] = field(default_factory=dict)
    citation: Optional[Citation] = None


@dataclass
class SubQueryResult:
    """Result from sub-query processing"""
    original_query: str
    sub_queries: List[str]
    web_results: List[SearchResult] = field(default_factory=list)
    db_results: List[DatabaseChunk] = field(default_factory=list)
    markdown_content: str = ""
    agent_info: Dict[str, Any] = field(default_factory=dict)
    citations: List[Citation] = field(default_factory=list)  # Add explicit citations


class CitationManager:
    """Manages citations and provides structured citation tracking"""
    
    def __init__(self):
        self.citations = []
        self.citation_map = {}  # Maps content hash to citation ID
        self.logger = logging.getLogger(__name__)
    
    def create_citation(self, 
                       result: Union[DatabaseChunk, SearchResult],
                       sub_query_idx: int,
                       result_idx: int,
                       citation_type: str) -> Citation:
        """Create a structured citation from a result"""
        
        citation_id = f"{citation_type}-{sub_query_idx + 1}-{result_idx + 1}"
        
        if isinstance(result, DatabaseChunk):
            content_snippet = result.chunk_text[:200] + "..." if len(result.chunk_text) > 200 else result.chunk_text
            citation = Citation(
                citation_id=citation_id,
                citation_type="database",
                title=result.title or result.source,
                url=result.source,
                source=result.source_domain or result.source,
                content_snippet=content_snippet,
                full_content=result.chunk_text,
                relevance_score=result.similarity_score,
                timestamp=datetime.now().isoformat(),
                sub_query_idx=sub_query_idx,
                result_idx=result_idx,
                metadata={
                    'chunk_id': result.chunk_id,
                    'source_domain': result.source_domain
                }
            )
        else:  # SearchResult
            content_snippet = result.snippet[:200] + "..." if len(result.snippet) > 200 else result.snippet
            citation = Citation(
                citation_id=citation_id,
                citation_type="web",
                title=result.title,
                url=result.url,
                source=result.url,
                content_snippet=content_snippet,
                full_content=result.content or result.snippet,
                relevance_score=result.relevance_score,
                timestamp=datetime.now().isoformat(),
                sub_query_idx=sub_query_idx,
                result_idx=result_idx,
                metadata={
                    'publication_date': getattr(result, 'publication_date', None),
                    'author': getattr(result, 'author', None)
                }
            )
        
        # Store citation
        self.citations.append(citation)
        
        # Create content hash for deduplication
        content_hash = hashlib.md5(citation.full_content.encode()).hexdigest()
        self.citation_map[content_hash] = citation_id
        
        return citation
    
    def get_citation_by_id(self, citation_id: str) -> Optional[Citation]:
        """Retrieve citation by ID"""
        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation
        return None
    
    def get_all_citations(self) -> List[Citation]:
        """Get all citations"""
        return self.citations
    
    def generate_citation_index(self) -> str:
        """Generate formatted citation index"""
        citation_index = "## 📖 Citation Index\n\n"
        
        for i, citation in enumerate(self.citations, 1):
            citation_index += f"{i}. {citation.to_markdown()}\n"
            if citation.metadata.get('publication_date'):
                citation_index += f"   Published: {citation.metadata['publication_date']}\n"
            if citation.metadata.get('author'):
                citation_index += f"   Author: {citation.metadata['author']}\n"
            citation_index += f"   Relevance: {citation.relevance_score:.2f}\n\n"
        
        return citation_index
    
    def extract_citations_from_text(self, text: str) -> List[str]:
        """Extract citation IDs from text"""
        import re
        citation_pattern = r'\[(DB|WEB)-(\d+)-(\d+)\]'
        matches = re.findall(citation_pattern, text)
        return [f"{m[0]}-{m[1]}-{m[2]}" for m in matches]
    
    def validate_citations_in_text(self, text: str) -> Tuple[List[str], List[str]]:
        """Validate citations in text and return valid and invalid citations"""
        extracted = self.extract_citations_from_text(text)
        valid_ids = {c.citation_id for c in self.citations}
        
        valid = [cid for cid in extracted if cid in valid_ids]
        invalid = [cid for cid in extracted if cid not in valid_ids]
        
        return valid, invalid


class QueryRefiner:
    """Refines user queries using Gemma3:1b model"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model = "gemma3:1b"
        self.logger = logging.getLogger(__name__)
    
    def refine_query(self, query: str) -> str:
        """Refine a crude user query to be more specific and searchable"""
        
        prompt = f"""You are an agricultural AI assistant. Transform the user's query into a more specific and searchable agricultural query while preserving their intent.

Rules:
1. Understand the context and agricultural issue being described
2. Make reasonable assumptions about common agricultural scenarios
3. Transform it into a clear, searchable query
4. Keep the original intent and urgency
5. Add relevant agricultural keywords
6. Return ONLY the refined query, no explanations

Examples:
- "My crops are dying" → "crop disease symptoms identification and treatment for dying plants"
- "Rain damaged my field" → "monsoon crop damage assessment and government compensation schemes in India"
- "Need money for farming" → "agricultural loans subsidies and financial assistance programs for farmers"

User query: {query}

Refined query:"""

        try:
            response = requests.post(
                f'{self.ollama_host}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.1,
                        'top_p': 0.9,
                        'num_ctx': 2048
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                refined = response.json()['response'].strip()
                self.logger.info(f"Query refined: '{query}' -> '{refined}'")
                return refined
            else:
                self.logger.warning(f"Query refinement failed, using original: {query}")
                return query
                
        except Exception as e:
            self.logger.error(f"Error refining query: {e}")
            return query


class SubQueryGenerator:
    """Generates sub-queries using Gemma3:1b model"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.model = "gemma3:1b"
        self.logger = logging.getLogger(__name__)
    
    def generate_sub_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple sub-queries for comprehensive search"""
        
        prompt = f"""You are an agricultural research assistant. Break down the user's question into {num_queries} CONCRETE search queries.

CRITICAL RULES:
1. Use ONLY information from the user's question - DO NOT invent regions, locations, or specific varieties not mentioned
2. Generate ACTUAL search queries, NOT templates with placeholders like [crop type] or [region]
3. If the user asks a general question, keep sub-queries general
4. Each query should cover a different aspect: symptoms/identification, treatment/control, prevention/management
5. DO NOT add invented details like "Midwest", "California", "winter wheat" if not in original query

USER'S QUESTION: {query}

EXAMPLES:
Question: "What crops suffer from rust disease?"
Good sub-queries:
1. agricultural crops affected by rust disease symptoms
2. rust disease control methods cereal crops
3. rust fungal infection crop management practices

Bad sub-queries (inventing details):
❌ wheat rust disease in Midwest farmers
❌ rust treatment for California orchards

Question: "How to control aphids in tomato?"
Good sub-queries:
1. tomato aphid control methods organic
2. tomato aphid infestation symptoms identification
3. biological pest control aphids tomato plants

Now generate {num_queries} CONCRETE search queries using ONLY information from the user's question:

1."""

        try:
            response = requests.post(
                f'{self.ollama_host}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.3,  # Lower to reduce hallucination while maintaining variation
                        'top_p': 0.85,
                        'top_k': 40,
                        'repeat_penalty': 1.1,
                        'num_ctx': 2048,
                        'num_predict': 256,  # Limit to prevent long explanations
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                response_text = response.json()['response'].strip()
                
                # Parse sub-queries from numbered list
                sub_queries = []
                lines = response_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('•') or line.startswith('-')):
                        # Remove numbering/bullets and clean up
                        clean_query = line.split('.', 1)[-1].strip() if '.' in line else line
                        clean_query = clean_query.lstrip('•-').strip()
                        # Remove quotes and extra formatting
                        clean_query = clean_query.strip('"').strip("'").strip('*').strip()
                        
                        # FILTER OUT PLACEHOLDER QUERIES
                        if clean_query and len(clean_query) > 10:
                            # Skip if contains placeholders
                            if '[' in clean_query and ']' in clean_query:
                                self.logger.warning(f"Skipping placeholder query: {clean_query}")
                                continue
                            if not clean_query.lower().startswith(('here are', 'okay', 'the following', 'sure', 'i can')):
                                sub_queries.append(clean_query)
                
                if not sub_queries:
                    # Fallback: split by lines and take meaningful ones (skip placeholders)
                    for line in lines:
                        line = line.strip()
                        if line and len(line) > 15:
                            if '[' in line and ']' in line:
                                continue  # Skip placeholders
                            if not line.lower().startswith(('here are', 'okay', 'the following', 'sure', 'i can')):
                                clean_line = line.strip('"').strip("'").strip('*').strip()
                                sub_queries.append(clean_line)
                
                # If still no valid queries, create fallback queries based on original
                if not sub_queries:
                    self.logger.warning(f"No valid sub-queries generated, creating fallback queries")
                    # Create generic but valid queries from the original
                    sub_queries = [
                        query,  # Original query
                        f"{query} symptoms identification treatment",  # Symptom-focused
                        f"{query} management control practices India"  # Management-focused
                    ]
                
                # Limit to requested number
                sub_queries = sub_queries[:num_queries]
                
                self.logger.info(f"Generated {len(sub_queries)} sub-queries for: {query}")
                return sub_queries
                
            else:
                self.logger.warning(f"Sub-query generation failed, using original: {query}")
                return [query]
                
        except Exception as e:
            self.logger.error(f"Error generating sub-queries: {e}")
            return [query]


class DatabaseRetriever:
    """Retrieves chunks from the embeddings database"""
    
    def __init__(self, embeddings_dir: str, model_name: str = "Qwen/Qwen3-Embedding-8B"):
        self.embeddings_dir = embeddings_dir
        self.original_model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers is required for database retrieval")
        
        # Determine device (GPU if available) with memory check
        self.device = 'cuda' if HAS_TORCH and torch.cuda.is_available() and self._check_gpu_memory() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Find the best available embedding model
        self.model_name = self._find_best_embedding_model(model_name)
        self.logger.info(f"Selected embedding model: {self.model_name}")
        
        # Load embedding model with offline-first approach
        self.embedding_model = self._load_embedding_model()
        
        # Load pre-computed embeddings
        self.load_embeddings()
    
    def _find_best_embedding_model(self, preferred_model: str) -> str:
        """Find the best available embedding model, preferring offline models"""
        
        self.logger.info(f"Looking for embedding model: {preferred_model}")
        
        # First, check if the preferred model is available offline
        if check_offline_model(preferred_model):
            self.logger.info(f"✅ Found preferred model offline: {preferred_model}")
            return preferred_model
        
        # If preferred model not available, check other available models
        self.logger.warning(f"❌ Preferred model {preferred_model} not found offline")
        self.logger.info("🔍 Checking for alternative offline models...")
        
        available_models = get_available_embedding_models()
        
        if available_models:
            selected_model = available_models[0]
            self.logger.info(f"✅ Selected alternative model: {selected_model}")
            return selected_model
        else:
            # Fallback to a small model that can be downloaded
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.logger.warning(f"⚠️ No offline models found, will try to download: {fallback_model}")
            return fallback_model
    
    def _load_embedding_model(self) -> SentenceTransformer:
        """Load the embedding model with proper error handling"""
        
        try:
            self.logger.info(f"🔄 Loading embedding model: {self.model_name}")
            
            # First try to load with local_files_only=True for offline models
            try:
                model = SentenceTransformer(self.model_name, device=self.device, local_files_only=True)
                self.logger.info(f"✅ Successfully loaded model offline: {self.model_name}")
                return model
            except Exception as offline_error:
                self.logger.info(f"⚠️ Failed to load offline, trying online: {offline_error}")
                # Try without local_files_only (will download if needed)
                model = SentenceTransformer(self.model_name, device=self.device)
                self.logger.info(f"✅ Successfully loaded model online: {self.model_name}")
                return model
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load embedding model {self.model_name}: {e}")
            
            # Try fallback models
            fallback_models = ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"]
            
            for fallback_model in fallback_models:
                try:
                    self.logger.info(f"🔄 Trying fallback model: {fallback_model}")
                    
                    # Try offline first
                    try:
                        model = SentenceTransformer(fallback_model, device=self.device, local_files_only=True)
                        self.logger.info(f"✅ Successfully loaded fallback model offline: {fallback_model}")
                        self.model_name = fallback_model  # Update the model name
                        return model
                    except Exception:
                        # Try online if offline fails
                        model = SentenceTransformer(fallback_model, device=self.device)
                        self.logger.info(f"✅ Successfully loaded fallback model online: {fallback_model}")
                        self.model_name = fallback_model  # Update the model name
                        return model
                    
                except Exception as fallback_error:
                    self.logger.warning(f"❌ Failed to load fallback model {fallback_model}: {fallback_error}")
                    continue
            
            # If all models fail, raise the original error
            raise RuntimeError(f"Failed to load both {self.model_name} and fallback models: {str(e)}")
        
        # Load pre-computed embeddings
        self.load_embeddings()
    
    def _check_gpu_memory(self) -> bool:
        """Check if GPU has sufficient memory for embeddings"""
        try:
            if HAS_TORCH and torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                # Require at least 2GB for embedding operations
                return gpu_memory > 2 * 1024**3
            return False
        except Exception:
            return False
    
    def load_embeddings(self):
        """Load FAISS index and metadata with improved error handling and optimization"""
        try:
            # Load FAISS index
            possible_index_names = ["faiss_index.bin", "faiss_index.index"]
            index_path = None
            
            for index_name in possible_index_names:
                potential_path = os.path.join(self.embeddings_dir, index_name)
                if os.path.exists(potential_path):
                    index_path = potential_path
                    break
            
            if not index_path:
                raise FileNotFoundError(f"FAISS index not found in {self.embeddings_dir}")
            
            self.logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            
            # Try to move index to GPU if available and beneficial
            if self.device == 'cuda' and self.index.ntotal > 1000:  # Only for larger indexes
                try:
                    import torch
                    if HAS_TORCH and torch.cuda.is_available() and hasattr(faiss, 'StandardGpuResources'):
                        gpu_id = 0
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, gpu_id, self.index)
                        self.logger.info(f"Moved FAISS index to GPU for faster search")
                    else:
                        self.logger.info("FAISS-GPU not available, using CPU version")
                except Exception as e:
                    self.logger.warning(f"Failed to move index to GPU: {e}")
            
            self.logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load metadata with improved handling for large files
            metadata_pkl_path = os.path.join(self.embeddings_dir, "metadata.pkl")
            metadata_json_path = os.path.join(self.embeddings_dir, "metadata.json")
            
            # Try pickle first (much faster)
            if os.path.exists(metadata_pkl_path):
                try:
                    self.logger.info("Loading metadata from pickle file...")
                    with open(metadata_pkl_path, 'rb') as f:
                        self.metadata = pickle.load(f)
                    self.logger.info(f"Successfully loaded metadata for {len(self.metadata)} chunks from pickle")
                except Exception as pickle_error:
                    self.logger.warning(f"Failed to load pickle metadata: {pickle_error}")
                    self.metadata = None
            
            # Fallback to JSON if pickle failed or doesn't exist
            if self.metadata is None and os.path.exists(metadata_json_path):
                try:
                    # Check file size first
                    file_size = os.path.getsize(metadata_json_path)
                    file_size_gb = file_size / (1024**3)
                    
                    if file_size_gb > 1.0:
                        self.logger.warning(f"JSON file is large ({file_size_gb:.1f}GB), this may take several minutes...")
                        self.logger.warning(f"JSON file is {file_size_gb:.1f}GB, this will take a while...")
                        self.logger.warning("Consider using the pickle file instead for faster loading")
                    
                    self.logger.info("Loading metadata from JSON file...")
                    with open(metadata_json_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                    self.logger.info(f"Successfully loaded metadata for {len(self.metadata)} chunks from JSON")
                    
                    # Create pickle file for faster future loading
                    self.logger.info("Creating pickle file for faster future loading...")
                    try:
                        with open(metadata_pkl_path, 'wb') as f:
                            pickle.dump(self.metadata, f)
                        self.logger.info("Created pickle file for faster loading next time")
                    except Exception as pickle_save_error:
                        self.logger.warning(f"Failed to save pickle file: {pickle_save_error}")
                        
                except Exception as json_error:
                    self.logger.error(f"Failed to load JSON metadata: {json_error}")
                    raise FileNotFoundError("Could not load metadata from either pickle or JSON files")
            
            if self.metadata is None:
                raise FileNotFoundError("No valid metadata file found")
            
            # Validate metadata compatibility with FAISS index
            if len(self.metadata) != self.index.ntotal:
                # Suppress warning - using the smaller count for safety
                self.logger.debug(f"Metadata count ({len(self.metadata)}) doesn't match FAISS index count ({self.index.ntotal})")
                self.logger.debug(f"Using the smaller count for safety: min({len(self.metadata)}, {self.index.ntotal}) = {min(len(self.metadata), self.index.ntotal)}")
                # Adjust search parameters to work with available data
                self.max_safe_index = min(len(self.metadata), self.index.ntotal) - 1
            
            self.logger.info(f"Successfully loaded embeddings system with {self.index.ntotal} vectors and {len(self.metadata)} metadata entries")
            
            # Set max_safe_index if not already set
            if not hasattr(self, 'max_safe_index'):
                self.max_safe_index = min(len(self.metadata), self.index.ntotal) - 1
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            raise
    
    def retrieve_chunks(self, query: str, top_k: int = 5) -> List[DatabaseChunk]:
        """Retrieve top-k most relevant chunks for the query with improved error handling"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                self.logger.error("FAISS index not loaded")
                return []
            
            if not hasattr(self, 'metadata') or self.metadata is None:
                self.logger.error("Metadata not loaded")
                return []
            
            # Embed the query
            query_embedding = self.embedding_model.encode([query]).astype('float32')
            
            # Normalize for cosine similarity if the index expects it
            import faiss
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            try:
                # Adjust top_k to not exceed available data
                safe_top_k = min(top_k, self.index.ntotal)
                if hasattr(self, 'max_safe_index'):
                    safe_top_k = min(safe_top_k, self.max_safe_index + 1)
                
                distances, indices = self.index.search(query_embedding, safe_top_k)
            except Exception as search_error:
                self.logger.error(f"Error searching FAISS index: {search_error}")
                return []
            
            # Prepare results
            results = []
            for distance, idx in zip(distances[0], indices[0]):
                try:
                    # Enhanced bounds checking
                    if idx < 0:
                        self.logger.warning(f"Negative index {idx}, skipping")
                        continue
                    
                    if idx >= len(self.metadata):
                        self.logger.warning(f"Index {idx} exceeds metadata bounds ({len(self.metadata)}), skipping")
                        continue
                    
                    # Additional safety check for max_safe_index
                    if hasattr(self, 'max_safe_index') and idx > self.max_safe_index:
                        self.logger.warning(f"Index {idx} exceeds safe bounds ({self.max_safe_index}), skipping")
                        continue
                        
                    chunk_data = self.metadata[idx]
                    
                    # Handle both dict and object metadata formats
                    if isinstance(chunk_data, dict):
                        chunk_text = chunk_data.get('chunk_text', '')
                        source = chunk_data.get('link', chunk_data.get('source', ''))
                        title = chunk_data.get('title', '')
                        chunk_id = chunk_data.get('chunk_id', str(idx))
                        source_domain = chunk_data.get('source_domain', '')
                    else:
                        # Handle object format (from pickle)
                        chunk_text = getattr(chunk_data, 'chunk_text', '')
                        source = getattr(chunk_data, 'link', getattr(chunk_data, 'source', ''))
                        title = getattr(chunk_data, 'title', '')
                        chunk_id = getattr(chunk_data, 'chunk_id', str(idx))
                        source_domain = getattr(chunk_data, 'source_domain', '')
                    
                    if not chunk_text:
                        self.logger.warning(f"Empty chunk text for index {idx}, skipping")
                        continue
                    
                    chunk = DatabaseChunk(
                        chunk_text=chunk_text,
                        source=source,
                        title=title,
                        chunk_id=str(chunk_id),
                        source_domain=source_domain,
                        similarity_score=float(1 / (1 + distance)) if distance >= 0 else 0.0,
                        metadata=chunk_data if isinstance(chunk_data, dict) else {}
                    )
                    results.append(chunk)
                    
                except Exception as chunk_error:
                    self.logger.warning(f"Error processing chunk {idx}: {chunk_error}")
                    continue
            
            self.logger.info(f"Retrieved {len(results)} chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {e}")
            return []


class WebSearcher:
    """Searches the web for relevant information with intelligent article selection and full content extraction"""
    
    def __init__(self, max_results: int = 5, ollama_host: str = "http://localhost:11434"):
        self.max_results = max_results
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        
        if not HAS_DDGS:
            raise ImportError("duckduckgo-search is required for web searching")
        
        # Import intelligent scraping components
        try:
            from intelligent_web_scraper import (
                ArticleSelectionAgent, IntelligentWebScraper, 
                ArticleCandidate, ExtractedArticle
            )
            self.article_selector = ArticleSelectionAgent(ollama_host)
            self.web_scraper = IntelligentWebScraper()
            self.use_intelligent_scraping = True
            self.logger.info("Intelligent web scraping enabled")
        except ImportError as e:
            self.logger.warning(f"Intelligent scraping not available: {e}, using fallback")
            self.use_intelligent_scraping = False
            self.article_selector = None
            self.web_scraper = None
        
        # Agricultural domain preferences for better search results
        self.trusted_domains = [
            'icar.org.in', 'agricoop.nic.in', 'nfsm.gov.in', 'farmer.gov.in',
            'iari.res.in', 'ncbi.nlm.nih.gov', 'fao.org', 'ifpri.org',
            'cgiar.org', 'worldbank.org', 'irri.org', 'icrisat.org'
        ]
        
        # Content quality indicators
        self.quality_indicators = [
            'research', 'study', 'journal', 'scientific', 'peer-reviewed',
            'published', 'findings', 'data', 'experiment', 'trial'
        ]
    
    def search(self, query: str, num_results: Optional[int] = None) -> List[SearchResult]:
        """Search the web with intelligent article selection and full content extraction"""
        if num_results is None:
            num_results = self.max_results
        
        try:
            # Phase 1: Gather candidate articles from multiple search strategies
            candidates = self._gather_article_candidates(query, num_results * 3)  # Get 3x candidates
            
            if not candidates:
                self.logger.warning(f"No candidates found for query: {query}")
                return []
            
            # Phase 2: Intelligently select best articles for full extraction
            if self.use_intelligent_scraping and len(candidates) > num_results:
                selected_candidates = self.article_selector.select_articles(
                    candidates, query, max_articles=num_results
                )
            else:
                # Fallback: Just take top N by relevance
                candidates.sort(key=lambda x: x.relevance_score, reverse=True)
                selected_candidates = candidates[:num_results]
                for c in selected_candidates:
                    c.selected_for_extraction = True
            
            # Phase 3: Fully extract content from selected articles
            results = []
            for candidate in selected_candidates:
                if candidate.selected_for_extraction:
                    if self.use_intelligent_scraping:
                        # Use intelligent scraper for comprehensive extraction
                        extracted = self.web_scraper.extract_full_article(
                            candidate.url, candidate.title
                        )
                        
                        if extracted and extracted.word_count >= 100:
                            # Create SearchResult with full content
                            search_result = SearchResult(
                                title=extracted.title,
                                url=extracted.url,
                                snippet=candidate.snippet,  # Keep original snippet
                                content=extracted.content,  # Full extracted content
                                timestamp=extracted.extraction_timestamp,
                                relevance_score=candidate.relevance_score
                            )
                            
                            # Add metadata
                            if 'publication_date' in extracted.metadata:
                                search_result.publication_date = extracted.metadata['publication_date']
                            if 'author' in extracted.metadata:
                                search_result.author = extracted.metadata['author']
                            
                            results.append(search_result)
                            self.logger.info(f"Extracted {extracted.word_count} words from {candidate.url}")
                        else:
                            # Fallback to snippet if extraction failed
                            search_result = SearchResult(
                                title=candidate.title,
                                url=candidate.url,
                                snippet=candidate.snippet,
                                content=candidate.snippet,
                                timestamp=datetime.now().timestamp(),
                                relevance_score=candidate.relevance_score
                            )
                            results.append(search_result)
                            self.logger.warning(f"Using snippet for {candidate.url} (extraction failed)")
                    else:
                        # Fallback: Use basic extraction
                        search_result = SearchResult(
                            title=candidate.title,
                            url=candidate.url,
                            snippet=candidate.snippet,
                            content=candidate.snippet,
                            timestamp=datetime.now().timestamp(),
                            relevance_score=candidate.relevance_score
                        )
                        results.append(search_result)
            
            self.logger.info(f"Successfully extracted {len(results)} full articles for query: {query}")
            return results
                
        except Exception as e:
            self.logger.error(f"Error in intelligent web search: {e}")
            # Fallback to old search method
            return self._fallback_search(query, num_results)
    
    def _gather_article_candidates(self, query: str, max_candidates: int) -> List:
        """Gather article candidates from multiple search strategies"""
        try:
            from intelligent_web_scraper import ArticleCandidate
        except ImportError:
            # Fallback: return empty list if module not available
            self.logger.warning("ArticleCandidate not available, using fallback search")
            return []
        
        candidates = []
        collected_urls = set()
        
        # Try multiple search strategies for better coverage
        search_strategies = [
            f"{query} site:.gov.in OR site:.org.in agriculture",  # Government/org sites
            f"{query} agriculture farming research",  # General agriculture
            f"{query} Indian agriculture practices"  # India-specific
        ]
        
        with DDGS() as ddgs:
            for strategy_idx, search_query in enumerate(search_strategies):
                if len(candidates) >= max_candidates:
                    break
                
                try:
                    search_results = ddgs.text(
                        search_query,
                        max_results=max_candidates,
                        safesearch='moderate',
                        region='in-en'
                    )
                    
                    for result in search_results:
                        url = result.get('href', '')
                        if url in collected_urls or not url:
                            continue
                        
                        collected_urls.add(url)
                        
                        title = result.get('title', '')
                        snippet = result.get('body', '')
                        
                        # Calculate scores
                        relevance = self._calculate_relevance_score(title, snippet, url)
                        quality = self._estimate_quality_score(title, snippet, url)
                        
                        from urllib.parse import urlparse
                        domain = urlparse(url).netloc
                        
                        candidate = ArticleCandidate(
                            url=url,
                            title=title,
                            snippet=snippet,
                            source_domain=domain,
                            relevance_score=relevance,
                            estimated_quality=quality
                        )
                        
                        candidates.append(candidate)
                        
                        if len(candidates) >= max_candidates:
                            break
                    
                except Exception as strategy_error:
                    self.logger.warning(f"Search strategy {strategy_idx} failed: {strategy_error}")
                    continue
        
        self.logger.info(f"Gathered {len(candidates)} article candidates")
        return candidates
    
    def _estimate_quality_score(self, title: str, snippet: str, url: str) -> float:
        """Estimate article quality based on various signals"""
        score = 0.5  # Base score
        
        text = f"{title} {snippet}".lower()
        
        # Trusted domain boost
        for domain in self.trusted_domains:
            if domain in url:
                score += 0.3
                break
        
        # Quality indicators
        quality_indicators = [
            'research', 'study', 'journal', 'university', 
            'institute', 'publication', 'peer-reviewed',
            'government', 'official', '.pdf', '.edu', '.gov'
        ]
        quality_count = sum(1 for indicator in quality_indicators if indicator in text or indicator in url)
        score += min(quality_count * 0.05, 0.25)
        
        # Length indicator (longer content often more comprehensive)
        if len(snippet) > 200:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_relevance_score(self, title: str, snippet: str, url: str) -> float:
        """Calculate relevance score based on multiple factors"""
        score = 0.5  # Base score
        
        text = f"{title} {snippet}".lower()
        
        # Boost for trusted domains
        for domain in self.trusted_domains:
            if domain in url:
                score += 0.3
                break
        
        # Boost for quality indicators
        quality_count = sum(1 for indicator in self.quality_indicators if indicator in text)
        score += min(quality_count * 0.05, 0.2)
        
        # Boost for agriculture keywords
        agriculture_keywords = ['crop', 'soil', 'farming', 'agriculture', 'cultivation', 'harvest', 'irrigation']
        ag_count = sum(1 for keyword in agriculture_keywords if keyword in text)
        score += min(ag_count * 0.03, 0.15)
        
        # Boost for India-specific content
        india_keywords = ['india', 'indian', 'bharatiya', 'desi']
        india_count = sum(1 for keyword in india_keywords if keyword in text)
        score += min(india_count * 0.02, 0.1)
        
        return min(score, 1.0)
    
    def _fallback_search(self, query: str, num_results: int) -> List[SearchResult]:
        """Fallback to basic search if intelligent scraping fails"""
        self.logger.info("Using fallback search method")
        results = []
        
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(
                    f"{query} agriculture",
                    max_results=num_results,
                    safesearch='moderate',
                    region='in-en'
                )
                
                for result in search_results:
                    url = result.get('href', '')
                    title = result.get('title', '')
                    snippet = result.get('body', '')
                    
                    relevance = self._calculate_relevance_score(title, snippet, url)
                    
                    search_result = SearchResult(
                        title=title,
                        url=url,
                        snippet=snippet,
                        content=snippet,  # Use snippet as content in fallback
                        timestamp=datetime.now().timestamp(),
                        relevance_score=relevance
                    )
                    
                    results.append(search_result)
                    
                    if len(results) >= num_results:
                        break
        except Exception as e:
            self.logger.error(f"Fallback search failed: {e}")
        
        return results
    
    def _extract_publication_date(self, result: Dict, content: str) -> Optional[str]:
        """Extract publication date from result or content"""
        import re
        from datetime import datetime
        
        # Common date patterns
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b\d{2}/\d{2}/\d{4}\b',  # DD/MM/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b'  # Month DD, YYYY
        ]
        
        text = f"{result.get('title', '')} {result.get('body', '')} {content[:500]}"
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return None
    
    def _extract_extended_content(self, result: Dict, fallback_snippet: str) -> str:
        """Extract extended content from search result with web scraping"""
        try:
            import requests
            from bs4 import BeautifulSoup
            from urllib.parse import urlparse
            
            # Try to fetch and extract content from the actual webpage
            url = result.get('href', '')
            if not url:
                return fallback_snippet
            
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                }
                
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                
                if response.status_code != 200:
                    return fallback_snippet
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Remove unwanted elements
                for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe', 'noscript']):
                    element.decompose()
                
                # Try to find main content using common patterns
                main_content = None
                
                # Strategy 1: Look for article or main tags
                for tag in ['article', 'main', 'div[role="main"]', 'div[id*="content"]', 'div[class*="content"]']:
                    main_content = soup.select_one(tag)
                    if main_content:
                        break
                
                # Strategy 2: Find the largest text block
                if not main_content:
                    max_text_len = 0
                    for div in soup.find_all(['div', 'section']):
                        text = div.get_text(strip=True)
                        if len(text) > max_text_len:
                            max_text_len = len(text)
                            main_content = div
                
                if main_content:
                    # Extract paragraphs from main content
                    paragraphs = []
                    for p in main_content.find_all(['p', 'li']):
                        text = p.get_text(strip=True)
                        # Filter for agriculture-relevant paragraphs
                        if len(text) > 50:
                            paragraphs.append(text)
                    
                    if paragraphs:
                        # Join top paragraphs, prioritizing agriculture-related content
                        agriculture_keywords = ['agriculture', 'farming', 'crop', 'soil', 'cultivation', 'harvest', 'irrigation']
                        
                        # Score and sort paragraphs by agriculture relevance
                        scored_paragraphs = []
                        for para in paragraphs:
                            para_lower = para.lower()
                            score = sum(1 for keyword in agriculture_keywords if keyword in para_lower)
                            scored_paragraphs.append((score, para))
                        
                        scored_paragraphs.sort(key=lambda x: x[0], reverse=True)
                        
                        # Take top relevant paragraphs
                        top_paragraphs = [para for score, para in scored_paragraphs[:8]]
                        full_content = ' '.join(top_paragraphs)
                        
                        # Truncate to reasonable length
                        return full_content[:3000] if len(full_content) > 3000 else full_content
            
            except requests.exceptions.Timeout:
                self.logger.debug(f"Timeout fetching {url}")
            except requests.exceptions.RequestException as e:
                self.logger.debug(f"Request failed for {url}: {e}")
            except Exception as e:
                self.logger.debug(f"Web scraping failed for {url}: {e}")
            
            # Enhanced fallback: improve the snippet
            content_parts = []
            
            # Add main snippet
            if fallback_snippet:
                content_parts.append(fallback_snippet)
            
            # Look for additional content in result
            for key in ['content', 'description', 'abstract']:
                if key in result and result[key]:
                    content_parts.append(str(result[key]))
            
            # Combine and clean content
            full_content = ' '.join(content_parts)
            
            # Remove duplicates and clean up
            sentences = full_content.split('.')
            unique_sentences = []
            seen = set()
            
            for sentence in sentences:
                cleaned = sentence.strip()
                if cleaned and cleaned not in seen and len(cleaned) > 20:
                    unique_sentences.append(cleaned)
                    seen.add(cleaned)
            
            return '. '.join(unique_sentences[:10])  # Limit to top 10 sentences
            
        except Exception as e:
            self.logger.warning(f"Content extraction failed: {e}")
            return fallback_snippet


class MarkdownGenerator:
    """Generates markdown reports from search results with structured citations"""
    
    def __init__(self, citation_manager: Optional[CitationManager] = None):
        self.logger = logging.getLogger(__name__)
        self.citation_manager = citation_manager or CitationManager()
    
    def generate_markdown(self, sub_query_results: List[SubQueryResult], original_query: str) -> str:
        """Generate comprehensive markdown report with complete content and structured citations"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown = f"""# Agriculture Research Report

**Original Query:** {original_query}
**Generated:** {timestamp}
**Sub-queries Processed:** {len(sub_query_results)}

---

"""
        
        for i, result in enumerate(sub_query_results, 1):
            markdown += f"## Sub-query {i}: {result.original_query}\n\n"
            
            # Database results section with complete content and citations
            if result.db_results:
                markdown += "### 📚 Database Results\n\n"
                for j, chunk in enumerate(result.db_results, 1):
                    # Create citation if not exists
                    if not chunk.citation:
                        chunk.citation = self.citation_manager.create_citation(
                            chunk, i - 1, j - 1, "DB"
                        )
                    
                    citation_id = chunk.citation.citation_id
                    
                    markdown += f"**{j}. [{chunk.title or 'Database Entry'}]** `{citation_id}`\n"
                    markdown += f"- **Source:** {chunk.source}\n"
                    if chunk.source_domain:
                        markdown += f"- **Domain:** {chunk.source_domain}\n"
                    markdown += f"- **Similarity Score:** {chunk.similarity_score:.3f}\n"
                    markdown += f"- **Chunk ID:** {chunk.chunk_id}\n"
                    markdown += f"- **Citation ID:** `{citation_id}` (use this in your answer)\n"
                    markdown += f"- **Full Content:**\n\n{chunk.chunk_text}\n\n"
                    markdown += f"- **How to cite:** Reference this content as `{citation_id}`\n\n"
                    markdown += "---\n\n"
            
            # Web results section with complete content and citations
            if result.web_results:
                markdown += "### 🌐 Web Search Results\n\n"
                for j, web_result in enumerate(result.web_results, 1):
                    # Create citation if not exists
                    if not web_result.citation:
                        web_result.citation = self.citation_manager.create_citation(
                            web_result, i - 1, j - 1, "WEB"
                        )
                    
                    citation_id = web_result.citation.citation_id
                    
                    markdown += f"**{j}. [{web_result.title}]({web_result.url})** `{citation_id}`\n"
                    markdown += f"- **URL:** {web_result.url}\n"
                    markdown += f"- **Citation ID:** `{citation_id}` (use this in your answer)\n"
                    markdown += f"- **Relevance Score:** {web_result.relevance_score:.3f}\n"
                    markdown += f"- **Timestamp:** {datetime.fromtimestamp(web_result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    if hasattr(web_result, 'publication_date') and web_result.publication_date:
                        markdown += f"- **Publication Date:** {web_result.publication_date}\n"
                    if web_result.content:
                        markdown += f"- **Full Content:**\n\n{web_result.content}\n\n"
                    else:
                        markdown += f"- **Summary:**\n\n{web_result.snippet}\n\n"
                    markdown += f"- **Source Citation:** `[WEB-{i}-{j}] {web_result.title} - {web_result.url}`\n\n"
                    markdown += "---\n\n"
            
            markdown += "---\n\n"
        
        # Create citation index
        markdown += "## 📖 Citation Index\n\n"
        citation_count = 0
        for i, result in enumerate(sub_query_results, 1):
            for j, chunk in enumerate(result.db_results, 1):
                citation_count += 1
                markdown += f"{citation_count}. `[DB-{i}-{j}]` - {chunk.source} (Database)\n"
            for j, web_result in enumerate(result.web_results, 1):
                citation_count += 1
                markdown += f"{citation_count}. `[WEB-{i}-{j}]` - {web_result.title} ({web_result.url})\n"
        
        # Summary statistics
        total_db_results = sum(len(r.db_results) for r in sub_query_results)
        total_web_results = sum(len(r.web_results) for r in sub_query_results)
        
        markdown += f"""

## 📊 Summary Statistics

- **Total Database Chunks Retrieved:** {total_db_results}
- **Total Web Results Retrieved:** {total_web_results}
- **Sub-queries Generated:** {len(sub_query_results)}
- **Total Citations Available:** {citation_count}

---

*Report generated by Enhanced RAG System*
"""
        
        return markdown
    
    def generate_comprehensive_markdown(self, original_query: str, refined_query: str, 
                                       sub_queries: List[str], sub_query_results: List[SubQueryResult]) -> str:
        """Generate comprehensive markdown report with all pipeline information"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        markdown = f"""# Enhanced RAG Agriculture Research Report

**Original Query:** {original_query}
**Refined Query:** {refined_query}
**Generated:** {timestamp}
**Pipeline Steps:** Query Refinement → Sub-query Generation → Multi-source Retrieval → Content Synthesis

---

## 🔍 Query Processing Pipeline

### 1. Original User Query
```
{original_query}
```

### 2. Refined Query
```
{refined_query}
```

### 3. Generated Sub-queries ({len(sub_queries)})
"""
        for i, sub_query in enumerate(sub_queries, 1):
            markdown += f"{i}. {sub_query}\n"
        
        markdown += "\n---\n\n"
        
        # Process each sub-query result
        for i, result in enumerate(sub_query_results, 1):
            markdown += f"## Sub-query {i}: {result.original_query}\n\n"
            
            # Add agent info if available
            if result.agent_info:
                markdown += f"**Agent Enhancement:** {result.agent_info}\n\n"
            
            # Database results section with complete content
            if result.db_results:
                markdown += "### 📚 Database Results\n\n"
                for j, chunk in enumerate(result.db_results, 1):
                    markdown += f"**{j}. [{chunk.title or 'Database Entry'}]**\n"
                    markdown += f"- **Source:** {chunk.source}\n"
                    if chunk.source_domain:
                        markdown += f"- **Domain:** {chunk.source_domain}\n"
                    markdown += f"- **Similarity Score:** {chunk.similarity_score:.3f}\n"
                    markdown += f"- **Chunk ID:** {chunk.chunk_id}\n"
                    markdown += f"- **Full Content:**\n\n{chunk.chunk_text}\n\n"
                    markdown += f"- **Source Citation:** `[DB-{i}-{j}] {chunk.source}`\n\n"
                    markdown += "---\n\n"
            
            # Web results section with complete content
            if result.web_results:
                markdown += "### 🌐 Web Search Results\n\n"
                for j, web_result in enumerate(result.web_results, 1):
                    markdown += f"**{j}. [{web_result.title}]({web_result.url})**\n"
                    markdown += f"- **URL:** {web_result.url}\n"
                    markdown += f"- **Timestamp:** {datetime.fromtimestamp(web_result.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
                    if web_result.content:
                        markdown += f"- **Full Content:**\n\n{web_result.content}\n\n"
                    else:
                        markdown += f"- **Summary:**\n\n{web_result.snippet}\n\n"
                    markdown += f"- **Source Citation:** `[WEB-{i}-{j}] {web_result.title} - {web_result.url}`\n\n"
                    markdown += "---\n\n"
            
            markdown += "---\n\n"
        
        # Create citation index
        markdown += "## 📖 Citation Index\n\n"
        citation_count = 0
        for i, result in enumerate(sub_query_results, 1):
            for j, chunk in enumerate(result.db_results, 1):
                citation_count += 1
                markdown += f"{citation_count}. `[DB-{i}-{j}]` - {chunk.source} (Database)\n"
            for j, web_result in enumerate(result.web_results, 1):
                citation_count += 1
                markdown += f"{citation_count}. `[WEB-{i}-{j}]` - {web_result.title} ({web_result.url})\n"
        
        # Summary statistics
        total_db_results = sum(len(r.db_results) for r in sub_query_results)
        total_web_results = sum(len(r.web_results) for r in sub_query_results)
        
        markdown += f"""

## 📊 Summary Statistics

- **Original Query:** {original_query}
- **Refined Query:** {refined_query}
- **Total Sub-queries Generated:** {len(sub_queries)}
- **Total Database Chunks Retrieved:** {total_db_results}
- **Total Web Results Retrieved:** {total_web_results}
- **Total Citations Available:** {citation_count}

---

*Comprehensive report generated by Enhanced RAG System*
"""
        
        return markdown


class AnswerSynthesizer:
    """Synthesizes final answer using LLM with citation validation"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", citation_manager: Optional[CitationManager] = None):
        self.ollama_host = ollama_host
        self.citation_manager = citation_manager
        self.logger = logging.getLogger(__name__)
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f'{self.ollama_host}/api/tags', timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            self.logger.error(f"Error getting available models: {e}")
            return []
    
    def synthesize_answer(self, original_query: str, markdown_content: str, model: str = "gemma3:27b",
                         available_citations: Optional[Dict[str, Dict]] = None) -> str:
        """Synthesize final answer from markdown content with enforced inline citations"""
        
        # Import citation enforcer
        try:
            from citation_enforcer import CitationEnforcer
            citation_enforcer = CitationEnforcer()
            use_citation_enforcement = available_citations is not None
        except ImportError as e:
            self.logger.warning(f"Citation enforcer not available: {e}, using basic validation")
            citation_enforcer = None
            use_citation_enforcement = False
        
        # NO TRUNCATION - Use full content for comprehensive synthesis
        # Large models can handle large context windows with proper timeout settings
        optimized_content = markdown_content
        self.logger.info(f"Using full markdown content: {len(markdown_content)} characters")
        
        # Build comprehensive citation reference list
        citation_reference = ""
        if available_citations:
            citation_lines = []
            for cit_id in sorted(list(available_citations.keys())):  # Show ALL citations
                cit_data = available_citations[cit_id]
                title = cit_data.get('title', 'Unknown')[:80]  # Longer titles for context
                source_type = cit_data.get('source', 'unknown')
                citation_lines.append(f"   [{cit_id}] {title} ({source_type})")
            citation_reference = "\n".join(citation_lines)
            self.logger.info(f"Available citations: {len(available_citations)}")
        
        prompt = f"""You are an expert agricultural research assistant. Your task is to write a COMPREHENSIVE, DETAILED research synthesis that extracts and presents ALL relevant information from the provided sources.

CRITICAL INSTRUCTIONS - READ CAREFULLY:

You MUST write a thorough, detailed answer that:
1. Extracts information from EVERY source provided in the research report
2. Includes ALL specific details: species names, scientific names, mechanisms, locations, studies, data points
3. Organizes information logically by themes/categories
4. Provides IN-DEPTH explanations, not superficial summaries
5. Cites EVERY factual sentence with [DB-X-Y] or [WEB-X-Y] format

YOUR WRITING MUST BE:
- **Comprehensive**: 1000-1500 words minimum (this is a DETAILED research synthesis)
- **Specific**: Include scientific names, locations, specific varieties, exact mechanisms
- **Well-structured**: Clear sections covering different aspects of the topic
- **Heavily cited**: 85-95% of sentences should have citations
- **Information-dense**: Every sentence should add new, specific information

RESEARCH REPORT ANALYSIS STRATEGY:

When you read the research report below, you MUST:
1. **Identify ALL unique information sources** (database chunks + web articles)
2. **Extract key information from EACH source**:
   - Specific crop species/varieties mentioned
   - Scientific names (genus, species)
   - Geographic information (regions, countries)
   - Quantitative data (percentages, yields, losses)
   - Mechanisms/processes described
   - Management strategies/solutions
   - Research findings/studies cited
3. **Organize information thematically**:
   - Group similar crops together
   - Explain mechanisms in depth
   - Provide regional context
   - Discuss management approaches
4. **Use EVERY relevant citation** - if a source is provided, extract its value

ANSWER STRUCTURE (MANDATORY):

**Introduction (100-150 words)**:
- Provide comprehensive context on the topic
- Explain scope and significance
- Preview main categories to be discussed
- Cite foundational sources [DB-X-X][WEB-Y-Y]

**Main Body (800-1200 words) - Multiple Detailed Sections**:

For the user's query about "{original_query}", organize your answer into logical sections such as:

Section 1: Major Affected Crops/Primary Information
- List ALL specific crops/species mentioned in sources
- Include scientific names where available
- Provide regional/geographic context
- Cite each crop/fact [DB-X-X][WEB-Y-Y]
- 200-300 words with dense information

Section 2: Detailed Mechanisms/Processes
- Explain HOW/WHY phenomena occur
- Include scientific terminology from sources
- Describe stages, symptoms, impacts
- Cite technical details [DB-X-X][WEB-Y-Y]
- 200-300 words

Section 3: Additional Information from Sources
- Geographic distribution/variations
- Management practices/solutions
- Research findings/studies
- Economic/social impacts
- Cite all specific details [DB-X-X][WEB-Y-Y]
- 200-300 words

Section 4: Contextual/Supporting Information
- Related factors (soil, climate, practices)
- Prevention/control measures
- Future directions/research
- Cite supporting information [DB-X-X][WEB-Y-Y]
- 200-300 words

**Conclusion (100-150 words)**:
- Synthesize key findings across all sources
- Highlight most critical information
- Note gaps or areas needing more research
- Final citations [DB-X-X][WEB-Y-Y]

CITATION REQUIREMENTS (CRITICAL):

- Cite at the END of sentences, before the period: "...information [DB-1-2]."
- Multiple sources for combined info: "...statement [DB-1-2][WEB-2-3]."
- EVERY factual statement needs a citation
- Target: 85-95% citation coverage
- If you mention it, cite it immediately

QUALITY CHECKLIST BEFORE SUBMITTING:

✓ Word count 1000-1500+ words?
✓ Information extracted from 90%+ of provided sources?
✓ Specific details (scientific names, numbers, locations) included?
✓ Each source's unique contribution identified and used?
✓ Citations present in 85-95% of sentences?
✓ Answer organized into clear, logical sections?
✓ Technical terms and precise language used?
✓ No generic statements - all specific to sources?

---

AVAILABLE CITATIONS (USE ALL OF THESE):
{citation_reference if citation_reference else "See research report for all citation IDs"}

---

COMPLETE RESEARCH REPORT:
{optimized_content}

---

USER'S QUESTION: {original_query}

---

Now write your COMPREHENSIVE, DETAILED synthesis. Remember: 
- Extract information from EVERY source
- Be SPECIFIC with names, data, details
- Write 1000-1500+ words
- Cite 85-95% of sentences
- Organize clearly with sections

BEGIN YOUR DETAILED SYNTHESIS:

BEGIN YOUR DETAILED SYNTHESIS:"""

        try:
            # Extended timeout for comprehensive synthesis
            if '70b' in model.lower() or '72b' in model.lower():
                timeout_seconds = 1200  # 20 minutes for 70B+ models (comprehensive)
            elif '27b' in model.lower() or '30b' in model.lower():
                timeout_seconds = 900  # 15 minutes for 27B-30B models (comprehensive)
            elif '13b' in model.lower() or '14b' in model.lower():
                timeout_seconds = 600  # 10 minutes for 13B-14B models
            elif '7b' in model.lower() or '8b' in model.lower():
                timeout_seconds = 400  # 6-7 minutes for 7B-8B models
            else:
                timeout_seconds = 240  # 4 minutes for smaller models
            
            self.logger.info(f"Using {timeout_seconds}s timeout for comprehensive synthesis with {model}")
            
            response = requests.post(
                f'{self.ollama_host}/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'temperature': 0.2,  # Low for precise, factual synthesis
                        'top_p': 0.95,  # Slightly higher for more comprehensive vocabulary
                        'num_ctx': 16384,  # LARGE context window for full content
                        'repeat_penalty': 1.15,  # Stronger penalty to avoid repetition in long responses
                        'num_predict': 6144,  # MUCH LARGER - allow 1500+ word responses
                        'stop': [],  # Don't stop early
                    }
                },
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                answer = response.json()['response'].strip()
                self.logger.info(f"Synthesized answer using {model}")
                
                # Enforce citations using CitationEnforcer
                if use_citation_enforcement and citation_enforcer:
                    self.logger.info("Enforcing citation requirements...")
                    answer = citation_enforcer.enforce_citations(
                        answer, available_citations, original_query, 
                        self.ollama_host, model
                    )
                else:
                    # Fallback validation
                    import re
                    citation_pattern = r'\[(DB|WEB)-(\d+)-(\d+)\]'
                    citations_found = len(re.findall(citation_pattern, answer))
                    
                    if citations_found == 0:
                        self.logger.warning("No citations found in answer")
                        answer += "\n\n**⚠️ Citation Notice:** This answer lacks inline citations. Please refer to the detailed research report for source information."
                    else:
                        self.logger.info(f"Found {citations_found} citations in answer")
                
                return answer
            else:
                return f"Error generating answer: {response.status_code}"
                
        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {e}")
            # Enhanced fallback handling with timeout-specific messaging
            if "Read timed out" in str(e) or "timeout" in str(e).lower():
                return f"""**⏱️ Processing Timeout Notice**

The AI synthesis step timed out after {timeout_seconds} seconds due to the computational requirements of the large model `{model}`. However, comprehensive research has been completed.

**📊 Research Summary:**
- Database chunks retrieved and analyzed
- Web search results gathered and processed  
- Information synthesized into structured report

**🔍 Complete Research Report:**

{markdown_content[:3000]}...

*[Report continues in the Pipeline Info and Full Markdown tabs]*

**💡 Recommendations:**
- Switch to a smaller model (e.g., llama3.2:3b, gemma2:9b) for faster responses
- Review the complete research in the "Full Markdown" tab
- The retrieved information is comprehensive and ready for analysis

**📚 Note**: All retrieved information, citations, and detailed pipeline information are available in the respective tabs above."""
            else:
                return f"""**❌ Processing Error**

An error occurred during AI synthesis: {str(e)}

**📋 Fallback Information Available:**
Please check the following tabs for complete research data:
- **Pipeline Info**: Detailed processing steps and debug information
- **Full Markdown**: Complete structured research report  
- **Citations**: Source references and links

**🔧 Troubleshooting:**
- Verify that Ollama is running: `ollama serve`
- Check if the model is available: `ollama list`
- Try a different model if the current one is unavailable

The research and retrieval phases completed successfully - only the final synthesis step encountered issues."""


class MultiAgentRetriever:
    """Multi-agent system for enhanced retrieval using specialized agents"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host
        self.logger = logging.getLogger(__name__)
        
        # Define specialized agents for different aspects
        self.agents = {
            'crop_specialist': {
                'focus': 'crops, varieties, cultivation, planting, cereal, grain',
                'expertise': 'plant breeding, seed selection, crop rotation, growth stages'
            },
            'soil_expert': {
                'focus': 'soil health, nutrients, fertilizers, amendments',
                'expertise': 'soil chemistry, pH, organic matter, erosion'
            },
            'pest_manager': {
                'focus': 'pests, diseases, IPM, biological control, pathogen, infection',
                'expertise': 'insect control, fungal, bacterial, viral, rust, blight, mildew, resistance, treatment'
            },
            'sustainability_advisor': {
                'focus': 'sustainable practices, organic farming, environment',
                'expertise': 'conservation, renewable energy, water management'
            }
        }
    
    def retrieve_with_agents(self, query: str, sub_queries: List[str]) -> List[Dict[str, Any]]:
        """Use specialized agents to enhance retrieval"""
        try:
            enhanced_results = []
            
            for sub_query in sub_queries:
                # Determine best agent for this sub-query
                best_agent = self._select_best_agent(sub_query)
                
                # Generate agent-specific enhanced query
                enhanced_query = self._enhance_query_with_agent(sub_query, best_agent)
                
                # Store the enhanced query and agent info
                enhanced_results.append({
                    'original_query': sub_query,
                    'enhanced_query': enhanced_query,
                    'agent': best_agent,
                    'agent_focus': self.agents[best_agent]['focus'] if best_agent else '',
                    'agent_expertise': self.agents[best_agent]['expertise'] if best_agent else ''
                })
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Multi-agent retrieval failed: {e}")
            return []
    
    def _select_best_agent(self, query: str) -> str:
        """Select the best agent based on query content"""
        query_lower = query.lower()
        
        # Score each agent based on keyword matching
        scores = {}
        for agent_name, agent_info in self.agents.items():
            score = 0
            focus_keywords = agent_info['focus'].lower().split(', ')
            expertise_keywords = agent_info['expertise'].lower().split(', ')
            
            for keyword in focus_keywords:
                if keyword in query_lower:
                    score += 2
            
            for keyword in expertise_keywords:
                if keyword in query_lower:
                    score += 1
            
            scores[agent_name] = score
        
        # Return agent with highest score, or None if no match
        best_agent = max(scores.items(), key=lambda x: x[1])[0]
        return best_agent if scores[best_agent] > 0 else None
    
    def _enhance_query_with_agent(self, query: str, agent: str) -> str:
        """Enhance query with agent-specific knowledge"""
        # If no agent matched, return original query
        if agent is None:
            return query
            
        agent_info = self.agents[agent]
        
        # Add agent-specific context to the query
        enhanced_query = f"{query} {agent_info['focus']} {agent_info['expertise']}"
        
        return enhanced_query


class EnhancedRAGSystem:
    """Main enhanced RAG system combining database and web search with structured citations"""
    
    def __init__(self, 
                 embeddings_dir: Optional[str] = None,
                 ollama_host: str = "http://localhost:11434",
                 temp_dir: Optional[str] = None):
        
        if embeddings_dir is None:
            # Default to relative path
            embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'agriculture_embeddings')
        
        self.embeddings_dir = embeddings_dir
        self.ollama_host = ollama_host
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.logger = logging.getLogger(__name__)
        
        # Initialize citation manager
        self.citation_manager = CitationManager()
        
        # Initialize components with error handling
        self.query_refiner = QueryRefiner(ollama_host)
        self.sub_query_generator = SubQueryGenerator(ollama_host)
        
        # Try to initialize database retriever
        self.db_retriever = None
        self.db_available = False
        try:
            if os.path.exists(embeddings_dir):
                self.db_retriever = DatabaseRetriever(embeddings_dir)
                self.db_available = True
                self.logger.info("Database retriever initialized successfully")
            else:
                self.logger.warning(f"Embeddings directory not found: {embeddings_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize database retriever: {e}")
            self.logger.info("RAG system will continue with web search only")
        
        self.web_searcher = WebSearcher(ollama_host=ollama_host)
        self.markdown_generator = MarkdownGenerator(self.citation_manager)
        self.answer_synthesizer = AnswerSynthesizer(ollama_host, self.citation_manager)
        
        # Initialize multi-agent system
        try:
            self.multi_agent_retriever = MultiAgentRetriever(ollama_host)
            self.logger.info("Multi-agent retrieval system initialized")
        except Exception as e:
            self.logger.warning(f"Multi-agent system not available: {e}")
            self.multi_agent_retriever = None
    
    def process_query(self, 
                     user_query: str,
                     num_sub_queries: int = 3,
                     db_chunks_per_query: int = 3,
                     web_results_per_query: int = 3,
                     synthesis_model: str = "gemma3:27b",
                     enable_database_search: bool = True,
                     enable_web_search: bool = True) -> Dict[str, Any]:
        """Process user query through the complete RAG pipeline with toggles"""
        
        start_time = datetime.now()
        print(f"\n🔍 Starting Enhanced RAG Pipeline")
        print(f"📝 Original Query: {user_query}")
        print(f"⚙️ Parameters:")
        print(f"   - Sub-queries: {num_sub_queries}")
        print(f"   - DB chunks per query: {db_chunks_per_query}")
        print(f"   - Web results per query: {web_results_per_query}")
        print(f"   - Synthesis model: {synthesis_model}")
        print(f"   - Database search: {'✅ enabled' if enable_database_search else '❌ disabled'}")
        print(f"   - Web search: {'✅ enabled' if enable_web_search else '❌ disabled'}")
        
        self.logger.info(f"Processing query: {user_query}")
        self.logger.info(f"Database search: {'enabled' if enable_database_search else 'disabled'}")
        self.logger.info(f"Web search: {'enabled' if enable_web_search else 'disabled'}")
        
        # Validate that at least one search method is enabled and available
        if not enable_database_search and not enable_web_search:
            error_msg = 'At least one search method (database or web) must be enabled'
            print(f"❌ Error: {error_msg}")
            return {
                'error': error_msg,
                'original_query': user_query,
                'processing_time': 0
            }
        
        # Check if database search is requested but not available
        if enable_database_search and not self.db_available:
            print("⚠️ Warning: Database search requested but database not available, using web search only")
            self.logger.warning("Database search requested but database not available, using web search only")
            enable_database_search = False
            
        # Ensure at least web search is available if database is not
        if not enable_database_search and not enable_web_search:
            error_msg = 'No search methods available (database failed to load and web search disabled)'
            print(f"❌ Error: {error_msg}")
            return {
                'error': error_msg,
                'original_query': user_query,
                'processing_time': 0
            }
        
        # Step 1: Refine the query
        print(f"\n🔧 Step 1: Refining query...")
        refined_query = self.query_refiner.refine_query(user_query)
        print(f"✨ Refined Query: {refined_query}")
        
        # Step 2: Generate sub-queries
        print(f"\n🔗 Step 2: Generating {num_sub_queries} sub-queries...")
        sub_queries = self.sub_query_generator.generate_sub_queries(refined_query, num_sub_queries)
        print(f"📋 Generated Sub-queries:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq}")
        
        # Step 3: Process each sub-query
        print(f"\n🔍 Step 3: Processing sub-queries...")
        sub_query_results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all sub-query processing tasks
            future_to_query = {}
            
            for i, sub_query in enumerate(sub_queries, 1):
                print(f"⚡ Submitting sub-query {i} for processing...")
                future = executor.submit(
                    self._process_sub_query, 
                    sub_query, 
                    db_chunks_per_query if enable_database_search else 0, 
                    web_results_per_query if enable_web_search else 0,
                    enable_database_search,
                    enable_web_search
                )
                future_to_query[future] = (i, sub_query)
            
            # Collect results
            for future in as_completed(future_to_query):
                query_num, sub_query = future_to_query[future]
                try:
                    print(f"📥 Collecting results for sub-query {query_num}...")
                    result = future.result()
                    sub_query_results.append(result)
                    
                    # Log results for this sub-query
                    db_count = len(result.db_results) if hasattr(result, 'db_results') else 0
                    web_count = len(result.web_results) if hasattr(result, 'web_results') else 0
                    print(f"   Sub-query {query_num}: {db_count} DB chunks, {web_count} web results")
                    
                except Exception as e:
                    print(f"❌ Error processing sub-query {query_num}: {e}")
                    self.logger.error(f"Error processing sub-query '{sub_query}': {e}")
        
        # Step 4: Generate comprehensive markdown report
        print(f"\n📄 Step 4: Generating comprehensive markdown report...")
        markdown_content = self.markdown_generator.generate_comprehensive_markdown(
            original_query=user_query,
            refined_query=refined_query,
            sub_queries=sub_queries,
            sub_query_results=sub_query_results
        )
        
        # Save markdown to temporary file
        temp_file_path = None
        try:
            temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md', encoding='utf-8')
            temp_file.write(markdown_content)
            temp_file.close()
            temp_file_path = temp_file.name
            print(f"💾 Markdown report saved to: {temp_file_path}")
        except Exception as e:
            print(f"⚠️ Warning: Could not save markdown file: {e}")
            self.logger.warning(f"Could not save markdown file: {e}")
        
        # Step 5: Synthesize final answer with citation enforcement
        print(f"\n🤖 Step 5: Synthesizing final answer using {synthesis_model}...")
        
        # Build available citations dict for enforcement
        all_citations_dict = {}
        for result_idx, result in enumerate(sub_query_results, 1):
            for db_idx, db_chunk in enumerate(result.db_results, 1):
                cit_id = f"DB-{result_idx}-{db_idx}"
                all_citations_dict[cit_id] = {
                    'title': db_chunk.title or db_chunk.source,
                    'source': 'database',
                    'url': db_chunk.source
                }
            for web_idx, web_result in enumerate(result.web_results, 1):
                cit_id = f"WEB-{result_idx}-{web_idx}"
                all_citations_dict[cit_id] = {
                    'title': web_result.title,
                    'source': 'web',
                    'url': web_result.url
                }
        
        final_answer = self.answer_synthesizer.synthesize_answer(
            original_query=user_query,
            markdown_content=markdown_content,
            model=synthesis_model,
            available_citations=all_citations_dict
        )
        
        # Calculate processing time and statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        total_db_chunks = sum(len(r.db_results) for r in sub_query_results)
        total_web_results = sum(len(r.web_results) for r in sub_query_results)
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"⏱️ Total processing time: {processing_time:.2f} seconds")
        print(f"📊 Final Statistics:")
        print(f"   - Total database chunks retrieved: {total_db_chunks}")
        print(f"   - Total web results retrieved: {total_web_results}")
        print(f"   - Sub-queries processed: {len(sub_queries)}")
        print(f"   - Final answer length: {len(final_answer)} characters")
        print(f"   - Markdown report length: {len(markdown_content)} characters")
        
        # Extract all citations and filter to only those used in the final answer
        all_citations = self._extract_citations(sub_query_results)
        used_citations = self._extract_citations_from_answer(final_answer, all_citations)
        
        print(f"   - Total citations available: {len(all_citations)}")
        print(f"   - Citations used in final answer: {len(used_citations)}")
        
        # Create comprehensive result structure
        result = {
            'success': True,
            'answer': final_answer,
            'original_query': user_query,
            'pipeline_info': {
                'refined_query': refined_query,
                'sub_queries': sub_queries,
                'sub_query_results': [
                    {
                        'query': r.original_query,
                        'db_chunks': len(r.db_results),
                        'web_results': len(r.web_results),
                        'agent_info': r.agent_info
                    } for r in sub_query_results
                ],
                'total_db_chunks': total_db_chunks,
                'total_web_results': total_web_results,
                'synthesis_model': synthesis_model,
                'search_settings': {
                    'database_search_enabled': enable_database_search,
                    'web_search_enabled': enable_web_search,
                    'db_chunks_per_query': db_chunks_per_query if enable_database_search else 0,
                    'web_results_per_query': web_results_per_query if enable_web_search else 0
                }
            },
            'markdown_content': markdown_content,
            'markdown_file_path': temp_file_path,
            'citations': used_citations,  # Only citations actually used in the answer
            'all_citations': all_citations,  # All available citations for reference
            'processing_time': processing_time
        }
        
        return result
    
    def _extract_citations(self, sub_query_results: List[SubQueryResult]) -> List[Dict[str, Any]]:
        """Extract citations from sub-query results"""
        citations = []
        
        for i, result in enumerate(sub_query_results):
            # Database citations
            for j, db_result in enumerate(result.db_results):
                citations.append({
                    'type': 'database',
                    'id': f'DB-{i+1}-{j+1}',
                    'title': db_result.title or db_result.source,
                    'source': db_result.source,
                    'content_preview': db_result.chunk_text[:200] + "..." if len(db_result.chunk_text) > 200 else db_result.chunk_text,
                    'similarity_score': db_result.similarity_score,
                    'full_content': db_result.chunk_text
                })
            
            # Web citations
            for j, web_result in enumerate(result.web_results):
                citations.append({
                    'type': 'web',
                    'id': f'WEB-{i+1}-{j+1}',
                    'title': web_result.title,
                    'url': web_result.url,
                    'content_preview': web_result.snippet[:200] + "..." if len(web_result.snippet) > 200 else web_result.snippet,
                    'relevance_score': web_result.relevance_score,
                    'full_content': web_result.content or web_result.snippet
                })
        
        return citations
    
    def _extract_citations_from_answer(self, answer: str, all_citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations that are actually used in the final answer"""
        import re
        
        # Find all citation patterns in the answer
        citation_pattern = r'\[(DB|WEB)-(\d+)-(\d+)\]'
        used_citation_ids = set()
        
        for match in re.finditer(citation_pattern, answer):
            citation_id = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            used_citation_ids.add(citation_id)
        
        # Filter citations to only include those used in the answer
        used_citations = [
            citation for citation in all_citations
            if citation['id'] in used_citation_ids
        ]
        
        self.logger.info(f"Found {len(used_citation_ids)} citations used in answer out of {len(all_citations)} total citations")
        
        return used_citations
    
    def _process_sub_query(self, sub_query: str, db_chunks: int, web_results: int, 
                          enable_db: bool = True, enable_web: bool = True) -> SubQueryResult:
        """Process a single sub-query to get database and web results with multi-agent enhancement"""
        
        print(f"  🔍 Processing sub-query: {sub_query}")
        
        # Use multi-agent enhancement if available
        enhanced_query = sub_query
        agent_info = {}
        
        if self.multi_agent_retriever:
            try:
                agent_enhancements = self.multi_agent_retriever.retrieve_with_agents(sub_query, [sub_query])
                if agent_enhancements:
                    enhancement = agent_enhancements[0]
                    enhanced_query = enhancement['enhanced_query']
                    agent_info = {
                        'agent': enhancement['agent'],
                        'agent_focus': enhancement['agent_focus'],
                        'agent_expertise': enhancement['agent_expertise']
                    }
                    print(f"    🤖 Enhanced by {enhancement['agent']}: {enhanced_query}")
                    self.logger.info(f"Enhanced query using {enhancement['agent']}: {enhanced_query}")
            except Exception as e:
                print(f"    ⚠️ Multi-agent enhancement failed: {e}")
                self.logger.warning(f"Multi-agent enhancement failed: {e}")
        
        db_results = []
        web_results_list = []
        
        # Get database results if enabled (use enhanced query)
        if enable_db and db_chunks > 0:
            try:
                print(f"    📚 Searching database for {db_chunks} chunks...")
                db_results = self.db_retriever.retrieve_chunks(enhanced_query, db_chunks)
                print(f"    ✅ Retrieved {len(db_results)} database chunks")
                self.logger.info(f"Retrieved {len(db_results)} database chunks for: {enhanced_query}")
            except Exception as e:
                print(f"    ❌ Database retrieval failed: {e}")
                self.logger.error(f"Database retrieval failed for '{enhanced_query}': {e}")
        
        # Get web results if enabled (use enhanced query)
        if enable_web and web_results > 0:
            try:
                print(f"    🌐 Searching web for {web_results} results...")
                web_results_list = self.web_searcher.search(enhanced_query, web_results)
                print(f"    ✅ Retrieved {len(web_results_list)} web results")
                self.logger.info(f"Retrieved {len(web_results_list)} web results for: {enhanced_query}")
            except Exception as e:
                print(f"    ❌ Web search failed: {e}")
                self.logger.error(f"Web search failed for '{enhanced_query}': {e}")
        
        result = SubQueryResult(
            original_query=sub_query,
            sub_queries=[sub_query],
            web_results=web_results_list,
            db_results=db_results
        )
        
        # Add agent information if available
        if agent_info:
            result.agent_info = agent_info
        
        return result
    
    def get_available_synthesis_models(self) -> List[str]:
        """Get available models for synthesis"""
        return self.answer_synthesizer.get_available_models()


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# Test function
if __name__ == "__main__":
    # Setup logging
    setup_logging()
    
    # Initialize the system with relative path
    embeddings_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agriculture_embeddings')
    
    try:
        rag_system = EnhancedRAGSystem(embeddings_dir)
        
        # Test query
        result = rag_system.process_query(
            "What are the best practices for wheat cultivation?",
            num_sub_queries=2,
            db_chunks_per_query=3,
            web_results_per_query=2
        )
        
        print("=== ENHANCED RAG RESULT ===")
        print(f"Original Query: {result['original_query']}")
        print(f"Refined Query: {result.get('refined_query', 'N/A')}")
        print(f"Sub-queries: {result.get('sub_queries', [])}")
        print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
        print(f"Final Answer: {result.get('answer', 'No answer')[:500]}...")
        print(f"Markdown saved to: {result.get('markdown_file_path', 'N/A')}")
        
    except Exception as e:
        print(f"Error: {e}")
