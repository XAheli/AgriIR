#!/usr/bin/env python3
"""
Agriculture Embedding System
Core module for generating embeddings from agricultural text data using Qwen3-Embedding-8B
"""

import json
import os
import pickle
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for each text chunk"""
    record_id: str
    chunk_id: int
    title: str
    author: str
    link: str
    source_domain: str
    publication_year: str
    indian_regions: List[str]
    crop_types: List[str]
    farming_methods: List[str]
    soil_types: List[str]
    climate_info: List[str]
    fertilizers: List[str]
    plant_species: List[str]
    tags: List[str]
    chunk_text: str
    chunk_start: int
    chunk_end: int
    content_length: int
    relevance_score: float

class AgricultureEmbeddingSystem:
    """
    Main class for generating embeddings from agricultural text data
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen3-Embedding-8B", 
                 chunk_size: int = 256, 
                 chunk_overlap: int = 25,
                 device: str = "auto"):
        """
        Initialize the embedding system
        
        Args:
            model_name: HuggingFace model name for embeddings
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Overlap between chunks in tokens
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Device selection
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Loading {model_name} model...")
        
        # Load model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Successfully loaded {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
        
        # Initialize storage
        self.embeddings = []
        self.metadata = []
        self.index = None
        
        # Optional preprocessing function
        self.preprocess_function: Optional[Callable[[str], str]] = None
        
    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of (chunk_text, start_pos, end_pos) tuples
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Apply preprocessing if available
        if self.preprocess_function:
            text = self.preprocess_function(text)
            
        # Tokenize the text
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
        except Exception as e:
            logger.warning(f"Tokenization failed: {e}")
            return []
        
        if len(tokens) <= self.chunk_size:
            return [(text, 0, len(text))]
        
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Get the chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Decode back to text
            try:
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            except Exception as e:
                logger.warning(f"Decoding failed: {e}")
                start_idx += self.chunk_size - self.chunk_overlap
                continue
            
            # Find approximate character positions
            char_start = int((start_idx / len(tokens)) * len(text))
            char_end = int((end_idx / len(tokens)) * len(text))
            
            chunks.append((chunk_text, char_start, char_end))
            
            # Move start position with overlap
            start_idx += self.chunk_size - self.chunk_overlap
            
        return chunks
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a text chunk using the loaded model
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            # Tokenize with truncation
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.chunk_size, 
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract hidden states
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_state = outputs.hidden_states[-1]
                else:
                    # Fallback for different model architectures
                    base_outputs = self.model.model(**inputs)
                    last_hidden_state = base_outputs.last_hidden_state
                
                # Use mean pooling for embedding
                embedding = last_hidden_state.mean(dim=1).cpu().numpy()[0]
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            # Return zero vector as fallback
            return np.zeros(4096, dtype=np.float32)  # Qwen3-Embedding-8B dimension
    
    def process_record(self, record: Dict[str, Any], record_idx: int) -> List[ChunkMetadata]:
        """
        Process a single JSONL record into chunks with embeddings
        
        Args:
            record: Dictionary containing record data
            record_idx: Index of the record
            
        Returns:
            List of ChunkMetadata objects
        """
        # Extract main text content
        main_text = ""
        if record.get('text_extracted'):
            main_text += record['text_extracted'] + " "
        if record.get('abstract'):
            main_text += record['abstract'] + " "
        
        if not main_text.strip():
            return []
        
        # Create unique record ID
        record_id = hashlib.md5(f"{record.get('link', '')}{record_idx}".encode()).hexdigest()
        
        # Chunk the text
        chunks = self.chunk_text(main_text.strip())
        
        chunk_metadata_list = []
        
        for chunk_idx, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            if len(chunk_text.strip()) < 50:  # Skip very short chunks
                continue
                
            # Create embedding
            embedding = self.create_embedding(chunk_text)
            self.embeddings.append(embedding)
            
            # Create metadata
            metadata = ChunkMetadata(
                record_id=record_id,
                chunk_id=chunk_idx,
                title=record.get('title', ''),
                author=record.get('author', ''),
                link=record.get('link', ''),
                source_domain=record.get('source_domain', ''),
                publication_year=record.get('publication_year', ''),
                indian_regions=record.get('indian_regions', []),
                crop_types=record.get('crop_types', []),
                farming_methods=record.get('farming_methods', []),
                soil_types=record.get('soil_types', []),
                climate_info=record.get('climate_info', []),
                fertilizers=record.get('fertilizers', []),
                plant_species=record.get('plant_species', []),
                tags=record.get('tags', []),
                chunk_text=chunk_text,
                chunk_start=start_pos,
                chunk_end=end_pos,
                content_length=len(chunk_text),
                relevance_score=record.get('relevance_score', 1.0)
            )
            
            chunk_metadata_list.append(metadata)
            self.metadata.append(metadata)
            
        return chunk_metadata_list
    
    def process_dataset(self, 
                       jsonl_file: str, 
                       max_records: Optional[int] = None,
                       filter_function: Optional[Callable[[Dict], bool]] = None):
        """
        Process the entire JSONL dataset
        
        Args:
            jsonl_file: Path to JSONL file
            max_records: Maximum number of records to process
            filter_function: Optional function to filter records
        """
        logger.info(f"Processing dataset: {jsonl_file}")
        
        if not os.path.exists(jsonl_file):
            raise FileNotFoundError(f"Dataset file not found: {jsonl_file}")
        
        total_records = 0
        processed_records = 0
        total_chunks = 0
        
        # Count total records first
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    total_records += 1
        
        logger.info(f"Total records in dataset: {total_records}")
        
        if max_records:
            total_records = min(total_records, max_records)
            logger.info(f"Processing first {total_records} records")
        
        # Process records
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            pbar = tqdm(total=total_records, desc="Processing records")
            
            for record_idx, line in enumerate(f):
                if max_records and processed_records >= max_records:
                    break
                    
                if not line.strip():
                    continue
                
                try:
                    record = json.loads(line)
                    
                    # Apply filter if provided
                    if filter_function and not filter_function(record):
                        continue
                    
                    chunks = self.process_record(record, record_idx)
                    total_chunks += len(chunks)
                    processed_records += 1
                    
                    pbar.set_postfix({
                        'chunks': total_chunks,
                        'avg_chunks_per_record': f"{total_chunks/processed_records:.1f}"
                    })
                    pbar.update(1)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line {record_idx}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing record {record_idx}: {e}")
                    continue
            
            pbar.close()
        
        logger.info(f"Processed {processed_records} records into {total_chunks} chunks")
        return processed_records, total_chunks
    
    def build_faiss_index(self, index_type: str = "flat"):
        """
        Build FAISS index for similarity search
        
        Args:
            index_type: Type of index ('flat', 'ivf')
        """
        if not self.embeddings:
            raise ValueError("No embeddings to index")
        
        logger.info("Building FAISS index...")
        embeddings_array = np.array(self.embeddings)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        dimension = embeddings_array.shape[1]
        
        if index_type == "flat":
            # Use IndexFlatIP for exact cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "ivf":
            # Use IVF for faster approximate search
            nlist = min(100, len(self.embeddings) // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.index.train(embeddings_array)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.index.add(embeddings_array)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dimension}")
    
    def save_embeddings(self, output_dir: str = "embeddings_output"):
        """
        Save embeddings, metadata, and index to disk
        
        Args:
            output_dir: Directory to save outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Saving embeddings to {output_dir}/...")
        
        # Save embeddings
        embeddings_array = np.array(self.embeddings)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings_array)
        
        # Save metadata
        metadata_dicts = []
        for meta in self.metadata:
            metadata_dicts.append({
                'record_id': meta.record_id,
                'chunk_id': meta.chunk_id,
                'title': meta.title,
                'author': meta.author,
                'link': meta.link,
                'source_domain': meta.source_domain,
                'publication_year': meta.publication_year,
                'indian_regions': meta.indian_regions,
                'crop_types': meta.crop_types,
                'farming_methods': meta.farming_methods,
                'soil_types': meta.soil_types,
                'climate_info': meta.climate_info,
                'fertilizers': meta.fertilizers,
                'plant_species': meta.plant_species,
                'tags': meta.tags,
                'chunk_text': meta.chunk_text,
                'chunk_start': meta.chunk_start,
                'chunk_end': meta.chunk_end,
                'content_length': meta.content_length,
                'relevance_score': meta.relevance_score
            })
        
        with open(os.path.join(output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
            json.dump(metadata_dicts, f, ensure_ascii=False, indent=2)
        
        # Save as pickle for faster loading
        with open(os.path.join(output_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, os.path.join(output_dir, "faiss_index.bin"))
        
        # Save configuration
        config = {
            'model_name': self.model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_embeddings': len(self.embeddings),
            'embedding_dimension': embeddings_array.shape[1] if len(self.embeddings) > 0 else 0
        }
        
        with open(os.path.join(output_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {len(self.embeddings)} embeddings and metadata")
        
        # Create summary statistics
        self._create_summary_stats(output_dir)
    
    def _create_summary_stats(self, output_dir: str):
        """Create summary statistics about the dataset"""
        if not self.metadata:
            return
        
        stats = {
            'total_chunks': len(self.metadata),
            'unique_records': len(set(meta.record_id for meta in self.metadata)),
            'avg_chunk_length': np.mean([meta.content_length for meta in self.metadata]),
            'total_content_length': sum(meta.content_length for meta in self.metadata),
            'source_domains': {},
            'crop_types': {},
            'farming_methods': {},
            'soil_types': {},
            'climate_info': {},
            'fertilizers': {},
            'tags': {}
        }
        
        # Count occurrences
        for meta in self.metadata:
            # Source domains
            domain = meta.source_domain
            stats['source_domains'][domain] = stats['source_domains'].get(domain, 0) + 1
            
            # Agricultural categories
            for crop in meta.crop_types:
                stats['crop_types'][crop] = stats['crop_types'].get(crop, 0) + 1
            
            for method in meta.farming_methods:
                stats['farming_methods'][method] = stats['farming_methods'].get(method, 0) + 1
            
            for soil in meta.soil_types:
                stats['soil_types'][soil] = stats['soil_types'].get(soil, 0) + 1
            
            for climate in meta.climate_info:
                stats['climate_info'][climate] = stats['climate_info'].get(climate, 0) + 1
            
            for fertilizer in meta.fertilizers:
                stats['fertilizers'][fertilizer] = stats['fertilizers'].get(fertilizer, 0) + 1
            
            for tag in meta.tags:
                stats['tags'][tag] = stats['tags'].get(tag, 0) + 1
        
        # Sort by frequency
        for key in ['source_domains', 'crop_types', 'farming_methods', 'soil_types', 
                   'climate_info', 'fertilizers', 'tags']:
            stats[key] = dict(sorted(stats[key].items(), key=lambda x: x[1], reverse=True))
        
        with open(os.path.join(output_dir, "summary_stats.json"), 'w') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created summary statistics in {output_dir}/summary_stats.json")