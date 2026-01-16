#!/usr/bin/env python3
"""
Unit tests for the Agriculture Embedding System
"""

import unittest
import tempfile
import json
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedding_system import AgricultureEmbeddingSystem, ChunkMetadata

class TestAgricultureEmbeddingSystem(unittest.TestCase):
    """Test cases for AgricultureEmbeddingSystem"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.embedding_system = AgricultureEmbeddingSystem(
            model_name="Qwen/Qwen3-Embedding-8B",
            chunk_size=128,  # Smaller for testing
            chunk_overlap=10,
            device="cpu"  # Use CPU for testing
        )
        
        self.sample_record = {
            "title": "Test Agricultural Paper",
            "text_extracted": "This is a test document about rice farming. " * 20,  # Long enough to chunk
            "abstract": "Test abstract about sustainable agriculture.",
            "link": "https://test.com/paper",
            "source_domain": "test.com",
            "crop_types": ["rice", "wheat"],
            "farming_methods": ["sustainable", "organic"],
            "soil_types": ["alluvial soil"],
            "climate_info": ["tropical"],
            "fertilizers": ["compost"],
            "tags": ["test", "agriculture"]
        }
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.embedding_system.model)
        self.assertIsNotNone(self.embedding_system.tokenizer)
        self.assertEqual(self.embedding_system.chunk_size, 128)
        self.assertEqual(self.embedding_system.chunk_overlap, 10)
    
    def test_text_chunking(self):
        """Test text chunking functionality"""
        text = "This is a test sentence. " * 50  # Long text
        chunks = self.embedding_system.chunk_text(text)
        
        self.assertGreater(len(chunks), 1)  # Should create multiple chunks
        
        for chunk_text, start, end in chunks:
            self.assertIsInstance(chunk_text, str)
            self.assertGreater(len(chunk_text), 0)
            self.assertIsInstance(start, int)
            self.assertIsInstance(end, int)
            self.assertGreaterEqual(end, start)
    
    def test_empty_text_chunking(self):
        """Test chunking with empty text"""
        chunks = self.embedding_system.chunk_text("")
        self.assertEqual(len(chunks), 0)
        
        chunks = self.embedding_system.chunk_text("   ")
        self.assertEqual(len(chunks), 0)
    
    def test_embedding_creation(self):
        """Test embedding creation"""
        text = "Rice farming is important for food security."
        embedding = self.embedding_system.create_embedding(text)
        
        self.assertIsInstance(embedding, np.ndarray)
        self.assertEqual(embedding.dtype, np.float32)
        self.assertGreater(len(embedding), 0)
        
        # Check that embedding is not all zeros
        self.assertFalse(np.allclose(embedding, 0))
    
    def test_record_processing(self):
        """Test processing of a single record"""
        chunks = self.embedding_system.process_record(self.sample_record, 0)
        
        self.assertGreater(len(chunks), 0)
        
        for chunk in chunks:
            self.assertIsInstance(chunk, ChunkMetadata)
            self.assertEqual(chunk.title, self.sample_record["title"])
            self.assertEqual(chunk.crop_types, self.sample_record["crop_types"])
            self.assertGreater(len(chunk.chunk_text), 0)
    
    def test_empty_record_processing(self):
        """Test processing of empty record"""
        empty_record = {"title": "Empty", "text_extracted": "", "abstract": ""}
        chunks = self.embedding_system.process_record(empty_record, 0)
        
        self.assertEqual(len(chunks), 0)
    
    def test_dataset_processing(self):
        """Test processing of JSONL dataset"""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            json.dump(self.sample_record, f)
            f.write('\n')
            json.dump(self.sample_record, f)  # Add second record
            f.write('\n')
            temp_file = f.name
        
        try:
            processed_records, total_chunks = self.embedding_system.process_dataset(
                temp_file, max_records=2
            )
            
            self.assertEqual(processed_records, 2)
            self.assertGreater(total_chunks, 0)
            self.assertEqual(len(self.embedding_system.embeddings), total_chunks)
            self.assertEqual(len(self.embedding_system.metadata), total_chunks)
            
        finally:
            os.unlink(temp_file)
    
    def test_faiss_index_building(self):
        """Test FAISS index building"""
        # Process a record first
        self.embedding_system.process_record(self.sample_record, 0)
        
        # Build index
        self.embedding_system.build_faiss_index()
        
        self.assertIsNotNone(self.embedding_system.index)
        self.assertEqual(self.embedding_system.index.ntotal, len(self.embedding_system.embeddings))
    
    def test_saving_embeddings(self):
        """Test saving embeddings to disk"""
        # Process a record
        self.embedding_system.process_record(self.sample_record, 0)
        self.embedding_system.build_faiss_index()
        
        # Save to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            self.embedding_system.save_embeddings(temp_dir)
            
            # Check that all files are created
            expected_files = [
                'embeddings.npy',
                'metadata.json',
                'metadata.pkl',
                'config.json',
                'summary_stats.json',
                'faiss_index.bin'
            ]
            
            for file in expected_files:
                file_path = os.path.join(temp_dir, file)
                self.assertTrue(os.path.exists(file_path), f"File {file} not created")
                self.assertGreater(os.path.getsize(file_path), 0, f"File {file} is empty")
    
    def test_custom_preprocessing(self):
        """Test custom preprocessing function"""
        def custom_preprocess(text):
            return text.upper()
        
        self.embedding_system.preprocess_function = custom_preprocess
        
        text = "rice farming"
        chunks = self.embedding_system.chunk_text(text)
        
        self.assertEqual(chunks[0][0], "RICE FARMING")
    
    def test_filter_function(self):
        """Test dataset filtering"""
        records = [
            {"title": "Rice Paper", "text_extracted": "Rice content", "source_domain": "research.org"},
            {"title": "Wheat Paper", "text_extracted": "Wheat content", "source_domain": "farming.com"},
        ]
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for record in records:
                json.dump(record, f)
                f.write('\n')
            temp_file = f.name
        
        try:
            # Filter for research.org only
            def research_filter(record):
                return record.get('source_domain') == 'research.org'
            
            processed, chunks = self.embedding_system.process_dataset(
                temp_file, filter_function=research_filter
            )
            
            self.assertEqual(processed, 1)  # Only one record should pass filter
            
        finally:
            os.unlink(temp_file)

class TestChunkMetadata(unittest.TestCase):
    """Test cases for ChunkMetadata dataclass"""
    
    def test_chunk_metadata_creation(self):
        """Test ChunkMetadata creation"""
        metadata = ChunkMetadata(
            record_id="test123",
            chunk_id=0,
            title="Test Title",
            author="Test Author",
            link="https://test.com",
            source_domain="test.com",
            publication_year="2024",
            indian_regions=["Punjab"],
            crop_types=["rice"],
            farming_methods=["organic"],
            soil_types=["alluvial"],
            climate_info=["tropical"],
            fertilizers=["compost"],
            plant_species=["oryza sativa"],
            tags=["test"],
            chunk_text="Test chunk text",
            chunk_start=0,
            chunk_end=100,
            content_length=100,
            relevance_score=1.0
        )
        
        self.assertEqual(metadata.record_id, "test123")
        self.assertEqual(metadata.chunk_id, 0)
        self.assertEqual(metadata.title, "Test Title")
        self.assertEqual(metadata.crop_types, ["rice"])

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)