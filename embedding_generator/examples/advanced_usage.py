#!/usr/bin/env python3
"""
Advanced usage examples for Agriculture Embedding Generator
"""

import sys
import os
import json
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedding_system import AgricultureEmbeddingSystem

def custom_preprocessing_example():
    """Example with custom text preprocessing"""
    
    print("🔧 Advanced Example 1: Custom Preprocessing")
    print("-" * 50)
    
    def custom_preprocess(text):
        """Custom preprocessing function"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase for consistency
        text = text.lower()
        
        # Add agricultural context markers
        if 'rice' in text:
            text = "[CROP:RICE] " + text
        if 'organic' in text:
            text = "[METHOD:ORGANIC] " + text
            
        return text
    
    # Initialize with custom preprocessing
    embedding_system = AgricultureEmbeddingSystem(
        chunk_size=128,  # Smaller chunks for this example
        chunk_overlap=10
    )
    
    # Set custom preprocessing function
    embedding_system.preprocess_function = custom_preprocess
    
    # Test with sample text
    sample_text = "Rice farming using ORGANIC methods is sustainable."
    chunks = embedding_system.chunk_text(sample_text)
    
    print(f"Original: {sample_text}")
    print(f"Processed: {chunks[0][0] if chunks else 'No chunks'}")
    print()

def filtering_example():
    """Example with record filtering"""
    
    print("🔍 Advanced Example 2: Record Filtering")
    print("-" * 50)
    
    # Create sample dataset
    sample_records = [
        {
            "title": "Rice Research Paper",
            "text_extracted": "Rice cultivation techniques...",
            "source_domain": "research.org",
            "crop_types": ["rice"]
        },
        {
            "title": "Wheat Farming Guide",
            "text_extracted": "Wheat growing methods...",
            "source_domain": "farming.com",
            "crop_types": ["wheat"]
        },
        {
            "title": "Organic Rice Study",
            "text_extracted": "Organic rice farming benefits...",
            "source_domain": "research.org",
            "crop_types": ["rice"],
            "farming_methods": ["organic"]
        }
    ]
    
    # Save as temporary JSONL
    temp_file = "temp_dataset.jsonl"
    with open(temp_file, 'w') as f:
        for record in sample_records:
            f.write(json.dumps(record) + '\n')
    
    embedding_system = AgricultureEmbeddingSystem(chunk_size=128)
    
    # Filter 1: Only research domain
    def research_filter(record):
        return record.get('source_domain') == 'research.org'
    
    print("Processing with research domain filter...")
    processed, chunks = embedding_system.process_dataset(
        temp_file,
        filter_function=research_filter
    )
    print(f"Processed {processed} records, created {chunks} chunks")
    
    # Clean up
    os.remove(temp_file)
    print()

def similarity_search_example():
    """Example of similarity search with generated embeddings"""
    
    print("🔍 Advanced Example 3: Similarity Search")
    print("-" * 50)
    
    # Create a small dataset
    records = [
        {
            "title": "Rice Irrigation Methods",
            "text_extracted": "Drip irrigation is efficient for rice farming. It conserves water and improves yield.",
            "crop_types": ["rice"],
            "farming_methods": ["irrigation"]
        },
        {
            "title": "Wheat Pest Control",
            "text_extracted": "Organic pest control methods for wheat include companion planting and natural predators.",
            "crop_types": ["wheat"],
            "farming_methods": ["organic", "pest control"]
        },
        {
            "title": "Sustainable Rice Production",
            "text_extracted": "Sustainable rice production involves water management, soil health, and organic practices.",
            "crop_types": ["rice"],
            "farming_methods": ["sustainable", "organic"]
        }
    ]
    
    # Process records
    embedding_system = AgricultureEmbeddingSystem(chunk_size=128)
    
    for i, record in enumerate(records):
        embedding_system.process_record(record, i)
    
    # Build index
    embedding_system.build_faiss_index()
    
    # Perform similarity search
    query_text = "water management in rice farming"
    query_embedding = embedding_system.create_embedding(query_text)
    
    # Normalize for cosine similarity
    import faiss
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = embedding_system.index.search(query_embedding, k=3)
    
    print(f"Query: '{query_text}'")
    print("Top 3 similar chunks:")
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        chunk = embedding_system.metadata[idx]
        print(f"{i+1}. Similarity: {dist:.3f}")
        print(f"   Title: {chunk.title}")
        print(f"   Text: {chunk.chunk_text[:100]}...")
        print()

def batch_processing_example():
    """Example of processing large datasets in batches"""
    
    print("📦 Advanced Example 4: Batch Processing")
    print("-" * 50)
    
    # Simulate large dataset processing
    embedding_system = AgricultureEmbeddingSystem(chunk_size=128)
    
    # Process in smaller batches to manage memory
    batch_size = 100
    total_records = 500  # Simulate 500 records
    
    print(f"Simulating processing of {total_records} records in batches of {batch_size}")
    
    for batch_start in range(0, total_records, batch_size):
        batch_end = min(batch_start + batch_size, total_records)
        batch_num = (batch_start // batch_size) + 1
        
        print(f"Processing batch {batch_num}: records {batch_start}-{batch_end}")
        
        # In real usage, you would process actual records here
        # For demo, we'll just simulate the processing
        simulated_chunks = batch_end - batch_start
        
        print(f"  Processed {simulated_chunks} records in batch {batch_num}")
    
    print("Batch processing simulation completed!")
    print()

def main():
    """Run all advanced examples"""
    
    print("🌾 Agriculture Embedding Generator - Advanced Usage Examples")
    print("=" * 70)
    print()
    
    try:
        custom_preprocessing_example()
        filtering_example()
        similarity_search_example()
        batch_processing_example()
        
        print("✅ All advanced examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in advanced examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()