#!/usr/bin/env python3
"""
Basic usage example for Agriculture Embedding Generator
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from embedding_system import AgricultureEmbeddingSystem

def main():
    """Basic usage example"""
    
    print("🌾 Agriculture Embedding Generator - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the embedding system
    print("1. Initializing embedding system...")
    embedding_system = AgricultureEmbeddingSystem(
        model_name="Qwen/Qwen3-Embedding-8B",
        chunk_size=256,
        chunk_overlap=25,
        device="auto"  # Will use GPU if available
    )
    
    # Example agricultural record
    sample_record = {
        "title": "Sustainable Rice Farming Techniques in India",
        "text_extracted": """
        Rice is one of the most important staple crops in India, feeding millions of people.
        Sustainable farming practices are crucial for maintaining soil health and ensuring
        long-term productivity. Organic farming methods, including the use of compost and
        natural fertilizers, can significantly improve soil fertility. Crop rotation with
        legumes helps fix nitrogen in the soil, reducing the need for synthetic fertilizers.
        Water management through drip irrigation and rainwater harvesting is essential
        for sustainable rice production, especially in water-scarce regions.
        """,
        "abstract": "This study examines sustainable rice farming practices in India.",
        "link": "https://example.com/rice-farming-study",
        "source_domain": "example.com",
        "crop_types": ["rice"],
        "farming_methods": ["sustainable", "organic"],
        "soil_types": ["alluvial soil"],
        "climate_info": ["monsoon", "tropical"],
        "fertilizers": ["compost", "nitrogen"],
        "tags": ["sustainability", "organic", "research"]
    }
    
    print("2. Processing sample record...")
    chunks = embedding_system.process_record(sample_record, 0)
    print(f"   Created {len(chunks)} chunks from the record")
    
    # Display chunk information
    for i, chunk in enumerate(chunks):
        print(f"   Chunk {i+1}: {len(chunk.chunk_text)} characters")
        print(f"   Preview: {chunk.chunk_text[:100]}...")
        print()
    
    print("3. Building FAISS index...")
    embedding_system.build_faiss_index()
    
    print("4. Saving embeddings...")
    output_dir = "example_output"
    embedding_system.save_embeddings(output_dir)
    
    print(f"✅ Example completed! Results saved to '{output_dir}/'")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        file_path = os.path.join(output_dir, file)
        size = os.path.getsize(file_path)
        print(f"   {file}: {size:,} bytes")

if __name__ == "__main__":
    main()