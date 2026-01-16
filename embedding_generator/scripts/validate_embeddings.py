#!/usr/bin/env python3
"""
Script to validate generated embeddings
"""

import argparse
import json
import os
import pickle
import numpy as np
import faiss
from pathlib import Path

def validate_embeddings(embeddings_dir: str):
    """Validate the generated embeddings directory"""
    
    print(f"🔍 Validating embeddings in: {embeddings_dir}")
    
    # Check if directory exists
    if not os.path.exists(embeddings_dir):
        print(f"❌ Directory not found: {embeddings_dir}")
        return False
    
    # Required files
    required_files = [
        'embeddings.npy',
        'metadata.json',
        'metadata.pkl',
        'config.json',
        'summary_stats.json',
        'faiss_index.bin'
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(embeddings_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    
    try:
        # Load and validate config
        with open(os.path.join(embeddings_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        print(f"📋 Configuration:")
        print(f"   Model: {config.get('model_name', 'Unknown')}")
        print(f"   Chunk size: {config.get('chunk_size', 'Unknown')}")
        print(f"   Total embeddings: {config.get('total_embeddings', 'Unknown')}")
        print(f"   Embedding dimension: {config.get('embedding_dimension', 'Unknown')}")
        
        # Load and validate embeddings
        embeddings = np.load(os.path.join(embeddings_dir, 'embeddings.npy'))
        print(f"✅ Embeddings shape: {embeddings.shape}")
        
        # Validate embeddings are not all zeros
        if np.allclose(embeddings, 0):
            print("⚠️  Warning: All embeddings are zero")
        else:
            print("✅ Embeddings contain non-zero values")
        
        # Load and validate metadata
        with open(os.path.join(embeddings_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        print(f"✅ Metadata entries: {len(metadata)}")
        
        # Check consistency
        if len(metadata) != len(embeddings):
            print(f"❌ Mismatch: {len(embeddings)} embeddings vs {len(metadata)} metadata entries")
            return False
        
        print("✅ Embeddings and metadata counts match")
        
        # Load and validate FAISS index
        index = faiss.read_index(os.path.join(embeddings_dir, 'faiss_index.bin'))
        print(f"✅ FAISS index loaded: {index.ntotal} vectors")
        
        if index.ntotal != len(embeddings):
            print(f"❌ FAISS index size mismatch: {index.ntotal} vs {len(embeddings)}")
            return False
        
        print("✅ FAISS index size matches embeddings")
        
        # Load summary stats
        with open(os.path.join(embeddings_dir, 'summary_stats.json'), 'r') as f:
            stats = json.load(f)
        
        print(f"📊 Dataset Statistics:")
        print(f"   Total chunks: {stats.get('total_chunks', 'Unknown')}")
        print(f"   Unique records: {stats.get('unique_records', 'Unknown')}")
        print(f"   Avg chunk length: {stats.get('avg_chunk_length', 'Unknown'):.1f}")
        print(f"   Top domains: {list(stats.get('source_domains', {}).keys())[:3]}")
        print(f"   Top crops: {list(stats.get('crop_types', {}).keys())[:5]}")
        
        # Test similarity search
        print("\n🔍 Testing similarity search...")
        query_vector = embeddings[0:1].copy()  # Use first embedding as query
        faiss.normalize_L2(query_vector)
        
        distances, indices = index.search(query_vector, k=5)
        print(f"✅ Search successful. Top 5 similarities: {distances[0]}")
        
        print("\n✅ All validations passed!")
        return True
        
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate generated embeddings")
    parser.add_argument(
        'embeddings_dir',
        help='Directory containing generated embeddings'
    )
    
    args = parser.parse_args()
    
    success = validate_embeddings(args.embeddings_dir)
    
    if success:
        print("\n🎉 Embeddings validation completed successfully!")
        exit(0)
    else:
        print("\n💥 Embeddings validation failed!")
        exit(1)

if __name__ == "__main__":
    main()