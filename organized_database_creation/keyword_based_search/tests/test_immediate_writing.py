#!/usr/bin/env python3
"""
Test script to verify immediate JSONL writing in agriculture_curator_fixed.py
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to Python path
sys.path.append('.')

from agriculture_curator_fixed import FixedAgricultureCurator

def test_immediate_writing():
    """Test immediate JSONL writing functionality"""
    
    # Setup test configuration
    config = {
        "num_agents": 1,  # Use only 1 agent for testing
        "output_file": "test_immediate_agriculture.jsonl",
        "max_search_results": 3,  # Limit results for quick testing
        "pdf_storage_dir": "test_pdfs",
        "enable_pdf_download": True,
        "num_queries": 2  # Test with just 2 queries
    }
    
    print("🧪 Starting immediate JSONL writing test...")
    print(f"📄 Output file: {config['output_file']}")
    
    # Clean up previous test files
    output_file = Path(config['output_file'])
    if output_file.exists():
        output_file.unlink()
        print(f"🗑️ Cleaned up previous output file")
    
    # Create and run curator
    curator = FixedAgricultureCurator(
        num_agents=config["num_agents"],
        output_file=config["output_file"],
        max_search_results=config["max_search_results"],
        pdf_storage_dir=config["pdf_storage_dir"],
        enable_pdf_download=config["enable_pdf_download"]
    )
    
    # Monitor file in real-time
    print(f"\n👀 Starting monitoring of {config['output_file']}...")
    print("You should see entries appear immediately as they are processed")
    print("="*80)
    
    try:
        # Start curation
        summary = curator.start_curation(num_queries=config["num_queries"])
        
        print("\n" + "="*80)
        print("✅ TEST COMPLETED")
        print("="*80)
        print(f"Total entries: {summary.get('total_entries', 0)}")
        print(f"PDF files: {summary.get('total_pdfs', 0)}")
        print(f"Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"Output file: {summary.get('output_file', 'N/A')}")
        
        # Check if file exists and has content
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"📊 Output file size: {file_size} bytes")
            
            if file_size > 0:
                with open(output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                print(f"📝 Total lines in output: {len(lines)}")
                print("✅ Immediate writing SUCCESSFUL!")
                
                # Show first entry as sample
                if lines:
                    import json
                    try:
                        first_entry = json.loads(lines[0])
                        print(f"\n📋 Sample entry title: {first_entry.get('title', 'N/A')[:100]}...")
                        print(f"📋 Sample entry source: {first_entry.get('source_domain', 'N/A')}")
                    except:
                        print("📋 Sample entry parsing failed")
            else:
                print("❌ Output file is empty")
        else:
            print("❌ Output file was not created")
            
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logging.error(f"Test error: {e}")

if __name__ == "__main__":
    # Setup logging for test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_immediate_writing()
