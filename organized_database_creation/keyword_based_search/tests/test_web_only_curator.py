#!/usr/bin/env python3
"""
Test script for web-only curation (no PDF processing)
"""

import logging
from agriculture_curator_fixed import FixedAgricultureCurator

def main():
    """Test with PDF processing disabled"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚀 Testing Web-Only Agriculture Curator (No PDF Processing)")
    
    # Create curator with PDF processing disabled
    curator = FixedAgricultureCurator(
        num_agents=2,  # Reduced for testing
        output_file="test_web_only_agriculture.jsonl",
        max_search_results=5,  # Reduced for testing
        pdf_storage_dir="test_pdfs",
        enable_pdf_download=False  # DISABLE PDF processing
    )
    
    try:
        # Use a very limited set of queries for testing
        summary = curator.start_curation(num_queries=4)  # Only first 4 queries
        
        print("\n" + "="*60)
        print("WEB-ONLY AGRICULTURE CURATION TEST COMPLETED")
        print("="*60)
        print(f"✅ Total entries: {summary.get('total_entries', 0)}")
        print(f"⏱️ Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"📁 Output file: {summary.get('output_file', 'N/A')}")
        print(f"🤖 Successful agents: {summary.get('successful_agents', 0)}")
        print(f"❌ Failed agents: {summary.get('failed_agents', 0)}")
        print(f"📊 PDF processing: DISABLED")
        
        # Check if output file was created
        from pathlib import Path
        output_file = summary.get('output_file', '')
        if output_file and Path(output_file).exists():
            with open(output_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            print(f"📝 JSONL entries written: {len(lines)}")
            
            if lines:
                import json
                first_entry = json.loads(lines[0])
                print(f"📋 Sample title: {first_entry.get('title', 'Unknown')[:100]}")
                print(f"🔗 Sample URL: {first_entry.get('link', 'Unknown')[:100]}")
                print(f"📊 Sample content length: {first_entry.get('content_length', 0)}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logging.error(f"Test execution failed: {e}")

if __name__ == "__main__":
    main()
