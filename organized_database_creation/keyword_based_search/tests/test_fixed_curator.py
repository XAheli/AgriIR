#!/usr/bin/env python3
"""
Test script for the fixed agriculture curator with limited queries
"""

from agriculture_curator_fixed import FixedAgricultureCurator
import json

def test_fixed_curator():
    """Test the fixed curator with a small subset of queries"""
    
    print("Testing Fixed Agriculture Data Curator...")
    
    # Create curator with limited scope for testing
    curator = FixedAgricultureCurator(
        num_agents=2,  # Reduced for testing
        output_file="test_agriculture_output.jsonl",
        max_search_results=5,  # Reduced for testing
        pdf_storage_dir="test_pdfs",
        enable_pdf_download=True
    )
    
    try:
        # Run with limited queries for testing
        summary = curator.start_curation(num_queries=10)  # Only 10 queries for testing
        
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Success: {summary.get('success', False)}")
        print(f"Total entries curated: {summary.get('total_entries', 0)}")
        print(f"PDF files processed: {summary.get('total_pdfs', 0)}")
        print(f"Execution time: {summary.get('execution_time', 0):.2f} seconds")
        print(f"Output file: {summary.get('output_file', 'N/A')}")
        print(f"Successful agents: {summary.get('successful_agents', 0)}")
        
        # Check if output file was created and contains data
        try:
            with open("test_agriculture_output.jsonl", 'r') as f:
                lines = f.readlines()
                print(f"Output file contains: {len(lines)} entries")
                
                if lines:
                    # Show first entry as sample
                    first_entry = json.loads(lines[0])
                    print(f"Sample entry title: {first_entry.get('title', 'N/A')}")
                    print(f"Sample entry contains soil_types: {len(first_entry.get('soil_types', []))}")
                    print(f"Sample entry contains climate_info: {len(first_entry.get('climate_info', []))}")
                    
        except FileNotFoundError:
            print("Output file not created")
        except Exception as e:
            print(f"Error reading output file: {e}")
            
        return summary.get('success', False)
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_curator()
    if success:
        print("\nFixed curator test PASSED!")
    else:
        print("\nFixed curator test FAILED!")
