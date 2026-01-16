#!/usr/bin/env python3
"""
Quick test for LLM JSON parsing and PDF downloading
"""
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "keyword_based_search" / "src"))

print("="*80)
print("TESTING LLM AND PDF ISSUES")
print("="*80)

# Test 1: LLM JSON Parsing
print("\n" + "="*60)
print("TEST 1: LLM JSON Response Handling")
print("="*60)

try:
    from enhanced_autonomous_curator_with_ollama import OllamaLLMProcessor, OllamaConfig
    
    config = OllamaConfig(model="gemma3:1b", max_retries=1)
    llm = OllamaLLMProcessor(config)
    
    # Test with simple content
    test_content = "Rice cultivation in Tamil Nadu uses traditional methods."
    print(f"\nTest content: {test_content}")
    
    result = llm.enhance_agriculture_content(test_content, "http://test.com")
    
    print(f"\nResult keys: {list(result.keys())}")
    print(f"LLM Processed: {result.get('llm_processed', False)}")
    print(f"Domain: {result.get('domain', 'N/A')}")
    print(f"Relevance: {result.get('relevance_score', 0.0)}")
    
    if result.get('llm_processed'):
        print("✅ LLM processing SUCCESSFUL")
    else:
        print("⚠️ LLM processing FAILED - using fallback")
        print(f"   Error: {result.get('error', 'Unknown')}")
        
except Exception as e:
    print(f"❌ LLM test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: PDF Downloading
print("\n" + "="*60)
print("TEST 2: PDF Downloading")
print("="*60)

try:
    from agriculture_curator_fixed import ImprovedPDFProcessor
    
    pdf_processor = ImprovedPDFProcessor(
        storage_dir="test_pdfs",
        max_size_mb=50
    )
    
    # Test with a known government PDF
    test_urls = [
        "https://desagri.gov.in/wp-content/uploads/2024/03/2014-15-ASSESSMENT-OF-PRE-AND-POST-HARVEST-LOSSES-OF-IMPORTANT-CROPS-IN-INDIA.pdf",
        "https://agritech.tnau.ac.in/pdf/agri_e_pn_2023_24_230406_092519.pdf"
    ]
    
    for url in test_urls:
        print(f"\n  Testing: {url}")
        
        # Check robots
        if pdf_processor._check_robots_and_license(url):
            print(f"  ✅ Robots/license check PASSED")
        else:
            print(f"  ❌ Robots/license check FAILED")
        
        # Check if it's detected as PDF
        if pdf_processor._is_pdf_url_strict(url):
            print(f"  ✅ PDF URL detection PASSED")
        else:
            print(f"  ❌ PDF URL detection FAILED")
            
except Exception as e:
    print(f"❌ PDF test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
