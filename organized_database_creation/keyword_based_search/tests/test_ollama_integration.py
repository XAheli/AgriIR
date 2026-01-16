#!/usr/bin/env python3
"""
Test script for Enhanced Autonomous Agriculture Curator with Ollama
Tests Ollama connectivity and basic LLM functionality
"""

import sys
import time
import logging
from enhanced_autonomous_curator_with_ollama import OllamaConfig, OllamaLLMProcessor

def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("🧪 Testing Ollama Connection...")
    
    config = OllamaConfig(
        base_url="http://localhost:11434",
        model="gemma3:1b"  # Using available model
    )
    
    try:
        processor = OllamaLLMProcessor(config)
        print("✅ Ollama connection successful!")
        return processor
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        print("💡 Please ensure Ollama is running: ollama serve")
        print("💡 And that you have the model: ollama pull llama3.1:8b")
        return None

def test_llm_content_analysis(processor):
    """Test LLM content analysis"""
    print("\n🧪 Testing LLM Content Analysis...")
    
    sample_content = """
    Rice cultivation in Punjab has seen significant improvements with the adoption of precision agriculture techniques. 
    Farmers are now using GPS-guided tractors and variable rate fertilizer application, leading to 15% increase in yield 
    and 20% reduction in fertilizer usage. The Punjab Agricultural University has reported that these technologies 
    are being adopted by over 2,000 farmers across Ludhiana and Patiala districts.
    """
    
    try:
        analysis = processor.enhance_agriculture_content(sample_content)
        print("✅ LLM content analysis successful!")
        print(f"📊 Domain: {analysis.get('domain', 'N/A')}")
        print(f"🎯 Relevance Score: {analysis.get('relevance_score', 'N/A')}")
        print(f"🔍 Key Insights: {len(analysis.get('key_insights', []))} insights found")
        print(f"🧠 LLM Processed: {analysis.get('llm_processed', False)}")
        return True
    except Exception as e:
        print(f"❌ LLM content analysis failed: {e}")
        return False

def test_query_generation(processor):
    """Test LLM query generation"""
    print("\n🧪 Testing LLM Query Generation...")
    
    try:
        queries = processor.generate_enhanced_queries("Precision Agriculture Technology")
        print("✅ LLM query generation successful!")
        print(f"📝 Generated {len(queries)} queries:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
        return True
    except Exception as e:
        print(f"❌ LLM query generation failed: {e}")
        return False

def test_basic_llm_response(processor):
    """Test basic LLM response generation"""
    print("\n🧪 Testing Basic LLM Response...")
    
    prompt = "Explain the importance of soil health in Indian agriculture in 2 sentences."
    
    try:
        response = processor.generate_response(prompt)
        print("✅ Basic LLM response successful!")
        print(f"🤖 Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Basic LLM response failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🌾 ENHANCED AUTONOMOUS AGRICULTURE CURATOR - OLLAMA INTEGRATION TEST 🌾")
    print("="*80)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test 1: Ollama connection
    processor = test_ollama_connection()
    if not processor:
        print("\n❌ Cannot proceed with other tests - Ollama not available")
        sys.exit(1)
    
    # Test 2: Basic LLM response
    basic_test = test_basic_llm_response(processor)
    
    # Test 3: Content analysis
    content_test = test_llm_content_analysis(processor)
    
    # Test 4: Query generation
    query_test = test_query_generation(processor)
    
    # Summary
    print("\n📊 TEST SUMMARY:")
    print("="*50)
    print(f"🔗 Ollama Connection: {'✅ PASS' if processor else '❌ FAIL'}")
    print(f"🤖 Basic LLM Response: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"📊 Content Analysis: {'✅ PASS' if content_test else '❌ FAIL'}")
    print(f"🔍 Query Generation: {'✅ PASS' if query_test else '❌ FAIL'}")
    
    all_passed = processor and basic_test and content_test and query_test
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! The enhanced curator is ready to use.")
        print("🚀 You can now run: python enhanced_autonomous_curator_with_ollama.py")
    else:
        print("\n⚠️ Some tests failed. Please check Ollama setup.")
        print("💡 Make sure Ollama is running and the model is available.")

if __name__ == "__main__":
    main()
