#!/usr/bin/env python3
"""
Agriculture Chatbot Demonstration Script

This script demonstrates both the detailed analysis and exact answer modes
of the agriculture chatbot with side-by-side comparison.
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from agriculture_chatbot import AgricultureChatbot


def demonstrate_modes():
    """Demonstrate both chatbot modes with a comparison"""
    
    # Sample queries for demonstration
    demo_queries = [
        "How to control aphids in tomato plants?",
        "What are the signs of nitrogen deficiency in crops?",
        "Best irrigation methods for drought-prone areas?",
        "How to improve soil health for vegetable farming?"
    ]
    
    print("🌾 Agriculture Multi-Agent Chatbot - Mode Comparison Demo")
    print("=" * 70)
    
    # Create chatbot instance
    chatbot = AgricultureChatbot(base_port=11434, num_agents=1)  # Using 1 agent for speed
    
    # Check availability
    available_ports = chatbot.check_ollama_instances()
    if not available_ports:
        print("❌ No Ollama instances available. Please start Ollama first.")
        return
    
    print(f"✅ Found {len(available_ports)} available Ollama instance(s)")
    print()
    
    # Use first query for demo
    query = demo_queries[0]
    print(f"📋 Demo Query: {query}")
    print("=" * 70)
    
    # Test Detailed Mode
    print("🔍 DETAILED ANALYSIS MODE")
    print("-" * 40)
    
    start_time = time.time()
    detailed_result = chatbot.answer_query(query, num_searches=1, exact_answer=False)
    detailed_time = time.time() - start_time
    
    if detailed_result["success"]:
        print("✅ SUCCESS")
        print(f"⏱️  Time: {detailed_time:.1f}s")
        print(f"📚 Citations: {len(detailed_result['citations'])}")
        print(f"📝 Word Count: ~{len(detailed_result['answer'].split())} words")
        print()
        print("📄 SAMPLE OUTPUT (first 500 chars):")
        print("-" * 30)
        sample_detailed = detailed_result['answer'][:500] + "..." if len(detailed_result['answer']) > 500 else detailed_result['answer']
        print(sample_detailed)
    else:
        print(f"❌ FAILED: {detailed_result['error']}")
    
    print("\n" + "=" * 70)
    
    # Test Exact Answer Mode
    print("🎯 EXACT ANSWER MODE")
    print("-" * 40)
    
    start_time = time.time()
    exact_result = chatbot.answer_query(query, num_searches=1, exact_answer=True)
    exact_time = time.time() - start_time
    
    if exact_result["success"]:
        print("✅ SUCCESS")
        print(f"⏱️  Time: {exact_time:.1f}s")
        print(f"📚 Citations: {len(exact_result['citations'])}")
        print(f"📝 Word Count: ~{len(exact_result['answer'].split())} words")
        print()
        print("📄 COMPLETE OUTPUT:")
        print("-" * 30)
        print(exact_result['answer'])
    else:
        print(f"❌ FAILED: {exact_result['error']}")
    
    print("\n" + "=" * 70)
    
    # Comparison Summary
    if detailed_result["success"] and exact_result["success"]:
        print("📊 COMPARISON SUMMARY")
        print("-" * 40)
        print(f"Detailed Mode: {len(detailed_result['answer'].split())} words in {detailed_time:.1f}s")
        print(f"Exact Mode:    {len(exact_result['answer'].split())} words in {exact_time:.1f}s")
        print(f"Size Reduction: {(1 - len(exact_result['answer'].split()) / len(detailed_result['answer'].split())) * 100:.1f}%")
        print(f"Time Difference: {abs(detailed_time - exact_time):.1f}s")
        
        print("\n🎯 Use Cases:")
        print("• Detailed Mode: Research, comprehensive analysis, educational content")
        print("• Exact Mode: Quick advice, field reference, mobile-friendly responses")
    
    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    demonstrate_modes()
