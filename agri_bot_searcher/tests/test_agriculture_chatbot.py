#!/usr/bin/env python3
"""
Simple test script for the Agriculture Multi-Agent Chatbot
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from agriculture_chatbot import AgricultureChatbot
import time


def test_chatbot():
    """Test the chatbot with sample agricultural queries"""
    
    # Sample queries for testing
    test_queries = [
        "What are the best practices for rice cultivation in monsoon season?",
        "How to identify and treat bacterial blight in rice plants?",
        "What are the current market prices for wheat in India?",
        "How does climate change affect crop yields?",
        "What are the latest precision farming technologies?",
        "What government subsidies are available for organic farming?"
    ]
    
    print("🌾 Agriculture Multi-Agent Chatbot Test")
    print("=" * 60)
    
    # Create chatbot instance
    chatbot = AgricultureChatbot(base_port=11434, num_agents=3)
    
    # Check if Ollama instances are available
    available_ports = chatbot.check_ollama_instances()
    print(f"📡 Available Ollama instances: {available_ports}")
    
    if not available_ports:
        print("❌ No Ollama instances available. Please start Ollama servers first.")
        print("\nTo start Ollama instances:")
        print("Terminal 1: ollama serve --port 11434")
        print("Terminal 2: ollama serve --port 11435") 
        print("Terminal 3: ollama serve --port 11436")
        return
    
    print(f"✅ Found {len(available_ports)} available instances")
    print()
    
    # Test with first query
    query = test_queries[0]
    print(f"🔍 Testing query: {query}")
    print("-" * 60)
    
    start_time = time.time()
    result = chatbot.answer_query(query, num_searches=2, exact_answer=False)
    end_time = time.time()
    
    if result["success"]:
        print("✅ SUCCESS! (Detailed Analysis)")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        print(f"👥 Agents used: {result['agent_count']}")
        print(f"📚 Citations: {len(result['citations'])}")
        print()
        print("📝 DETAILED ANSWER:")
        print("-" * 40)
        print(result["answer"])
        
        if result.get("failed_agents", 0) > 0:
            print(f"\n⚠️  {result['failed_agents']} agent(s) failed")
    else:
        print("❌ FAILED!")
        print(f"Error: {result['error']}")
        print(result["answer"])
    
    print("\n" + "=" * 60)
    
    # Test exact answer mode
    print(f"🔍 Testing EXACT ANSWER mode for same query")
    print("-" * 60)
    
    start_time = time.time()
    result_exact = chatbot.answer_query(query, num_searches=2, exact_answer=True)
    end_time = time.time()
    
    if result_exact["success"]:
        print("✅ SUCCESS! (Exact Answer)")
        print(f"⏱️  Total time: {end_time - start_time:.1f} seconds")
        print(f"👥 Agents used: {result_exact['agent_count']}")
        print(f"📚 Citations: {len(result_exact['citations'])}")
        print()
        print("📝 EXACT ANSWER:")
        print("-" * 40)
        print(result_exact["answer"])
        
        if result_exact.get("failed_agents", 0) > 0:
            print(f"\n⚠️  {result_exact['failed_agents']} agent(s) failed")
    else:
        print("❌ FAILED!")
        print(f"Error: {result_exact['error']}")
        print(result_exact["answer"])
    
    print("\n" + "=" * 60)
    print("Test completed!")


def interactive_mode():
    """Interactive mode for testing queries"""
    print("🌾 Agriculture Chatbot - Interactive Mode")
    print("Type 'quit' to exit, 'exact' to toggle exact answer mode")
    print("=" * 50)
    
    chatbot = AgricultureChatbot(base_port=11434, num_agents=3)
    exact_mode = False
    
    # Check availability
    available_ports = chatbot.check_ollama_instances()
    if not available_ports:
        print("❌ No Ollama instances available. Please start Ollama servers first.")
        return
    
    print(f"✅ Found {len(available_ports)} available instances")
    print(f"🎯 Exact answer mode: {'ON' if exact_mode else 'OFF'}")
    
    while True:
        print()
        query = input("🔍 Enter your agricultural query (or 'exact' to toggle mode): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            break
        
        if query.lower() == 'exact':
            exact_mode = not exact_mode
            print(f"🎯 Exact answer mode: {'ON' if exact_mode else 'OFF'}")
            continue
        
        if not query:
            continue
        
        mode_str = "EXACT" if exact_mode else "DETAILED"
        print(f"\n🤖 Processing query ({mode_str} mode): {query}")
        print("-" * 50)
        
        start_time = time.time()
        result = chatbot.answer_query(query, num_searches=2, exact_answer=exact_mode)
        end_time = time.time()
        
        if result["success"]:
            print(result["answer"])
            print(f"\n⏱️  Completed in {end_time - start_time:.1f} seconds ({mode_str} mode)")
        else:
            print(f"❌ Error: {result['error']}")
            print(result["answer"])
    
    print("\n👋 Goodbye!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        test_chatbot()
