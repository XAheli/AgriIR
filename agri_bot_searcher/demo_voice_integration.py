#!/usr/bin/env python3
"""
Demo script for testing voice transcription integration
with the Agriculture Bot Searcher
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from voice_transcription import create_transcriber
    from agriculture_chatbot import AgricultureChatbot
    HAS_VOICE = True
except ImportError as e:
    print(f"Warning: Voice transcription not available: {e}")
    HAS_VOICE = False

def test_voice_transcription():
    """Test voice transcription functionality"""
    print("🎤 Testing Voice Transcription System")
    print("=" * 50)
    
    if not HAS_VOICE:
        print("❌ Voice transcription dependencies not available")
        return False
    
    try:
        # Create transcriber
        transcriber = create_transcriber()
        
        # Check status
        status = transcriber.is_model_ready()
        print("\n📊 Voice Transcription Status:")
        for model, ready in status.items():
            status_icon = "✅" if ready else "❌"
            print(f"   {model}: {status_icon}")
        
        # Show supported languages
        print("\n🗣️ Supported Languages:")
        languages = transcriber.get_supported_languages()
        for code, info in languages.items():
            print(f"   {code}: {info['name']}")
        
        # Test with sample audio if available
        audio_dir = Path("../audio_stuff")
        sample_audio = audio_dir / "marathi01.wav"
        
        if sample_audio.exists():
            print(f"\n🎵 Testing with sample audio: {sample_audio}")
            result = transcriber.transcribe_audio(
                str(sample_audio), 
                language='mr', 
                translate_to_english=True
            )
            
            if result['success']:
                print(f"   📝 Transcription: {result['transcription']}")
                print(f"   🔄 Translation: {result['translation']}")
                print(f"   🔧 Method: {result['method']}")
            else:
                print(f"   ❌ Transcription failed: {result['error']}")
        else:
            print(f"\n⚠️  Sample audio file not found at {sample_audio}")
            print("   Please place a sample audio file to test transcription")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing voice transcription: {e}")
        return False

def test_chatbot_integration():
    """Test chatbot integration"""
    print("\n🤖 Testing Chatbot Integration")
    print("=" * 50)
    
    try:
        # Create chatbot instance
        chatbot = AgricultureChatbot(base_port=11434, num_agents=1)
        
        # Test query
        test_query = "What is the best fertilizer for wheat crops in India?"
        print(f"\n📤 Test Query: {test_query}")
        
        print("⏳ Processing query...")
        start_time = time.time()
        
        result = chatbot.answer_query(
            query=test_query,
            num_searches=1,
            exact_answer=False
        )
        
        execution_time = time.time() - start_time
        
        if result["success"]:
            print(f"✅ Query processed successfully in {execution_time:.1f}s")
            print(f"\n📋 Answer: {result['answer'][:200]}...")
            
            if result.get("citations"):
                print(f"\n📚 Citations: {len(result['citations'])} sources")
                for i, citation in enumerate(result['citations'][:2], 1):
                    print(f"   [{i}] {citation['title'][:50]}...")
        else:
            print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
            
        return result["success"]
        
    except Exception as e:
        print(f"❌ Error testing chatbot: {e}")
        return False

def test_end_to_end():
    """Test end-to-end voice to answer workflow"""
    print("\n🔄 Testing End-to-End Workflow")
    print("=" * 50)
    
    if not HAS_VOICE:
        print("❌ Voice transcription not available for end-to-end test")
        return False
    
    try:
        # Sample voice transcription (simulated)
        simulated_voice_text = "किसान को गेहूं की फसल के लिए कौन सा खाद अच्छा है"
        simulated_translation = "What fertilizer is good for wheat crop for farmers"
        
        print(f"🎤 Simulated Voice Input (Hindi): {simulated_voice_text}")
        print(f"🔄 Translated to English: {simulated_translation}")
        
        # Process with chatbot
        chatbot = AgricultureChatbot(base_port=11434, num_agents=1)
        
        print("⏳ Processing translated query...")
        result = chatbot.answer_query(
            query=simulated_translation,
            num_searches=1,
            exact_answer=False
        )
        
        if result["success"]:
            print("✅ End-to-end workflow successful!")
            print(f"📋 Final Answer: {result['answer'][:150]}...")
            return True
        else:
            print(f"❌ Workflow failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error in end-to-end test: {e}")
        return False

def main():
    """Main demo function"""
    print("🌾 Agriculture Bot Searcher - Voice Integration Demo")
    print("=" * 60)
    
    # Test results
    results = {
        "voice_transcription": False,
        "chatbot_integration": False,
        "end_to_end": False
    }
    
    # Run tests
    results["voice_transcription"] = test_voice_transcription()
    results["chatbot_integration"] = test_chatbot_integration()
    results["end_to_end"] = test_end_to_end()
    
    # Summary
    print("\n📊 Demo Results Summary")
    print("=" * 50)
    
    for test_name, success in results.items():
        status_icon = "✅" if success else "❌"
        print(f"{status_icon} {test_name.replace('_', ' ').title()}: {'PASS' if success else 'FAIL'}")
    
    # Overall status
    all_passed = all(results.values())
    overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
    
    print(f"\n🎯 Overall Status: {overall_status}")
    
    if all_passed:
        print("\n🚀 Your voice-enabled agriculture bot is ready!")
        print("   Run 'python src/voice_web_ui.py' to start the web interface")
    else:
        print("\n🔧 Please check the setup guide and resolve the issues above")
        print("   Refer to VOICE_SETUP_GUIDE.md for detailed instructions")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
