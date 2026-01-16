#!/usr/bin/env python3
"""
Agriculture Bot Searcher - Web UI Demo
Demonstrates the web interface capabilities with sample queries
"""

import sys
import os
import time
import webbrowser
import subprocess
import requests
from threading import Thread

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

def check_server_ready(url="http://localhost:5000", timeout=30):
    """Check if the web server is ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/api/status", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False

def demo_web_ui():
    """Run the web UI demo"""
    print("🌾 Agriculture Bot Searcher - Web UI Demo")
    print("==========================================")
    
    # Check if Flask is available
    try:
        import flask
        import flask_cors
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.web_ui import app
        print("✅ Flask and web UI components loaded successfully")
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Run: pip install flask flask-cors pyyaml")
        return False
    
    print("\n🚀 Starting web server...")
    
    # Start the web server in a separate thread
    def run_server():
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    
    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    if check_server_ready():
        print("✅ Server is ready!")
        
        # Test API endpoints
        print("\n🔧 Testing API endpoints...")
        
        try:
            # Test status endpoint
            response = requests.get("http://localhost:5000/api/status", timeout=10)
            if response.status_code == 200:
                status_data = response.json()
                available_ports = status_data.get("available_ports", [])
                print(f"📊 Status endpoint: ✅ Working")
                print(f"📡 Available Ollama ports: {available_ports if available_ports else 'None (Ollama not running)'}")
            else:
                print(f"📊 Status endpoint: ❌ Error {response.status_code}")
        except Exception as e:
            print(f"📊 Status endpoint: ❌ Error - {e}")
        
        # Open browser
        print("\n🌐 Opening web interface in browser...")
        try:
            webbrowser.open("http://localhost:5000")
            print("✅ Browser opened successfully")
        except Exception as e:
            print(f"❌ Could not open browser: {e}")
            print("💡 Manually open: http://localhost:5000")
        
        print("\n🎯 Web Interface Features:")
        print("   📝 Interactive query input with large text area")
        print("   🔧 Configurable Ollama port and agent count")
        print("   🎯 Toggle between detailed/concise answer modes")
        print("   📊 Real-time system status monitoring")
        print("   📈 Performance statistics after each query")
        print("   🔍 Live web search integration")
        print("   📚 Automatic citation formatting")
        
        print("\n💡 Sample Queries to Try:")
        sample_queries = [
            "What are the best organic pest control methods for tomatoes?",
            "How to improve soil fertility for wheat cultivation?",
            "What are the signs of nitrogen deficiency in rice plants?",
            "Best irrigation techniques for drought-prone areas?",
            "How to prevent fungal diseases in cucumber crops?"
        ]
        
        for i, query in enumerate(sample_queries, 1):
            print(f"   {i}. {query}")
        
        print("\n🚀 Server is running at: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the demo")
        
        # Keep the demo running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Demo stopped by user")
            
    else:
        print("❌ Server failed to start within timeout")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = demo_web_ui()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)
