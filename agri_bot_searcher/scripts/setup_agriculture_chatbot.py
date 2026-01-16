#!/usr/bin/env python3
"""
Setup script for Agriculture Multi-Agent Chatbot

This script helps set up multiple Ollama instances and download required models.
"""

import subprocess
import time
import requests
import sys
import threading
import os
from typing import List


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Ollama installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Ollama not found")
            return False
    except FileNotFoundError:
        print("❌ Ollama not installed")
        return False


def check_model_available(model_name: str) -> bool:
    """Check if a model is available locally"""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            return model_name in result.stdout
    except:
        pass
    return False


def pull_model(model_name: str):
    """Pull a model from Ollama registry"""
    print(f"📥 Pulling model: {model_name}")
    try:
        result = subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"✅ Model {model_name} pulled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull model {model_name}: {e}")
        return False


def start_ollama_instance(port: int, model: str = "gemma2:2b"):
    """Start an Ollama instance on a specific port"""
    print(f"🚀 Starting Ollama instance on port {port} with model {model}")
    
    env = os.environ.copy()
    env['OLLAMA_HOST'] = f'0.0.0.0:{port}'
    
    try:
        # Start ollama serve in background
        process = subprocess.Popen(
            ["ollama", "serve"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama instance running on port {port}")
                return process
            else:
                print(f"❌ Ollama instance on port {port} not responding")
                process.terminate()
                return None
        except requests.RequestException:
            print(f"❌ Cannot connect to Ollama instance on port {port}")
            process.terminate()
            return None
            
    except Exception as e:
        print(f"❌ Failed to start Ollama instance on port {port}: {e}")
        return None


def check_instance_health(port: int) -> bool:
    """Check if an Ollama instance is healthy"""
    try:
        response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def setup_ollama_instances(num_instances: int = 3, base_port: int = 11434, model: str = "gemma3:1b"):
    """Set up multiple Ollama instances"""
    print(f"🔧 Setting up {num_instances} Ollama instances")
    print(f"📊 Base port: {base_port}, Model: {model}")
    print("=" * 60)
    
    # Check if Ollama is installed
    if not check_ollama_installed():
        print("\n❌ Please install Ollama first:")
        print("Visit: https://ollama.ai/download")
        return False
    
    # Check if model is available
    if not check_model_available(model):
        print(f"\n📥 Model {model} not found locally. Pulling...")
        if not pull_model(model):
            print(f"❌ Failed to pull model {model}")
            return False
    else:
        print(f"✅ Model {model} already available")
    
    # Check for existing instances
    existing_instances = []
    for i in range(num_instances):
        port = base_port + i
        if check_instance_health(port):
            existing_instances.append(port)
            print(f"✅ Ollama instance already running on port {port}")
    
    if len(existing_instances) >= num_instances:
        print(f"\n🎉 All {num_instances} instances are already running!")
        return True
    
    print(f"\n🚀 Need to start {num_instances - len(existing_instances)} more instances...")
    
    # Instructions for manual setup
    print("\n" + "=" * 60)
    print("MANUAL SETUP INSTRUCTIONS:")
    print("=" * 60)
    print("\nPlease open separate terminal windows and run these commands:\n")
    
    for i in range(num_instances):
        port = base_port + i
        if port not in existing_instances:
            print(f"Terminal {i+1}:")
            print(f"export OLLAMA_HOST=0.0.0.0:{port}")
            print(f"ollama serve")
            print()
    
    print("After starting all instances, test with:")
    print("python test_agriculture_chatbot.py")
    print("\nOr run in interactive mode:")
    print("python test_agriculture_chatbot.py --interactive")
    
    return False


def install_requirements():
    """Install required Python packages"""
    requirements = [
        "requests",
        "duckduckgo-search",
        "beautifulsoup4",
        "lxml"
    ]
    
    print("📦 Installing required packages...")
    
    for package in requirements:
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {package} installed")
            else:
                print(f"❌ Failed to install {package}: {result.stderr}")
        except Exception as e:
            print(f"❌ Error installing {package}: {e}")


def main():
    """Main setup function"""
    print("🌾 Agriculture Multi-Agent Chatbot Setup")
    print("=" * 50)
    
    # Install requirements
    install_requirements()
    print()
    
    # Setup Ollama instances
    setup_ollama_instances(num_instances=3, base_port=11434, model="gemma3:1b")


if __name__ == "__main__":
    main()
