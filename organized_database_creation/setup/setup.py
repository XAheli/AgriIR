#!/usr/bin/env python3
"""
Setup script for Agriculture Data Curator
"""

import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required Python packages"""
    print("Installing Python requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Python requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_ollama():
    """Check if Ollama is installed and available"""
    print("Checking Ollama installation...")
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Ollama found: {result.stdout.strip()}")
            return True
        else:
            print("✗ Ollama not found")
            return False
    except FileNotFoundError:
        print("✗ Ollama not installed")
        return False

def pull_models():
    """Pull required Ollama models"""
    models = ["deepseek-r1:70b", "gemma3:27b"]
    
    for model in models:
        print(f"Checking model: {model}")
        try:
            # Check if model exists
            result = subprocess.run(["ollama", "list"], 
                                  capture_output=True, text=True)
            
            if model in result.stdout:
                print(f"✓ Model {model} already available")
                continue
                
            # Pull model if not available
            print(f"Pulling model {model} (this may take a while)...")
            subprocess.check_call(["ollama", "pull", model])
            print(f"✓ Model {model} pulled successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to pull model {model}: {e}")
            print(f"  You can manually pull it later with: ollama pull {model}")

def create_directories():
    """Create necessary directories"""
    dirs = ["logs", "data", "cache"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created directory: {dir_name}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("Agriculture Data Curator Setup")
    print("=" * 60)
    
    # Check current directory
    if not Path("agriculture_data_curator.py").exists():
        print("✗ agriculture_data_curator.py not found in current directory")
        print("Please run this script from the same directory as agriculture_data_curator.py")
        return False
    
    # Install Python requirements
    if not install_requirements():
        print("Setup failed at Python requirements installation")
        return False
    
    # Check Ollama
    if not check_ollama():
        print("\nOllama not found. Please install Ollama first:")
        print("Visit: https://ollama.ai for installation instructions")
        print("After installing Ollama, run this setup script again.")
        return False
    
    # Create directories
    create_directories()
    
    # Pull models (optional - user can do this manually)
    print("\nModel pulling (optional - may take significant time and disk space):")
    response = input("Do you want to pull the required models now? (y/N): ").lower()
    if response == 'y':
        pull_models()
    else:
        print("Skipping model pulling. You can pull models manually later:")
        print("  ollama pull deepseek-r1:70b")
        print("  ollama pull gemma3:27b")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("You can now run the agriculture data curator:")
    print("  python agriculture_data_curator.py")
    print("\nMake sure to have models pulled before running:")
    print("  ollama pull deepseek-r1:70b  # or")
    print("  ollama pull gemma3:27b")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
