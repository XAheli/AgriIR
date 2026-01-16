#!/bin/bash

# Agriculture Bot Searcher - Installation Script
# This script sets up the complete environment for the agriculture chatbot

set -e  # Exit on any error

echo "🌾 Agriculture Bot Searcher - Installation Script"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python is installed
print_info "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
print_status "Python $PYTHON_VERSION found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

print_status "pip3 found"

# Check if Ollama is installed
print_info "Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    print_warning "Ollama is not installed."
    print_info "Please install Ollama from: https://ollama.ai/download"
    print_info "After installing Ollama, run this script again."
    echo
    print_info "Installation commands for different systems:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.ai/install.sh | sh"
    echo "  Windows: Download from https://ollama.ai/download"
    exit 1
else
    OLLAMA_VERSION=$(ollama --version | head -n1)
    print_status "Ollama found: $OLLAMA_VERSION"
fi

# Create virtual environment if it doesn't exist
print_info "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# Install Python dependencies
print_info "Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1
print_status "Python dependencies installed"

# Check if Ollama is running
print_info "Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_status "Ollama service is running on port 11434"
    
    # Check for required model
    print_info "Checking for required model..."
    if ollama list | grep -q "gemma3:1b"; then
        print_status "Model gemma3:1b is available"
    else
        print_warning "Model gemma3:1b not found"
        print_info "Pulling gemma3:1b model (this may take a few minutes)..."
        ollama pull gemma3:1b
        print_status "Model gemma3:1b downloaded"
    fi
else
    print_warning "Ollama service is not running on port 11434"
    print_info "Please start Ollama service:"
    echo "  ollama serve"
fi

# Make scripts executable
print_info "Making scripts executable..."
chmod +x scripts/*.py
chmod +x tests/*.py
chmod +x src/*.py
print_status "Scripts made executable"

# Run setup script
print_info "Running agriculture chatbot setup..."
python scripts/setup_agriculture_chatbot.py

echo
echo "🎉 Installation completed!"
echo "=================================================="
print_status "Agriculture Bot Searcher is ready to use!"
echo
print_info "Quick start commands:"
echo "  # Activate virtual environment:"
echo "  source venv/bin/activate"
echo
echo "  # Test the chatbot:"
echo "  python tests/test_agriculture_chatbot.py"
echo
echo "  # Interactive mode:"
echo "  python tests/test_agriculture_chatbot.py --interactive"
echo
echo "  # Command line usage:"
echo "  python src/agriculture_chatbot.py --query 'How to grow tomatoes?' --exact"
echo
print_info "Documentation available in docs/ folder"
print_info "For Docker deployment, run: docker-compose up -d"
echo
print_warning "Note: Make sure Ollama is running before using the chatbot!"
echo "Start Ollama with: ollama serve"
