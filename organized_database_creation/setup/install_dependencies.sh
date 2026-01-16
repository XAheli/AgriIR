#!/bin/bash

# Install Dependencies for Agriculture Database Creation System
# This script installs all required dependencies for both approaches

echo "🌾 Installing Agriculture Database Creation System Dependencies"
echo "=============================================================="

# Check if Python 3.9+ is available
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.9+ required. Current version: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install requests>=2.31.0
pip install duckduckgo-search>=4.0.0
pip install beautifulsoup4>=4.12.0
pip install lxml>=4.9.0
pip install pyyaml>=6.0.1
pip install urllib3>=2.0.0
pip install charset-normalizer>=3.0.0

# Install PDF processing dependencies
echo "📄 Installing PDF processing dependencies..."
pip install pypdf2>=3.0.0
pip install pymupdf>=1.23.0

# Install OCR dependencies (optional)
echo "🔍 Installing OCR dependencies..."
pip install pytesseract>=0.3.10
pip install pillow>=10.0.0

# Install file type detection
echo "🔎 Installing file type detection..."
pip install python-magic>=0.4.27

# Install from requirements files
echo "📋 Installing from keyword-based requirements..."
if [ -f "keyword_based_search/requirements.txt" ]; then
    pip install -r keyword_based_search/requirements.txt
fi

echo "📋 Installing from autonomous agent requirements..."
if [ -f "autonomous_agent_search/requirements.txt" ]; then
    pip install -r autonomous_agent_search/requirements.txt
fi

# System dependencies check
echo "🔧 Checking system dependencies..."

# Check for tesseract (for OCR)
if command -v tesseract &> /dev/null; then
    echo "✅ Tesseract OCR found: $(tesseract --version | head -1)"
else
    echo "⚠️ Tesseract OCR not found. Install with:"
    echo "   Ubuntu/Debian: sudo apt-get install tesseract-ocr"
    echo "   macOS: brew install tesseract"
    echo "   CentOS/RHEL: sudo yum install tesseract"
fi

# Check for libmagic (for file type detection)
if python3 -c "import magic" 2>/dev/null; then
    echo "✅ python-magic working correctly"
else
    echo "⚠️ python-magic may need system libraries:"
    echo "   Ubuntu/Debian: sudo apt-get install libmagic1"
    echo "   macOS: brew install libmagic"
    echo "   CentOS/RHEL: sudo yum install file-devel"
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p logs
mkdir -p outputs
mkdir -p downloads/pdfs
mkdir -p cache

# Set permissions
chmod +x keyword_based_search/src/*.py
chmod +x autonomous_agent_search/src/*.py

echo ""
echo "✅ Installation completed successfully!"
echo ""
echo "🚀 Quick Start:"
echo "   1. Activate virtual environment: source .venv/bin/activate"
echo "   2. Run keyword-based approach: cd keyword_based_search && python src/agriculture_data_curator.py"
echo "   3. Run autonomous approach: cd autonomous_agent_search && python src/autonomous_agriculture_curator.py"
echo ""
echo "📖 See README.md files in each approach directory for detailed usage instructions."