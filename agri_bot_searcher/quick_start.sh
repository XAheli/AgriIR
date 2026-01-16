#!/bin/bash
# Agriculture Bot Searcher - Quick Start Demo

echo "🌾 Agriculture Bot Searcher - Quick Start Demo"
echo "=============================================="

# Check if virtual environment exists
if [[ ! -d "venv" ]]; then
    echo "⚠️  Virtual environment not found. Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import requests, ddgs" 2>/dev/null; then
    echo "❌ Dependencies not found. Installing..."
    pip install -r requirements.txt
fi

echo ""
echo "🎯 Choose demo mode:"
echo "1. 🖥️  Command Line Demo (quick test)"
echo "2. 🌐 Web Interface Demo (full featured)"
echo "3. 📊 Both modes"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🖥️  Running Command Line Demo..."
        python tests/test_agriculture_chatbot.py
        ;;
    2)
        echo ""
        echo "🌐 Running Web Interface Demo..."
        python tests/demo_web_ui.py
        ;;
    3)
        echo ""
        echo "🖥️  First running Command Line Demo..."
        python tests/test_agriculture_chatbot.py
        echo ""
        echo "🌐 Now running Web Interface Demo..."
        python tests/demo_web_ui.py
        ;;
    *)
        echo "❌ Invalid choice. Running default command line demo..."
        python tests/test_agriculture_chatbot.py
        ;;
esac

# Agriculture Bot Searcher - Quick Start Script
# This script quickly demonstrates the agriculture chatbot functionality

set -e

echo "🌾 Agriculture Bot Searcher - Quick Start Demo"
echo "=============================================="

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  Virtual environment not found. Please run ./install.sh first"
    exit 1
fi

# Check if Ollama is running
echo "🔍 Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is running"
else
    echo "❌ Ollama service is not running"
    echo "Please start Ollama with: ollama serve"
    exit 1
fi

echo
echo "🚀 Running quick demo..."
echo "========================"

# Run a quick test
echo "Testing with a simple agricultural query..."
python src/agriculture_chatbot.py \
    --query "What are the signs of nitrogen deficiency in plants?" \
    --agents 1 \
    --searches 1 \
    --exact

echo
echo "🎉 Quick start demo completed!"
echo "=============================="
echo
echo "📚 Next steps:"
echo "  • Run interactive mode: python tests/test_agriculture_chatbot.py --interactive"
echo "  • Try detailed analysis: python src/agriculture_chatbot.py --query 'your question' --agents 2"
echo "  • Start web interface: python src/web_api.py"
echo "  • Read documentation: docs/README_AGRICULTURE_CHATBOT.md"
