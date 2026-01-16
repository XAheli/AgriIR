#!/bin/bash
# Agriculture Bot Searcher - Web UI Launcher

echo "🌾 Agriculture Bot Searcher - Web Interface"
echo "============================================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]] && [[ ! -d "venv" ]]; then
    echo "⚠️  No virtual environment detected. Please run ./install.sh first"
    exit 1
fi

# Activate virtual environment if it exists
if [[ -d "venv" ]] && [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Flask not found. Installing dependencies..."
    pip install -r requirements.txt
fi

# Set default values
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-5000}
DEBUG=${DEBUG:-false}

echo "🚀 Starting Agriculture Bot Web Interface..."
echo "📍 Server: http://$HOST:$PORT"
echo "🔧 Debug mode: $DEBUG"
echo "📝 Make sure Ollama is running on port 11434+"
echo ""
echo "💡 Configuration tips:"
echo "   - Adjust Ollama port in the web interface"
echo "   - Select number of agents (1-6)" 
echo "   - Choose between detailed/concise answers"
echo ""
echo "🛑 Press Ctrl+C to stop the server"
echo ""

# Start the web server
python src/web_ui.py
