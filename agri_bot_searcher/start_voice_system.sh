#!/bin/bash

# Agriculture Bot Searcher - Voice-Enabled Startup Script
# This script helps you start the voice-enabled agriculture chatbot system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a port is in use
port_in_use() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
}

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🌾 Agriculture Bot Searcher - Voice-Enabled System"
echo "=================================================="

# Check Python
if ! command_exists python3; then
    print_error "Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_success "Virtual environment created"
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    print_status "Activated local virtual environment"
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
    print_status "Activated parent virtual environment"
else
    print_warning "No virtual environment found, using system Python"
fi

# Check if requirements are installed
print_status "Checking dependencies..."

# Check basic requirements
if ! python3 -c "import flask, torch, transformers" 2>/dev/null; then
    print_warning "Some dependencies are missing. Installing requirements..."
    
    if [ -f "requirements_voice.txt" ]; then
        pip install -r requirements_voice.txt
        print_success "Dependencies installed"
    else
        print_error "requirements_voice.txt not found"
        exit 1
    fi
else
    print_success "Core dependencies are available"
fi

# Check for optional dependencies
print_status "Checking optional dependencies..."

python3 -c "
try:
    import nemo
    print('✓ NeMo toolkit available')
except ImportError:
    print('⚠ NeMo toolkit not available (voice features limited)')

try:
    from IndicTransToolkit.processor import IndicProcessor
    print('✓ IndicTrans toolkit available')
except ImportError:
    print('⚠ IndicTrans toolkit not available (translation limited)')

try:
    from sarvamai import SarvamAI
    print('✓ SarvamAI available')
except ImportError:
    print('⚠ SarvamAI not available (premium voice features limited)')
"

# Check for Ollama
print_status "Checking Ollama..."
if ! command_exists ollama; then
    print_error "Ollama is not installed. Please install it first:"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Check if Ollama is running
if ! port_in_use 11434; then
    print_warning "Ollama is not running. Starting Ollama..."
    ollama serve &
    OLLAMA_PID=$!
    
    # Wait for Ollama to start
    sleep 5
    
    if port_in_use 11434; then
        print_success "Ollama started successfully"
    else
        print_error "Failed to start Ollama"
        exit 1
    fi
else
    print_success "Ollama is already running"
fi

# Check if required models are available
print_status "Checking Ollama models..."
if ! ollama list | grep -q "llama2"; then
    print_warning "llama2 model not found. Pulling model..."
    ollama pull llama2
    print_success "llama2 model pulled"
fi

# Check for environment file
if [ ! -f ".env" ]; then
    print_warning "No .env file found. Creating template..."
    cat > .env << EOL
# SarvamAI API Key (optional - get from https://sarvam.ai/)
SARVAM_API_KEY=your_api_key_here

# HuggingFace Token (optional - for model downloads)
HUGGINGFACE_TOKEN=your_hf_token_here

# Model paths
CONFORMER_MODEL_PATH=./models/conformer.nemo

# Ollama configuration
OLLAMA_BASE_PORT=11434
OLLAMA_NUM_AGENTS=2
EOL
    print_success ".env template created"
    print_warning "Please edit .env file with your API keys if you want premium features"
fi

# Check for models directory
if [ ! -d "models" ]; then
    print_status "Creating models directory..."
    mkdir -p models
    print_success "Models directory created"
fi

# Check if we have the conformer model
if [ ! -f "models/conformer.nemo" ] && [ ! -f "../audio_stuff/conformer.nemo" ]; then
    print_warning "Conformer model not found. Voice transcription will use SarvamAI only."
    print_status "To enable full voice features, download the Conformer model:"
    echo "  - Download from AI4Bharat repositories"
    echo "  - Place in models/conformer.nemo"
fi

# Choose startup mode
echo ""
echo "Choose startup mode:"
echo "1) Voice-enabled web interface (recommended)"
echo "2) Original web interface (text only)"
echo "3) Command line interface"
echo "4) Test voice transcription"
echo ""
read -p "Enter your choice [1-4]: " choice

case $choice in
    1)
        print_status "Starting voice-enabled web interface..."
        if [ -f "src/voice_web_ui.py" ]; then
            python3 src/voice_web_ui.py
        else
            print_error "voice_web_ui.py not found"
            exit 1
        fi
        ;;
    2)
        print_status "Starting original web interface..."
        if [ -f "src/web_ui.py" ]; then
            python3 src/web_ui.py
        else
            print_error "web_ui.py not found"
            exit 1
        fi
        ;;
    3)
        print_status "Starting command line interface..."
        if [ -f "src/agriculture_chatbot.py" ]; then
            python3 src/agriculture_chatbot.py
        else
            print_error "agriculture_chatbot.py not found"
            exit 1
        fi
        ;;
    4)
        print_status "Testing voice transcription..."
        python3 -c "
from src.voice_transcription import create_transcriber
print('Testing voice transcription system...')
transcriber = create_transcriber()
status = transcriber.is_model_ready()
print('\nVoice Transcription Status:')
for model, ready in status.items():
    status_icon = '✓' if ready else '✗'
    print(f'  {model}: {status_icon}')
    
print('\nSupported Languages:')
for code, info in transcriber.get_supported_languages().items():
    print(f'  {code}: {info[\"name\"]}')
    
print('\nVoice transcription test completed!')
"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

# Cleanup function
cleanup() {
    if [ ! -z "$OLLAMA_PID" ]; then
        print_status "Stopping Ollama..."
        kill $OLLAMA_PID 2>/dev/null || true
    fi
}

# Set trap for cleanup
trap cleanup EXIT
