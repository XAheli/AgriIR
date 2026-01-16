#!/bin/bash

# AgriIR - Streamlined Installation Script
# Unified installation for agriculture chatbot with voice capabilities

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_NAME="agriir_env"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
AGRIIR_PATH="$PROJECT_ROOT/agri_bot_searcher"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
PYTHON_VERSION="3.12"
EMBEDDINGS_URL="https://drive.google.com/uc?id=1OUhXyGmqrA4tfI97Ja6HZrjGOANkTXG0"
EMBEDDINGS_BROWSER_URL="https://drive.google.com/file/d/1OUhXyGmqrA4tfI97Ja6HZrjGOANkTXG0/view?usp=drive_link"
EMBEDDINGS_ZIP="embeddings.zip"
EMBEDDINGS_SIZE="~40GB"
DATASET_FILE="autonomous_indian_agriculture_complete_repaired.jsonl.tar.xz"

# Installation options
VOICE_ENABLED=true
RAG_ENABLED=true
OLLAMA_MODEL="llama3.2:3b"
DOWNLOAD_EMBEDDINGS=false
GENERATE_EMBEDDINGS=false
USE_EXISTING_DATASET=false
AUTO_DETECT_GPU=true

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --no-voice)
                VOICE_ENABLED=false
                shift
                ;;
            --no-rag)
                RAG_ENABLED=false
                shift
                ;;
            --model)
                OLLAMA_MODEL="$2"
                AUTO_DETECT_GPU=false
                shift 2
                ;;
            --download-embeddings)
                DOWNLOAD_EMBEDDINGS=true
                shift
                ;;
            --generate-embeddings)
                GENERATE_EMBEDDINGS=true
                shift
                ;;
            --use-existing-dataset)
                USE_EXISTING_DATASET=true
                shift
                ;;
            --no-auto-detect)
                AUTO_DETECT_GPU=false
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_warning "Unknown option: $1"
                shift
                ;;
        esac
    done
}

# Show help information
show_help() {
    echo "AgriIR Installation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-voice              Disable voice transcription features"
    echo "  --no-rag                Disable RAG (Retrieval Augmented Generation) system"
    echo "  --model MODEL           Set Ollama model (default: llama3.2:3b or auto-detected)"
    echo "  --download-embeddings   Download pre-trained embeddings ($EMBEDDINGS_SIZE) from Google Drive"
    echo "  --generate-embeddings   Generate embeddings from local dataset"
    echo "  --use-existing-dataset  Use existing dataset file to generate embeddings"
    echo "  --no-auto-detect        Disable GPU auto-detection for model selection"
    echo "  --help                  Show this help message"
    echo ""
    echo "Embeddings Options:"
    echo "  You have three ways to get embeddings:"
    echo "  1. Download pre-generated embeddings (--download-embeddings)"
    echo "  2. Generate from existing dataset (--generate-embeddings --use-existing-dataset)"
    echo "  3. Skip embeddings (continue without)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Full installation with all features"
    echo "  $0 --no-voice                        # Install without voice features"
    echo "  $0 --model gemma3:2b                 # Use Gemma 3 model instead of Llama"
    echo "  $0 --download-embeddings             # Download and extract embeddings"
    echo "  $0 --generate-embeddings --use-existing-dataset  # Generate from local dataset"
}

# System requirements check
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    python3 -c "
import sys
if sys.version_info < (3, 8):
    print(f'Python 3.8 or higher is required (found: {sys.version_info.major}.{sys.version_info.minor})')
    exit(1)
elif sys.version_info >= (3, 13):
    print(f'WARNING: Python 3.13+ detected. Some dependencies (thinc/spacy) may not be compatible.')
    print(f'Recommended: Python 3.11 or 3.12. Will use uv to install Python {sys.version_info.major}.{sys.version_info.minor}.')
    exit(0)
else:
    print(f'Python version {sys.version_info.major}.{sys.version_info.minor} is compatible')
    exit(0)
"
    if [ $? -ne 0 ]; then
        log_error "Python version check failed"
        exit 1
    fi
    
    # Check for pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check for git
    if ! command -v git &> /dev/null; then
        log_error "git is required but not installed"
        exit 1
    fi
    
    log_success "System requirements check passed"
}

# Ensure uv is installed
ensure_uv() {
    log_info "Checking for uv package manager..."
    if ! command -v uv &> /dev/null; then
        log_warning "uv not found. Installing uv..."
        if command -v pipx &> /dev/null; then
            pipx install uv
        else
            pip3 install --upgrade uv
        fi
        log_success "uv installed successfully"
    else
        log_success "uv is already installed"
    fi
}

# Detect GPU VRAM
detect_gpu_vram() {
    log_info "Detecting GPU configuration..."
    
    if command -v nvidia-smi &> /dev/null; then
        # Get total VRAM in MB
        vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
        vram_gb=$((vram_mb / 1024))
        
        log_info "Detected NVIDIA GPU with ${vram_gb}GB VRAM"
        
        if [ $vram_gb -ge 40 ]; then
            log_success "GPU has sufficient VRAM (${vram_gb}GB) for large models"
            return 0
        else
            log_info "GPU VRAM (${vram_gb}GB) is below 40GB threshold"
            return 1
        fi
    else
        log_warning "No NVIDIA GPU detected or nvidia-smi not available"
        return 1
    fi
}

# Prompt user for model selection
select_ollama_model() {
    if [ "$AUTO_DETECT_GPU" = true ]; then
        if detect_gpu_vram; then
            echo ""
            log_info "GPU with sufficient VRAM detected. Recommended model: gemma3:27b"
            read -p "$(echo -e ${YELLOW}Would you like to use gemma3:27b? [Y/n]: ${NC})" choice
            case "$choice" in
                n|N|no|No|NO)
                    prompt_model_selection
                    ;;
                *)
                    OLLAMA_MODEL="gemma3:27b"
                    log_success "Selected model: $OLLAMA_MODEL"
                    ;;
            esac
        else
            prompt_model_selection
        fi
    else
        if [ "$OLLAMA_MODEL" = "llama3.2:3b" ]; then
            prompt_model_selection
        fi
    fi
}

# Prompt user to choose model manually
prompt_model_selection() {
    echo ""
    log_info "Available Ollama models:"
    echo "  1) llama3.2:3b (Default, ~2GB VRAM)"
    echo "  2) gemma3:2b (~1.6GB VRAM)"
    echo "  3) gemma3:9b (~5.5GB VRAM)"
    echo "  4) gemma3:27b (~16GB VRAM, requires ~40GB VRAM)"
    echo "  5) llama3.1:8b (~4.7GB VRAM)"
    echo "  6) Custom (enter model name)"
    echo ""
    read -p "$(echo -e ${YELLOW}Select a model [1-6]: ${NC})" model_choice
    
    case "$model_choice" in
        1)
            OLLAMA_MODEL="llama3.2:3b"
            ;;
        2)
            OLLAMA_MODEL="gemma3:2b"
            ;;
        3)
            OLLAMA_MODEL="gemma3:9b"
            ;;
        4)
            OLLAMA_MODEL="gemma3:27b"
            ;;
        5)
            OLLAMA_MODEL="llama3.1:8b"
            ;;
        6)
            read -p "$(echo -e ${YELLOW}Enter custom model name: ${NC})" custom_model
            OLLAMA_MODEL="$custom_model"
            ;;
        *)
            log_warning "Invalid selection. Using default: llama3.2:3b"
            OLLAMA_MODEL="llama3.2:3b"
            ;;
    esac
    
    log_success "Selected model: $OLLAMA_MODEL"
}

# Prompt user about embeddings download
prompt_embeddings_download() {
    if [ "$DOWNLOAD_EMBEDDINGS" = false ] && [ "$GENERATE_EMBEDDINGS" = false ]; then
        echo ""
        log_info "═══════════════════════════════════════════════════════════════"
        log_info "                    EMBEDDINGS SETUP OPTIONS                   "
        log_info "═══════════════════════════════════════════════════════════════"
        echo ""
        echo "Choose how to set up embeddings for AgriIR:"
        echo ""
        echo "  1) Download pre-generated embeddings from Google Drive ($EMBEDDINGS_SIZE)"
        echo "     - Fastest option (requires stable internet)"
        echo "     - Download time: ~30-60 minutes (depending on connection)"
        echo "     - Can be resumed if interrupted"
        echo ""
        echo "  2) Generate embeddings from existing local dataset"
        echo "     - Uses: $DATASET_FILE"
        echo "     - Requires GPU for faster processing"
        echo "     - Generation time: ~2-4 hours (with GPU)"
        echo ""
        echo "  3) Skip embeddings setup for now"
        echo "     - You can set up embeddings later manually"
        echo ""
        read -p "$(echo -e ${YELLOW}Select option [1-3]: ${NC})" choice
        
        case "$choice" in
            1)
                DOWNLOAD_EMBEDDINGS=true
                log_success "Will download pre-generated embeddings"
                ;;
            2)
                if [ -f "$PROJECT_ROOT/$DATASET_FILE" ]; then
                    GENERATE_EMBEDDINGS=true
                    USE_EXISTING_DATASET=true
                    log_success "Will generate embeddings from local dataset"
                else
                    log_error "Dataset file not found: $PROJECT_ROOT/$DATASET_FILE"
                    log_info "Please ensure the dataset file exists or choose option 1"
                    prompt_embeddings_download
                fi
                ;;
            3)
                log_info "Skipping embeddings setup"
                ;;
            *)
                log_warning "Invalid selection. Please try again."
                prompt_embeddings_download
                ;;
        esac
    fi
}

# Create and setup virtual environment
setup_virtual_environment() {
    log_info "Setting up virtual environment with Python $PYTHON_VERSION..."
    
    if [ -d "$VENV_PATH" ]; then
        log_warning "Virtual environment already exists. Removing..."
        rm -rf "$VENV_PATH"
    fi
    
    # Ensure uv is installed
    ensure_uv
    
    # Create virtual environment with specific Python version using uv
    log_info "Creating virtual environment with uv (Python $PYTHON_VERSION)..."
    uv venv "$VENV_PATH" --python "$PYTHON_VERSION"
    
    if [ $? -ne 0 ]; then
        log_error "Failed to create virtual environment with Python $PYTHON_VERSION"
        log_info "Falling back to system Python..."
        python3 -m venv "$VENV_PATH"
    fi
    
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip in the virtual environment
    pip install --upgrade pip setuptools wheel
    
    log_success "Virtual environment created and activated"
}

# Install Python dependencies using uv
install_python_dependencies() {
    log_info "Installing Python dependencies from requirements.txt using uv..."
    
    if [ ! -f "$REQUIREMENTS_FILE" ]; then
        log_error "Requirements file not found: $REQUIREMENTS_FILE"
        exit 1
    fi
    
    # Set UV_LINK_MODE to copy to avoid hardlink warnings
    export UV_LINK_MODE=copy
    
    # Use uv inside the venv
    uv pip install -r "$REQUIREMENTS_FILE"
    
    # Install gdown for Google Drive downloads
    log_info "Installing gdown for embeddings download..."
    uv pip install gdown
    
    log_success "Python dependencies installed successfully with uv"
}

# Install Ollama
install_ollama() {
    log_info "Installing Ollama..."
    
    if command -v ollama &> /dev/null; then
        log_success "Ollama already installed"
    else
        curl -fsSL https://ollama.ai/install.sh | sh
        
        if command -v systemctl &> /dev/null; then
            sudo systemctl enable ollama
            sudo systemctl start ollama
        else
            nohup ollama serve > /tmp/ollama.log 2>&1 &
        fi
        
        sleep 5
    fi
    
    # Select and download model
    select_ollama_model
    
    log_info "Downloading Ollama model: $OLLAMA_MODEL"
    log_warning "This may take a while depending on the model size..."
    ollama pull "$OLLAMA_MODEL"
    
    log_success "Ollama installed and model downloaded"
}

# Download and extract embeddings
download_embeddings() {
    if [ "$DOWNLOAD_EMBEDDINGS" = true ]; then
        log_info "═══════════════════════════════════════════════════════════════"
        log_info "              DOWNLOADING PRE-GENERATED EMBEDDINGS             "
        log_info "═══════════════════════════════════════════════════════════════"
        echo ""
        log_warning "Download size: $EMBEDDINGS_SIZE"
        log_warning "This will take 30-60 minutes depending on your internet speed"
        log_warning "The download will run in the background"
        echo ""
        
        cd "$PROJECT_ROOT"
        
        if [ -f "$EMBEDDINGS_ZIP" ]; then
            log_warning "Embeddings zip already exists. Checking if complete..."
            # Try to extract existing file
            if unzip -t "$EMBEDDINGS_ZIP" &> /dev/null; then
                log_info "Existing zip is valid, skipping download"
            else
                log_warning "Existing zip is corrupted, removing..."
                rm -f "$EMBEDDINGS_ZIP"
            fi
        fi
        
        if [ ! -f "$EMBEDDINGS_ZIP" ]; then
            # Create download script
            cat > "$PROJECT_ROOT/download_embeddings.sh" << 'DOWNLOAD_EOF'
#!/bin/bash
EMBEDDINGS_ZIP="embeddings.zip"
EMBEDDINGS_URL="https://drive.google.com/uc?id=1OUhXyGmqrA4tfI97Ja6HZrjGOANkTXG0"
LOG_FILE="embeddings_download.log"

echo "Starting embeddings download at $(date)" | tee "$LOG_FILE"
echo "Download size: ~40GB" | tee -a "$LOG_FILE"

# Activate virtual environment
source agriir_env/bin/activate

# Try downloading with gdown (with retry logic)
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "Download attempt $((RETRY_COUNT + 1)) of $MAX_RETRIES..." | tee -a "$LOG_FILE"
    
    python3 -c "import gdown; gdown.download('$EMBEDDINGS_URL', '$EMBEDDINGS_ZIP', quiet=False, resume=True)" 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo "Download completed successfully at $(date)" | tee -a "$LOG_FILE"
        
        # Extract the zip file
        echo "Extracting embeddings..." | tee -a "$LOG_FILE"
        if command -v unzip &> /dev/null; then
            unzip -q "$EMBEDDINGS_ZIP" 2>&1 | tee -a "$LOG_FILE"
        else
            python3 -c "import zipfile; zipfile.ZipFile('$EMBEDDINGS_ZIP').extractall('.')" 2>&1 | tee -a "$LOG_FILE"
        fi
        
        if [ $? -eq 0 ]; then
            echo "Extraction completed successfully" | tee -a "$LOG_FILE"
            rm -f "$EMBEDDINGS_ZIP"
            echo "EMBEDDINGS_READY=true" > embeddings_status.txt
            exit 0
        else
            echo "Extraction failed" | tee -a "$LOG_FILE"
            exit 1
        fi
    else
        echo "Download failed, retrying in 10 seconds..." | tee -a "$LOG_FILE"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        sleep 10
    fi
done

echo "Download failed after $MAX_RETRIES attempts" | tee -a "$LOG_FILE"
echo "EMBEDDINGS_READY=false" > embeddings_status.txt
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "ALTERNATIVE DOWNLOAD METHOD:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "If the download keeps failing, you can download manually using a browser:"
echo ""
echo "1. Open this link in your browser:"
echo "   https://drive.google.com/file/d/1OUhXyGmqrA4tfI97Ja6HZrjGOANkTXG0/view?usp=drive_link"
echo ""
echo "2. Download the file to this directory:"
echo "   $(pwd)"
echo ""
echo "3. Save it as: embeddings.zip"
echo ""
echo "4. Then run this command to extract:"
echo "   unzip embeddings.zip && rm embeddings.zip"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
exit 1
DOWNLOAD_EOF
            
            chmod +x "$PROJECT_ROOT/download_embeddings.sh"
            
            # Start download in background
            log_info "Starting download in background..."
            nohup "$PROJECT_ROOT/download_embeddings.sh" > /dev/null 2>&1 &
            DOWNLOAD_PID=$!
            
            log_success "Download started (PID: $DOWNLOAD_PID)"
            log_info "Monitor progress with: tail -f embeddings_download.log"
            log_info "You can continue with the installation while download proceeds"
            echo ""
            log_warning "NOTE: If download fails, check embeddings_download.log for manual download instructions"
            echo ""
            sleep 2
        fi
    fi
}

# Generate embeddings from local dataset
generate_embeddings() {
    if [ "$GENERATE_EMBEDDINGS" = true ] && [ "$USE_EXISTING_DATASET" = true ]; then
        log_info "═══════════════════════════════════════════════════════════════"
        log_info "              GENERATING EMBEDDINGS FROM DATASET               "
        log_info "═══════════════════════════════════════════════════════════════"
        echo ""
        
        cd "$PROJECT_ROOT"
        
        # Check if dataset file exists
        if [ ! -f "$DATASET_FILE" ]; then
            log_error "Dataset file not found: $DATASET_FILE"
            log_error "Please ensure the file exists in: $PROJECT_ROOT"
            return 1
        fi
        
        log_info "Found dataset file: $DATASET_FILE"
        
        # Extract the tar.xz file
        EXTRACTED_JSONL="${DATASET_FILE%.tar.xz}.jsonl"
        
        if [ ! -f "$EXTRACTED_JSONL" ]; then
            log_info "Extracting dataset from tar.xz archive..."
            tar -xf "$DATASET_FILE"
            
            if [ $? -ne 0 ]; then
                log_error "Failed to extract dataset"
                return 1
            fi
            
            log_success "Dataset extracted: $EXTRACTED_JSONL"
        else
            log_info "Dataset already extracted: $EXTRACTED_JSONL"
        fi
        
        # Setup embedding generator
        EMBEDDING_GEN_PATH="$PROJECT_ROOT/embedding_generator"
        
        if [ ! -d "$EMBEDDING_GEN_PATH" ]; then
            log_error "Embedding generator not found at: $EMBEDDING_GEN_PATH"
            return 1
        fi
        
        # Install embedding generator requirements
        log_info "Installing embedding generator dependencies..."
        if [ -f "$EMBEDDING_GEN_PATH/requirements.txt" ]; then
            export UV_LINK_MODE=copy
            uv pip install -r "$EMBEDDING_GEN_PATH/requirements.txt"
        fi
        
        # Create embeddings output directory
        EMBEDDINGS_OUTPUT="$PROJECT_ROOT/embeddings_output"
        mkdir -p "$EMBEDDINGS_OUTPUT"
        
        # Create generation script
        cat > "$PROJECT_ROOT/generate_embeddings.sh" << GENERATE_EOF
#!/bin/bash

cd "$PROJECT_ROOT"
source "$VENV_PATH/bin/activate"

LOG_FILE="embedding_generation.log"

echo "Starting embedding generation at \$(date)" | tee "\$LOG_FILE"
echo "Input dataset: $EXTRACTED_JSONL" | tee -a "\$LOG_FILE"
echo "Output directory: $EMBEDDINGS_OUTPUT" | tee -a "\$LOG_FILE"
echo ""
echo "This will take 2-4 hours with GPU, longer with CPU..." | tee -a "\$LOG_FILE"
echo ""

# Run embedding generation
python3 "$EMBEDDING_GEN_PATH/src/create_embeddings.py" \\
    --input "$EXTRACTED_JSONL" \\
    --output "$EMBEDDINGS_OUTPUT" \\
    --device auto \\
    --chunk-size 256 \\
    2>&1 | tee -a "\$LOG_FILE"

if [ \$? -eq 0 ]; then
    echo "" | tee -a "\$LOG_FILE"
    echo "Embedding generation completed successfully at \$(date)" | tee -a "\$LOG_FILE"
    echo "EMBEDDINGS_READY=true" > embeddings_status.txt
    exit 0
else
    echo "" | tee -a "\$LOG_FILE"
    echo "Embedding generation failed" | tee -a "\$LOG_FILE"
    echo "EMBEDDINGS_READY=false" > embeddings_status.txt
    exit 1
fi
GENERATE_EOF
        
        chmod +x "$PROJECT_ROOT/generate_embeddings.sh"
        
        # Ask user if they want to start generation now or later
        echo ""
        log_info "Embedding generation script created: generate_embeddings.sh"
        log_warning "Generation will take 2-4 hours with GPU (longer with CPU)"
        echo ""
        read -p "$(echo -e ${YELLOW}Start embedding generation now? [y/N]: ${NC})" start_now
        
        case "$start_now" in
            y|Y|yes|Yes|YES)
                log_info "Starting embedding generation in background..."
                nohup "$PROJECT_ROOT/generate_embeddings.sh" > /dev/null 2>&1 &
                GEN_PID=$!
                log_success "Generation started (PID: $GEN_PID)"
                log_info "Monitor progress with: tail -f embedding_generation.log"
                ;;
            *)
                log_info "Embedding generation postponed"
                log_info "You can start it later by running: ./generate_embeddings.sh"
                ;;
        esac
    fi
}

# Setup environment file
setup_environment() {
    log_info "Setting up environment configuration..."
    
    cat > "$PROJECT_ROOT/.env" << EOF
# AgriIR Environment Configuration
# Generated on $(date)

VOICE_ENABLED=$VOICE_ENABLED
DEFAULT_VOICE_ENGINE=conformer

RAG_ENABLED=$RAG_ENABLED
RAG_MODEL=Qwen/Qwen3-Embedding-8B

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=$OLLAMA_MODEL

WEB_HOST=0.0.0.0
WEB_PORT=5000

# API Keys (set these if using external services)
# HUGGINGFACE_TOKEN=your_token_here
# SARVAM_API_KEY=your_api_key_here

LOG_LEVEL=INFO
EOF
    
    log_success "Environment file created at $PROJECT_ROOT/.env"
}

# Create startup script
create_startup_script() {
    log_info "Creating startup script..."
    
    cat > "$PROJECT_ROOT/start_agriir_bot.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/agriir_env"
AGRI_BOT_PATH="$PROJECT_ROOT/agri_bot_searcher"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}Starting AgriIR...${NC}"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}Virtual environment not found. Please run installation first.${NC}"
    exit 1
fi

source "$VENV_PATH/bin/activate"

if [ -f "$PROJECT_ROOT/.env" ]; then
    source "$PROJECT_ROOT/.env"
fi

if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama service...${NC}"
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama
    else
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
    fi
fi

cd "$AGRI_BOT_PATH"

if [ "$VOICE_ENABLED" = "true" ] && [ -f "src/enhanced_voice_web_ui.py" ]; then
    echo -e "${GREEN}🎤 Starting AgriIR with Voice Support${NC}"
    echo -e "${BLUE}Access the bot at: http://localhost:5000${NC}"
    python3 src/enhanced_voice_web_ui.py
elif [ "$RAG_ENABLED" = "true" ] && [ -f "src/enhanced_web_ui.py" ]; then
    echo -e "${GREEN}🚀 Starting AgriIR with RAG System${NC}"
    echo -e "${BLUE}Access the bot at: http://localhost:5000${NC}"
    python3 src/enhanced_web_ui.py
else
    echo -e "${GREEN}⚡ Starting AgriIR (Basic Mode)${NC}"
    echo -e "${BLUE}Access the bot at: http://localhost:5000${NC}"
    python3 src/web_ui.py
fi
EOF
    
    chmod +x "$PROJECT_ROOT/start_agriir_bot.sh"
    log_success "Startup script created: start_agriir_bot.sh"
}

# Main installation function
main() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║                  AgriIR Installer                     ║${NC}"
    echo -e "${BLUE}║              Agriculture Chatbot with Voice AI              ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    
    parse_arguments "$@"
    
    log_info "Installation configuration:"
    log_info "  Voice Features: $VOICE_ENABLED"
    log_info "  RAG System: $RAG_ENABLED"
    log_info "  Python Version: $PYTHON_VERSION"
    log_info "  Auto-detect GPU: $AUTO_DETECT_GPU"
    echo
    
    check_system_requirements
    prompt_embeddings_download
    setup_virtual_environment
    install_python_dependencies
    download_embeddings
    generate_embeddings
    install_ollama
    setup_environment
    create_startup_script
    
    echo
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                   Installation Complete!                    ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
    echo -e "${BLUE}To start AgriIR:${NC}"
    echo -e "${YELLOW}  ./start_agriir_bot.sh${NC}"
    echo
    echo -e "${BLUE}Or manually:${NC}"
    echo -e "${YELLOW}  source agriir_env/bin/activate${NC}"
    echo -e "${YELLOW}  cd agri_bot_searcher${NC}"
    echo -e "${YELLOW}  python3 src/enhanced_voice_web_ui.py${NC}"
    echo
    echo -e "${BLUE}Access the web interface at: http://localhost:5000${NC}"
    echo
    echo -e "${BLUE}Installed configuration:${NC}"
    echo -e "${YELLOW}  Python: $(python3 --version 2>&1 || echo 'N/A')${NC}"
    echo -e "${YELLOW}  Ollama Model: $OLLAMA_MODEL${NC}"
    
    if [ "$DOWNLOAD_EMBEDDINGS" = true ]; then
        echo -e "${YELLOW}  Embeddings: Downloading in background (check embeddings_download.log)${NC}"
    elif [ "$GENERATE_EMBEDDINGS" = true ]; then
        echo -e "${YELLOW}  Embeddings: Generating from local dataset (check embedding_generation.log)${NC}"
    else
        echo -e "${YELLOW}  Embeddings: Not configured (can be set up later)${NC}"
    fi
    
    echo ""
    
    if [ "$DOWNLOAD_EMBEDDINGS" = true ] || [ "$GENERATE_EMBEDDINGS" = true ]; then
        echo -e "${BLUE}Embeddings Status:${NC}"
        if [ "$DOWNLOAD_EMBEDDINGS" = true ]; then
            echo -e "${YELLOW}  - Download running in background${NC}"
            echo -e "${YELLOW}  - Monitor: tail -f embeddings_download.log${NC}"
            echo -e "${YELLOW}  - Check status: cat embeddings_status.txt${NC}"
        fi
        if [ "$GENERATE_EMBEDDINGS" = true ]; then
            echo -e "${YELLOW}  - Generation script: ./generate_embeddings.sh${NC}"
            echo -e "${YELLOW}  - Monitor: tail -f embedding_generation.log${NC}"
        fi
        echo ""
    fi
    echo
}

main "$@"
