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
