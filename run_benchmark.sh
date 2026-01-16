#!/bin/bash
# AgriIR Benchmark Runner Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_PATH="$PROJECT_ROOT/agriir_env"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  AgriIR - Benchmark Runner                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check virtual environment
if [ ! -d "$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found at: $VENV_PATH${NC}"
    echo -e "${YELLOW}Please run installation first.${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}🔧 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Load environment variables if they exist
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo -e "${GREEN}📋 Loading environment variables...${NC}"
    source "$PROJECT_ROOT/.env"
fi

# Check if Ollama is running
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not running. Starting Ollama service...${NC}"
    if command -v systemctl &> /dev/null; then
        sudo systemctl start ollama
        sleep 2
    else
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 3
    fi
    
    if pgrep -x "ollama" > /dev/null; then
        echo -e "${GREEN}✅ Ollama service started${NC}"
    else
        echo -e "${RED}❌ Failed to start Ollama service${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Ollama service is running${NC}"
fi

# Default values
INPUT_CSV="${INPUT_CSV:-benchmark/final_agri_query.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-benchmark/benchmark_results_$(date +%Y%m%d_%H%M%S).csv}"
WEB_RESULTS="${WEB_RESULTS:-8}"
DB_RESULTS="${DB_RESULTS:-3}"
OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:27b}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_CSV="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_CSV="$2"
            shift 2
            ;;
        -w|--web-results)
            WEB_RESULTS="$2"
            shift 2
            ;;
        -d|--db-results)
            DB_RESULTS="$2"
            shift 2
            ;;
        -m|--model)
            OLLAMA_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -i, --input FILE        Input CSV file (default: benchmark/final_agri_query.csv)"
            echo "  -o, --output FILE       Output CSV file (default: benchmark/benchmark_results_TIMESTAMP.csv)"
            echo "  -w, --web-results N     Number of web search results (default: 8)"
            echo "  -d, --db-results N      Number of database search results (default: 3)"
            echo "  -m, --model MODEL       Ollama model to use (default: gemma3:27b)"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Run with defaults"
            echo "  $0 -w 10 -d 5                        # Custom search results"
            echo "  $0 -m gemma3:27b                     # Use different model"
            echo "  $0 -i custom.csv -o results.csv      # Custom input/output files"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Display configuration
echo ""
echo -e "${BLUE}📊 Benchmark Configuration:${NC}"
echo -e "   Input CSV:         ${GREEN}$INPUT_CSV${NC}"
echo -e "   Output CSV:        ${GREEN}$OUTPUT_CSV${NC}"
echo -e "   Web Results:       ${GREEN}$WEB_RESULTS${NC}"
echo -e "   Database Results:  ${GREEN}$DB_RESULTS${NC}"
echo -e "   Ollama Model:      ${GREEN}$OLLAMA_MODEL${NC}"
echo ""

# Check if input file exists
if [ ! -f "$INPUT_CSV" ]; then
    echo -e "${RED}❌ Input file not found: $INPUT_CSV${NC}"
    exit 1
fi

# Create benchmark output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_CSV")
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}🚀 Starting benchmark...${NC}"
echo -e "${YELLOW}⏱️  This may take a while depending on the number of questions${NC}"
echo ""

# Run the benchmark
cd "$PROJECT_ROOT"
python3 run_benchmark.py \
    --input "$INPUT_CSV" \
    --output "$OUTPUT_CSV" \
    --web-results "$WEB_RESULTS" \
    --db-results "$DB_RESULTS" \
    --model "$OLLAMA_MODEL"

BENCHMARK_EXIT_CODE=$?

if [ $BENCHMARK_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  ✅ Benchmark Completed!                       ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${BLUE}📁 Results saved to: ${GREEN}$OUTPUT_CSV${NC}"
    echo ""
    
    # Show quick stats if file exists
    if [ -f "$OUTPUT_CSV" ]; then
        TOTAL_LINES=$(wc -l < "$OUTPUT_CSV")
        TOTAL_QUESTIONS=$((TOTAL_LINES - 1))  # Subtract header
        echo -e "${BLUE}📊 Quick Stats:${NC}"
        echo -e "   Total questions processed: ${GREEN}$TOTAL_QUESTIONS${NC}"
        
        # Check for any errors
        ERROR_COUNT=$(grep -c '"status": "error"' "$OUTPUT_CSV" 2>/dev/null || echo "0")
        SUCCESS_COUNT=$((TOTAL_QUESTIONS - ERROR_COUNT))
        
        echo -e "   Successful: ${GREEN}$SUCCESS_COUNT${NC}"
        if [ $ERROR_COUNT -gt 0 ]; then
            echo -e "   Failed: ${RED}$ERROR_COUNT${NC}"
        else
            echo -e "   Failed: ${GREEN}$ERROR_COUNT${NC}"
        fi
    fi
else
    echo ""
    echo -e "${RED}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                  ❌ Benchmark Failed!                          ║${NC}"
    echo -e "${RED}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}Check the logs above for error details${NC}"
    exit $BENCHMARK_EXIT_CODE
fi
