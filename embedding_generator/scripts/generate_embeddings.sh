#!/bin/bash
# Convenience script for generating embeddings

set -e

# Default values
INPUT_FILE=""
OUTPUT_DIR="outputs/embeddings_$(date +%Y%m%d_%H%M%S)"
MAX_RECORDS=""
DEVICE="auto"
CHUNK_SIZE=256

# Help function
show_help() {
    cat << EOF
🌾 Agriculture Embedding Generator

Usage: $0 [OPTIONS] INPUT_FILE

Arguments:
    INPUT_FILE          Path to JSONL dataset file

Options:
    -o, --output DIR    Output directory (default: outputs/embeddings_TIMESTAMP)
    -m, --max-records N Maximum records to process (default: all)
    -d, --device DEVICE Device to use: auto, cuda, cpu (default: auto)
    -c, --chunk-size N  Chunk size in tokens (default: 256)
    -h, --help          Show this help message

Examples:
    $0 data/agriculture_dataset.jsonl
    $0 -o my_embeddings -m 1000 data/dataset.jsonl
    $0 --device cpu --chunk-size 512 data/dataset.jsonl

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--max-records)
            MAX_RECORDS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -c|--chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
        *)
            if [ -z "$INPUT_FILE" ]; then
                INPUT_FILE="$1"
            else
                echo "Multiple input files not supported"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate input
if [ -z "$INPUT_FILE" ]; then
    echo "❌ Error: Input file required"
    show_help
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not found: $INPUT_FILE"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ] && [ -d "venv" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Build command
CMD="python src/create_embeddings.py --input \"$INPUT_FILE\" --output \"$OUTPUT_DIR\" --device $DEVICE --chunk-size $CHUNK_SIZE"

if [ -n "$MAX_RECORDS" ]; then
    CMD="$CMD --max-records $MAX_RECORDS"
fi

echo "🌾 Agriculture Embedding Generator"
echo "=================================="
echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Chunk size: $CHUNK_SIZE"
if [ -n "$MAX_RECORDS" ]; then
    echo "Max records: $MAX_RECORDS"
fi
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run the command
echo "🚀 Starting embedding generation..."
eval $CMD

echo ""
echo "✅ Embedding generation completed!"
echo "📁 Results saved to: $OUTPUT_DIR"