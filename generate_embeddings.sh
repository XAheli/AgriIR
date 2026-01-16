#!/bin/bash

cd "/store/testing/final/new/AgriIR"
source "/store/testing/final/new/AgriIR/agriir_env/bin/activate"

LOG_FILE="embedding_generation.log"

echo "Starting embedding generation at $(date)" | tee "$LOG_FILE"
echo "Input dataset: autonomous_indian_agriculture_complete_repaired.jsonl.jsonl" | tee -a "$LOG_FILE"
echo "Output directory: /store/testing/final/new/AgriIR/embeddings_output" | tee -a "$LOG_FILE"
echo ""
echo "This will take 2-4 hours with GPU, longer with CPU..." | tee -a "$LOG_FILE"
echo ""

# Run embedding generation
python3 "/store/testing/final/new/AgriIR/embedding_generator/src/create_embeddings.py" \
    --input "autonomous_indian_agriculture_complete_repaired.jsonl.jsonl" \
    --output "/store/testing/final/new/AgriIR/embeddings_output" \
    --device auto \
    --chunk-size 256 \
    2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Embedding generation completed successfully at $(date)" | tee -a "$LOG_FILE"
    echo "EMBEDDINGS_READY=true" > embeddings_status.txt
    exit 0
else
    echo "" | tee -a "$LOG_FILE"
    echo "Embedding generation failed" | tee -a "$LOG_FILE"
    echo "EMBEDDINGS_READY=false" > embeddings_status.txt
    exit 1
fi
