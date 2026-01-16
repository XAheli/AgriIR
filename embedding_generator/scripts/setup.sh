#!/bin/bash
# Setup script for Agriculture Embedding Generator

set -e  # Exit on any error

echo "🌾 Agriculture Embedding Generator Setup"
echo "========================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check CUDA availability
echo "🔍 Checking CUDA availability..."
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Install GPU version of FAISS if CUDA is available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "🚀 CUDA detected. Installing GPU-optimized FAISS..."
    pip uninstall -y faiss-cpu
    pip install faiss-gpu
else
    echo "💻 No CUDA detected. Using CPU version of FAISS."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p logs

# Make scripts executable
chmod +x scripts/*.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Place your JSONL dataset in the 'data/' directory"
echo "3. Run: python src/create_embeddings.py --input data/your_dataset.jsonl"
echo ""
echo "For help: python src/create_embeddings.py --help"