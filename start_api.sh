#!/bin/bash
# Startup script for Pneumonia Detection FastAPI Backend

set -e

echo ""
echo "==============================================="
echo "🫁 Pneumonia Detection API Startup"
echo "==============================================="
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found"
    echo "   Creating new virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install/update requirements
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Check if models exist
if [ ! -f "models/best_model.pth" ] && [ ! -f "models/pneumonia_detection_model.keras" ]; then
    echo "❌ No trained models found in ./models/"
    echo "   Please run the training notebooks first:"
    echo "   - notebooks/pneumonia-detection-with-resnet18-90-06-accuracy.ipynb"
    echo "   - notebooks/densenet.ipynb"
    exit 1
fi

echo "✅ Models found:"
[ -f "models/best_model.pth" ] && echo "   ✓ PyTorch ResNet18"
[ -f "models/pneumonia_detection_model.keras" ] && echo "   ✓ TensorFlow DenseNet121"

echo ""
echo "==============================================="
echo "🚀 Starting API Server"
echo "==============================================="
echo ""
echo "📍 Server running at: http://localhost:8000"
echo "📚 API Docs at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the app
python app/app.py
