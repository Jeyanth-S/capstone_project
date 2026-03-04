# 🫁 Pneumonia Detection from Chest X-Rays

**Production-Ready Deep Learning System for Automated Pneumonia Detection**

A comprehensive machine learning project combining state-of-the-art deep learning models with a REST API backend for pneumonia detection from chest X-ray images.

## 🎯 Project Overview

This capstone project implements and deploys two high-performance deep learning models:
- **PyTorch ResNet18**: 90%+ accuracy
- **TensorFlow DenseNet121**: 92%+ accuracy

Both models are optimized for CPU inference and accessible via a production-ready FastAPI backend.

## 📊 Dataset

- **Source**: Kaggle - [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Size**: ~5,800 images
- **Classes**: 2 (NORMAL / PNEUMONIA)
- **Split**: Train / Validation / Test
- **Auto-download**: Run `python dataset.py`

## 📁 Project Structure

```
capstone_project/
├── notebooks/                          # Jupyter notebooks
│   ├── pneumonia-detection-with-resnet18-90-06-accuracy.ipynb
│   ├── densenet.ipynb
│   └── chest-x-ray-images-pneumonia.ipynb
│
├── app.py                              # FastAPI backend (🆕)
├── client.py                           # API client demo (🆕)
├── test_api.py                         # API test suite (🆕)
├── start_api.sh                        # API startup script (🆕)
├── API_DOCS.md                         # API documentation (🆕)
│
├── models/                             # Trained models
│   ├── best_model.pth                  # PyTorch ResNet18
│   └── pneumonia_detection_model.keras # TensorFlow DenseNet121
│
├── datasets/                           # Chest X-ray data
│   └── chest_xray/                     # Auto-downloaded
│
├── dataset.py                          # Dataset downloader
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## 🚀 Quick Start

### 1. Clone & Setup

```bash
cd capstone_project
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python dataset.py
```

### 3. Train Models (Optional)

Run Jupyter notebooks:
```bash
jupyter notebook notebooks/
```

- `pneumonia-detection-with-resnet18-90-06-accuracy.ipynb` - PyTorch ResNet18
- `densenet.ipynb` - TensorFlow DenseNet121

### 4. Start API Server

```bash
python app.py
```

Or use the startup script:
```bash
./start_api.sh
```

Server available at: `http://localhost:8000`

## 🔌 FastAPI Backend

### Quick API Test

```bash
# Health check
curl http://localhost:8000/health

# Prediction (PyTorch)
curl -X POST "http://localhost:8000/predict/pytorch" \
  -F "file=@chest_xray.png"

# Try with Python client
python client.py --health
python client.py --image path/to/xray.png --model pytorch
```

### API Documentation

- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **Full API Guide**: See [API_DOCS.md](API_DOCS.md)

### Main Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check API status & model availability |
| POST | `/predict/pytorch` | Predict with PyTorch ResNet18 |
| POST | `/predict/tensorflow` | Predict with TensorFlow DenseNet121 |
| POST | `/batch-predict` | Batch prediction on multiple images |
| GET | `/info` | Get model info & configuration |

### Python Example

```python
import requests

# Single prediction
with open('xray.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/pytorch',
        files={'file': f}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

### Testing

Run the complete test suite:
```bash
python test_api.py
```

This will:
- ✅ Check API connectivity
- ✅ Verify model availability
- ✅ Test single predictions (PyTorch & TensorFlow)
- ✅ Test batch predictions
- ✅ Report timing statistics

## 📈 Model Performance

### PyTorch ResNet18
- **Architecture**: ResNet18 with custom 2-class head
- **Accuracy**: 90.06%
- **Inference**: ~100-150ms/image
- **File**: `models/best_model.pth`

### TensorFlow DenseNet121
- **Architecture**: DenseNet121 (pretrained ImageNet)
- **Accuracy**: 92%+
- **Inference**: ~150-200ms/image
- **File**: `models/pneumonia_detection_model.keras`

## 🔧 Configuration

### Models
- Location: `./models/`
- Both models are CPU-optimized
- To use GPU: Edit `app.py` for device configuration

### Image Processing
- **Input Size**: 128×128 pixels
- **Normalization**: ImageNet standard (mean, std)
- **Supported Formats**: PNG, JPG, JPEG

### API Server
- **Default Port**: 8000
- **Workers**: 1 (single-threaded)
- **For Production**: Use Gunicorn with multiple workers

## 📦 Dependencies

Core packages:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `tensorflow` - Deep learning
- `torch` - Deep learning
- `torchvision` - Computer vision
- `opencv-python` - Image processing
- `requests` - HTTP client
- `pillow` - Image handling
- `numpy`, `pandas`, `scikit-learn` - Data science

See `requirements.txt` for complete list.

## 🐳 Docker Deployment

Build image:
```bash
docker build -t pneumonia-api .
```

Run container:
```bash
docker run -p 8000:8000 pneumonia-api
```

## 🌐 Production Deployment

### Gunicorn + Uvicorn

```bash
pip install gunicorn

gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### With Nginx Reverse Proxy

```nginx
upstream api {
    server localhost:8000;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 30s;
    }
}
```

## 🔒 Security Considerations

For production deployment:
- ✅ Add API authentication (JWT tokens)
- ✅ Enable HTTPS/SSL
- ✅ Add rate limiting
- ✅ Configure CORS properly
- ✅ Add request validation
- ✅ Monitor predictions for model drift
- ✅ Log audit trail

## 📊 Monitoring & Logging

The API provides:
- Request/response logging
- Processing time tracking
- Error reporting with details
- Model availability status

## 🧪 Testing & Validation

### API Integration Tests
```bash
python test_api.py
```

### Model Validation
See notebooks for evaluation sections

## 📚 Notebooks

### 1. ResNet18 (PyTorch)
**File**: `notebooks/pneumonia-detection-with-resnet18-90-06-accuracy.ipynb`
- Dataset loading & exploration
- Model architecture & training
- Early stopping & validation
- Performance evaluation
- Grad-CAM visualization

### 2. DenseNet121 (TensorFlow)
**File**: `notebooks/densenet.ipynb`
- TensorFlow/Keras implementation
- Transfer learning approach
- Data augmentation
- Model evaluation

### 3. Comprehensive Pipeline
**File**: `notebooks/chest-x-ray-images-pneumonia.ipynb`
- Large-scale training pipeline
- Multiple model comparison
- Advanced visualization

## 🐛 Troubleshooting

### API Won't Start
```bash
lsof -i :8000  # Find process using port
kill -9 <PID>  # Kill it
```

### Model Not Found
Run training notebooks or ensure models are in `./models/`

### Connection Refused
Ensure `python app.py` is running

## 📞 Support

For issues:
1. Check [API_DOCS.md](API_DOCS.md)
2. Review notebook comments
3. Check API logs
4. Verify dataset is downloaded

## 🚀 Quick Start Command

```bash
# Complete setup in 4 commands
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && python dataset.py
python app.py &
python test_api.py
```

Then visit: **http://localhost:8000/docs**

---

**Made with ❤️ for Medical AI** | Pneumonia Detection Capstone 2024