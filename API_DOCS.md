# 🫁 Pneumonia Detection FastAPI Backend

**Production-ready REST API for chest X-ray pneumonia detection using deep learning models**

## 🎯 Features

- ✅ **Multi-Model Support**: PyTorch ResNet18 + TensorFlow DenseNet121
- ✅ **RESTful API**: FastAPI with auto-generated Swagger documentation
- ✅ **Batch Processing**: Predict on multiple images simultaneously
- ✅ **CORS Enabled**: Ready for frontend integration
- ✅ **CPU Optimized**: Works on standard hardware without GPU
- ✅ **Production Ready**: Proper error handling and logging
- ✅ **Fast Inference**: ~100-200ms per prediction

## 📋 Prerequisites

- Python 3.8+
- Trained models in `./models/` directory:
  - `best_model.pth` (PyTorch ResNet18)
  - `pneumonia_detection_model.keras` (TensorFlow DenseNet121)

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run the Server

```bash
python app.py
```

Server will start at: `http://localhost:8000`

### 3️⃣ Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Check if API is running and models are available.

**Response:**
```json
{
  "status": "healthy",
  "models_available": {
    "pytorch_resnet18": true,
    "tensorflow_densenet121": true
  },
  "device": "cpu"
}
```

### Single Prediction - PyTorch Model
```bash
POST /predict/pytorch
```
Predict using PyTorch ResNet18 model.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/pytorch" \
  -F "file=@chest_xray.png"
```

**Response:**
```json
{
  "filename": "chest_xray.png",
  "prediction": "NORMAL",
  "confidence": 0.9543,
  "class_0_prob": 0.9543,
  "class_1_prob": 0.0457,
  "model_used": "PyTorch ResNet18",
  "processing_time_ms": 127.45
}
```

### Single Prediction - TensorFlow Model
```bash
POST /predict/tensorflow
```
Predict using TensorFlow DenseNet121 model.

**Request:**
```bash
curl -X POST "http://localhost:8000/predict/tensorflow" \
  -F "file=@chest_xray.png"
```

**Response:**
```json
{
  "filename": "chest_xray.png",
  "prediction": "PNEUMONIA",
  "confidence": 0.8762,
  "class_0_prob": 0.1238,
  "class_1_prob": 0.8762,
  "model_used": "TensorFlow DenseNet121",
  "processing_time_ms": 156.23
}
```

### Batch Prediction
```bash
POST /batch-predict?model_type=pytorch
```
Predict on multiple images at once.

**Request:**
```bash
curl -X POST "http://localhost:8000/batch-predict?model_type=pytorch" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png"
```

**Response:**
```json
{
  "total_files": 3,
  "successful": 3,
  "failed": 0,
  "predictions": [
    {
      "filename": "image1.png",
      "prediction": "NORMAL",
      "confidence": 0.9234,
      ...
    },
    ...
  ],
  "errors": []
}
```

### Model Info
```bash
GET /info
```
Get information about available models.

**Response:**
```json
{
  "available_models": {
    "pytorch_resnet18": true,
    "tensorflow_densenet121": true
  },
  "model_paths": {
    "pytorch": "/path/to/best_model.pth",
    "tensorflow": "/path/to/pneumonia_detection_model.keras"
  },
  "image_size": [128, 128],
  "class_names": {
    "0": "NORMAL",
    "1": "PNEUMONIA"
  },
  "device": "cpu"
}
```

## 🖥️ Using the Python Client

The included `client.py` provides a convenient Python interface:

### Check API Health
```bash
python client.py --health
```

### Single Prediction
```bash
python client.py --image path/to/xray.png --model pytorch
```

### Batch Prediction
```bash
python client.py --batch image1.png image2.png image3.png --model tensorflow
```

### Get Model Info
```bash
python client.py --info
```

### Connect to Remote Server
```bash
python client.py --url http://your-server.com:8000 --health
```

## 🐍 Python Usage Example

```python
import requests

# Single prediction
with open('chest_xray.png', 'rb') as f:
    files = {'file': f}
    response = requests.post(
        'http://localhost:8000/predict/pytorch',
        files=files
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
```

## 📦 Deployment

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .
COPY models/ models/

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t pneumonia-api .
docker run -p 8000:8000 pneumonia-api
```

### Gunicorn Production Server
```bash
pip install gunicorn

gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

### Nginx Reverse Proxy (Optional)
```nginx
upstream api {
    server localhost:8000;
}

server {
    listen 80;
    server_name pneumonia-api.example.com;

    location / {
        proxy_pass http://api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Model Specifications

### PyTorch ResNet18
- **Architecture**: ResNet18 with custom 2-class head
- **Input**: 128×128×3 RGB image
- **Output**: Binary classification (NORMAL / PNEUMONIA)
- **Framework**: PyTorch
- **Device**: CPU
- **Speed**: ~100-150ms inference

### TensorFlow DenseNet121
- **Architecture**: DenseNet121 pretrained on ImageNet
- **Input**: 128×128×3 RGB image
- **Output**: Binary classification (NORMAL / PNEUMONIA)
- **Framework**: TensorFlow/Keras
- **Device**: CPU
- **Speed**: ~150-200ms inference

## 🔧 Configuration

Edit `app.py` to customize:

```python
# Model directory
MODEL_DIR = Path(__file__).parent / "models"

# Compute device (CPU by default)
DEVICE = torch.device('cpu')  # Change to 'cuda' for GPU

# Class names
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

# Image preprocessing
IMAGE_SIZE = (128, 128)
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406])
NORMALIZE_STD = np.array([0.229, 0.224, 0.225])
```

## 🐛 Troubleshooting

### Model Not Found
```
Error: FileNotFoundError: PyTorch model not found at...
```
**Solution**: Ensure trained models are in `./models/` directory with correct names.

### Port Already in Use
```bash
# Use different port
python -c "from app import app; import uvicorn; uvicorn.run(app, host='0.0.0.0', port=8001)"
```

### Connection Refused
```
ConnectionError: Could not connect to API
```
**Solution**: Make sure server is running (`python app.py`) before running client.

### Out of Memory
**Solution**: 
- Reduce batch size in `/batch-predict`
- Use smaller image preprocessing size
- Run on CPU (already configured)

## 📈 Performance Metrics

- **Throughput**: ~500-600 requests/min (single-threaded)
- **Latency (PyTorch)**: 100-150ms per image
- **Latency (TensorFlow)**: 150-200ms per image
- **Memory Usage**: ~800MB at startup
- **CPU**: Optimized for multi-core systems

## 🔒 Security Notes

- API is open (CORS enabled) for development
- **For production**: Add authentication, rate limiting, HTTPS
- Monitor model predictions for drift
- Log all predictions for audit trail

## 📚 API Testing

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict/pytorch" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.png"
```

### Using Python requests
```python
import requests

# Test health
response = requests.get('http://localhost:8000/health')
print(response.json())

# Test prediction
files = {'file': open('image.png', 'rb')}
response = requests.post('http://localhost:8000/predict/pytorch', files=files)
print(response.json())
```

### Using Swagger UI
Navigate to http://localhost:8000/docs and use interactive interface to test all endpoints.

## 📝 License

This project is provided as-is for educational and demonstration purposes.

## 🤝 Support

For issues or questions, refer to project documentation or contact the development team.

---

**Made with ❤️ for medical AI** | Pneumonia Detection Project 2024
