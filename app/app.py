"""
Pneumonia Detection FastAPI Backend
Serves chest X-ray predictions using trained deep learning models.
"""

import os
import cv2
import torch
import json
import base64
import numpy as np
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, List
from PIL import Image
import tensorflow as tf

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = Path(__file__).parent / "models"
DEVICE = torch.device('cpu')
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

# Ensure models directory exists
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# Pydantic Models (API Schemas)
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    models_available: Dict[str, bool]
    device: str


class PredictionResponse(BaseModel):
    filename: str
    prediction: str
    confidence: float
    class_0_prob: float
    class_1_prob: float
    model_used: str
    processing_time_ms: float


class BatchPredictionResponse(BaseModel):
    total_files: int
    successful: int
    failed: int
    predictions: List[PredictionResponse]
    errors: List[str]


# ============================================================================
# Model Loaders
# ============================================================================

class ModelManager:
    """Manages loading and caching of trained models"""
    
    def __init__(self):
        self.pytorch_model = None
        self.tensorflow_model = None
        self.pytorch_model_path = MODEL_DIR / "best_model.pth"
        self.tensorflow_model_path = MODEL_DIR / "pneumonia_detection_model.keras"
    
    def load_pytorch_model(self):
        """Load ResNet18 PyTorch model"""
        try:
            if self.pytorch_model is not None:
                return self.pytorch_model
            
            if not self.pytorch_model_path.exists():
                raise FileNotFoundError(f"PyTorch model not found at {self.pytorch_model_path}")
            
            from torchvision import models
            model = models.resnet18(pretrained=False)
            model.fc = torch.nn.Linear(model.fc.in_features, 2)
            model.load_state_dict(torch.load(self.pytorch_model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            self.pytorch_model = model
            print(f"✅ PyTorch ResNet18 model loaded from {self.pytorch_model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading PyTorch model: {e}")
            return None
    
    def load_tensorflow_model(self):
        """Load DenseNet121 TensorFlow model"""
        try:
            if self.tensorflow_model is not None:
                return self.tensorflow_model
            
            if not self.tensorflow_model_path.exists():
                raise FileNotFoundError(f"TensorFlow model not found at {self.tensorflow_model_path}")
            
            model = tf.keras.models.load_model(self.tensorflow_model_path)
            self.tensorflow_model = model
            print(f"✅ TensorFlow DenseNet121 model loaded from {self.tensorflow_model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading TensorFlow model: {e}")
            return None
    
    def get_models_status(self) -> Dict[str, bool]:
        """Check which models are available"""
        return {
            "pytorch_resnet18": self.pytorch_model_path.exists(),
            "tensorflow_densenet121": self.tensorflow_model_path.exists(),
        }


# ============================================================================
# Image Processing
# ============================================================================

class ImageProcessor:
    """Handles image loading and preprocessing"""
    
    IMAGE_SIZE = (128, 128)
    NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406])
    NORMALIZE_STD = np.array([0.229, 0.224, 0.225])
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """Load image from bytes"""
        try:
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            return np.array(image)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def preprocess_pytorch(image_array: np.ndarray) -> torch.Tensor:
        """Preprocess for PyTorch model (ResNet18)"""
        # Resize
        image = cv2.resize(image_array, ImageProcessor.IMAGE_SIZE)
        
        # Convert to tensor (0-1 range)
        image_tensor = torch.from_numpy(image).float() / 255.0
        
        # Normalize (channels last to channels first)
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        for i in range(3):
            image_tensor[i] = (image_tensor[i] - ImageProcessor.NORMALIZE_MEAN[i]) / ImageProcessor.NORMALIZE_STD[i]
        
        return image_tensor.unsqueeze(0).to(DEVICE)  # Add batch dimension
    
    @staticmethod
    def preprocess_tensorflow(image_array: np.ndarray) -> np.ndarray:
        """Preprocess for TensorFlow model (DenseNet)"""
        # Resize
        image = cv2.resize(image_array, ImageProcessor.IMAGE_SIZE)
        
        # Convert to numpy float (0-1 range)
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - ImageProcessor.NORMALIZE_MEAN) / ImageProcessor.NORMALIZE_STD
        
        # Add batch dimension
        return np.expand_dims(image, axis=0)


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="Pneumonia Detection API",
    description="Deep Learning Backend for Chest X-Ray Pneumonia Detection",
    version="1.0.0",
)

# Add CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
model_manager = ModelManager()
image_processor = ImageProcessor()

# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("\n" + "="*60)
    print("🚀 Pneumonia Detection API Starting Up")
    print("="*60)
    
    try:
        model_manager.load_pytorch_model()
        model_manager.load_tensorflow_model()
        print("✅ All available models loaded successfully\n")
    except Exception as e:
        print(f"⚠️ Error during startup: {e}\n")


@app.get("/", tags=["Root"])
def read_root():
    """API root endpoint with information"""
    return {
        "name": "Pneumonia Detection API",
        "version": "1.0.0",
        "description": "Chest X-Ray classification using deep learning",
        "endpoints": {
            "health": "/health",
            "predict_pytorch": "/predict/pytorch",
            "predict_tensorflow": "/predict/tensorflow",
            "batch_predict": "/batch-predict",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check API health and model availability"""
    return HealthResponse(
        status="healthy",
        models_available=model_manager.get_models_status(),
        device=str(DEVICE),
    )


@app.post("/predict/pytorch", response_model=PredictionResponse, tags=["Predictions"])
async def predict_pytorch(file: UploadFile = File(...)):
    """
    Predict pneumonia using PyTorch ResNet18 model
    
    - **file**: Chest X-ray image (JPG, PNG)
    - Returns: Prediction with confidence scores
    """
    import time
    start_time = time.time()
    
    try:
        # Load model
        model = model_manager.load_pytorch_model()
        if model is None:
            raise HTTPException(status_code=503, detail="PyTorch model not available")
        
        # Read image
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image_array = image_processor.load_image_from_bytes(image_bytes)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Preprocess
        image_tensor = image_processor.preprocess_pytorch(image_array)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
            class_0_prob = probabilities[0, 0].item()
            class_1_prob = probabilities[0, 1].item()
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            filename=file.filename,
            prediction=CLASS_NAMES[pred_class],
            confidence=round(confidence, 4),
            class_0_prob=round(class_0_prob, 4),
            class_1_prob=round(class_1_prob, 4),
            model_used="PyTorch ResNet18",
            processing_time_ms=round(processing_time, 2),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/tensorflow", response_model=PredictionResponse, tags=["Predictions"])
async def predict_tensorflow(file: UploadFile = File(...)):
    """
    Predict pneumonia using TensorFlow DenseNet121 model
    
    - **file**: Chest X-ray image (JPG, PNG)
    - Returns: Prediction with confidence scores
    """
    import time
    start_time = time.time()
    
    try:
        # Load model
        model = model_manager.load_tensorflow_model()
        if model is None:
            raise HTTPException(status_code=503, detail="TensorFlow model not available")
        
        # Read image
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        image_array = image_processor.load_image_from_bytes(image_bytes)
        if image_array is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        # Preprocess
        image_data = image_processor.preprocess_tensorflow(image_array)
        
        # Predict
        predictions = model.predict(image_data, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0, pred_class])
        class_0_prob = float(predictions[0, 0])
        class_1_prob = float(predictions[0, 1])
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            filename=file.filename,
            prediction=CLASS_NAMES[pred_class],
            confidence=round(confidence, 4),
            class_0_prob=round(class_0_prob, 4),
            class_1_prob=round(class_1_prob, 4),
            model_used="TensorFlow DenseNet121",
            processing_time_ms=round(processing_time, 2),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch-predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict(
    files: List[UploadFile] = File(...),
    model_type: str = "pytorch"
):
    """
    Batch predict multiple chest X-ray images
    
    - **files**: List of X-ray images
    - **model_type**: "pytorch" or "tensorflow"
    - Returns: Batch predictions with success/failure counts
    """
    predictions = []
    errors = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            if not image_bytes:
                errors.append(f"{file.filename}: Empty file")
                continue
            
            image_array = image_processor.load_image_from_bytes(image_bytes)
            if image_array is None:
                errors.append(f"{file.filename}: Failed to load image")
                continue
            
            if model_type == "pytorch":
                model = model_manager.load_pytorch_model()
                if model is None:
                    errors.append(f"{file.filename}: PyTorch model unavailable")
                    continue
                
                image_tensor = image_processor.preprocess_pytorch(image_array)
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    pred_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0, pred_class].item()
                    class_0_prob = probabilities[0, 0].item()
                    class_1_prob = probabilities[0, 1].item()
                
                model_used = "PyTorch ResNet18"
            
            else:  # tensorflow
                model = model_manager.load_tensorflow_model()
                if model is None:
                    errors.append(f"{file.filename}: TensorFlow model unavailable")
                    continue
                
                image_data = image_processor.preprocess_tensorflow(image_array)
                preds = model.predict(image_data, verbose=0)
                pred_class = np.argmax(preds[0])
                confidence = float(preds[0, pred_class])
                class_0_prob = float(preds[0, 0])
                class_1_prob = float(preds[0, 1])
                
                model_used = "TensorFlow DenseNet121"
            
            predictions.append(PredictionResponse(
                filename=file.filename,
                prediction=CLASS_NAMES[pred_class],
                confidence=round(confidence, 4),
                class_0_prob=round(class_0_prob, 4),
                class_1_prob=round(class_1_prob, 4),
                model_used=model_used,
                processing_time_ms=0.0,
            ))
        
        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")
    
    return BatchPredictionResponse(
        total_files=len(files),
        successful=len(predictions),
        failed=len(errors),
        predictions=predictions,
        errors=errors,
    )


@app.get("/info", tags=["Info"])
def model_info():
    """Get information about available models"""
    return {
        "available_models": model_manager.get_models_status(),
        "model_paths": {
            "pytorch": str(model_manager.pytorch_model_path),
            "tensorflow": str(model_manager.tensorflow_model_path),
        },
        "image_size": ImageProcessor.IMAGE_SIZE,
        "class_names": CLASS_NAMES,
        "device": str(DEVICE),
    }


# ============================================================================
# Execution
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
