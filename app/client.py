"""
Pneumonia Detection API Client
Demo script showing how to use the FastAPI backend
"""

import requests
import argparse
from pathlib import Path
from typing import Optional

# API Base URL
BASE_URL = "http://localhost:8000"


class PneumoniaClient:
    """Client for interacting with Pneumonia Detection API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """Check if API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_pytorch(self, image_path: str) -> dict:
        """Get prediction using PyTorch model"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/predict/pytorch", files=files)
            response.raise_for_status()
            return response.json()
    
    def predict_tensorflow(self, image_path: str) -> dict:
        """Get prediction using TensorFlow model"""
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{self.base_url}/predict/tensorflow", files=files)
            response.raise_for_status()
            return response.json()
    
    def predict_batch(self, image_paths: list, model_type: str = "pytorch") -> dict:
        """Get predictions for multiple images"""
        files = [('files', open(path, 'rb')) for path in image_paths]
        params = {'model_type': model_type}
        response = requests.post(
            f"{self.base_url}/batch-predict",
            files=files,
            params=params
        )
        response.raise_for_status()
        
        # Close files
        for _, file in files:
            file.close()
        
        return response.json()
    
    def get_info(self) -> dict:
        """Get model information"""
        response = requests.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()


def print_prediction(result: dict):
    """Pretty print prediction result"""
    print("\n" + "="*60)
    print(f"📄 File: {result['filename']}")
    print(f"🤖 Model: {result['model_used']}")
    print("-" * 60)
    print(f"🔍 Prediction: {result['prediction']}")
    print(f"📊 Confidence: {result['confidence']*100:.2f}%")
    print("-" * 60)
    print(f"   NORMAL prob:     {result['class_0_prob']*100:.2f}%")
    print(f"   PNEUMONIA prob:  {result['class_1_prob']*100:.2f}%")
    print("-" * 60)
    print(f"⏱️  Processing time: {result['processing_time_ms']:.2f}ms")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pneumonia Detection API Client"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check API health"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Get model information"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to chest X-ray image for prediction"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="Model to use for prediction"
    )
    parser.add_argument(
        "--batch",
        nargs='+',
        help="Paths to multiple images for batch prediction"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=BASE_URL,
        help=f"API base URL (default: {BASE_URL})"
    )
    
    args = parser.parse_args()
    
    client = PneumoniaClient(args.url)
    
    try:
        # Health check
        if args.health:
            print("\n🏥 Checking API health...")
            health = client.health_check()
            print(f"✅ Status: {health['status']}")
            print(f"📍 Device: {health['device']}")
            print("📦 Models available:")
            for model_name, available in health['models_available'].items():
                status = "✅" if available else "❌"
                print(f"   {status} {model_name}")
            return
        
        # Get info
        if args.info:
            print("\n📋 Model Information:")
            info = client.get_info()
            print(f"Image size: {info['image_size']}")
            print(f"Classes: {info['class_names']}")
            print(f"Device: {info['device']}")
            return
        
        # Single prediction
        if args.image:
            if not Path(args.image).exists():
                print(f"❌ Image not found: {args.image}")
                return
            
            print(f"\n🔄 Predicting with {args.model} model...")
            if args.model == "pytorch":
                result = client.predict_pytorch(args.image)
            else:
                result = client.predict_tensorflow(args.image)
            
            print_prediction(result)
            return
        
        # Batch prediction
        if args.batch:
            print(f"\n🔄 Batch prediction for {len(args.batch)} images...")
            result = client.predict_batch(args.batch, args.model)
            
            print(f"\n✅ Successful: {result['successful']}/{result['total_files']}")
            if result['failed'] > 0:
                print(f"❌ Failed: {result['failed']}")
                for error in result['errors']:
                    print(f"   - {error}")
            
            for pred in result['predictions']:
                print_prediction(pred)
            return
        
        # Default: show usage
        parser.print_help()
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to API at {args.url}")
        print("Make sure the server is running: python app.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ API Error: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
