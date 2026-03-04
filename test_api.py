#!/usr/bin/env python
"""
Demo script: Test the FastAPI backend with sample images from the dataset
"""

import os
import sys
import time
import requests
from pathlib import Path
from typing import List

# Configuration
API_BASE_URL = "http://localhost:8000"
DATASET_PATH = Path(__file__).parent / "datasets" / "chest_xray" / "chest_xray"
TEST_IMAGES_COUNT = 5  # Number of images to test


def find_test_images(limit: int = TEST_IMAGES_COUNT) -> List[str]:
    """Find chest X-ray images from the dataset"""
    images = []
    
    # Look for NORMAL images first
    normal_dir = DATASET_PATH / "test" / "NORMAL"
    if normal_dir.exists():
        normal_images = list(normal_dir.glob("*.png"))[:limit//2]
        images.extend([str(p) for p in normal_images])
    
    # Then PNEUMONIA images
    pneumonia_dir = DATASET_PATH / "test" / "PNEUMONIA"
    if pneumonia_dir.exists():
        pneumonia_images = list(pneumonia_dir.glob("*.png"))[:limit - len(images)]
        images.extend([str(p) for p in pneumonia_images])
    
    return images[:limit]


def test_health():
    """Test health endpoint"""
    print("\n🏥 Testing Health Endpoint...")
    print("-" * 60)
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        
        print(f"✅ Status: {health['status']}")
        print(f"📍 Device: {health['device']}")
        print("📦 Models Available:")
        for model_name, available in health['models_available'].items():
            status = "✅" if available else "❌"
            print(f"   {status} {model_name}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n📋 Testing Model Info Endpoint...")
    print("-" * 60)
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        response.raise_for_status()
        info = response.json()
        
        print(f"🖼️  Image Size: {info['image_size']}")
        print(f"🏷️  Classes: {info['class_names']}")
        print(f"📍 Device: {info['device']}")
        return True
    except Exception as e:
        print(f"❌ Model info failed: {e}")
        return False


def test_single_prediction(image_path: str, model_type: str = "pytorch"):
    """Test single prediction endpoint"""
    endpoint = f"/predict/{model_type}"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}{endpoint}",
                files=files,
                timeout=30
            )
            elapsed = time.time() - start_time
            response.raise_for_status()
            
            result = response.json()
            
            # Display results
            filename = Path(image_path).name
            print(f"\n📷 {filename}")
            print(f"   Model: {result['model_used']}")
            print(f"   🔍 Prediction: {result['prediction']}")
            print(f"   📊 Confidence: {result['confidence']*100:.2f}%")
            print(f"   ⏱️  Time: {elapsed:.2f}s")
            
            return result
    except Exception as e:
        print(f"❌ Prediction failed for {Path(image_path).name}: {e}")
        return None


def test_batch_prediction(image_paths: List[str], model_type: str = "pytorch"):
    """Test batch prediction endpoint"""
    print(f"\n📦 Testing Batch Prediction with {len(image_paths)} images...")
    print("-" * 60)
    
    try:
        files = [('files', open(path, 'rb')) for path in image_paths]
        params = {'model_type': model_type}
        
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/batch-predict",
            files=files,
            params=params,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        # Close files
        for _, f in files:
            f.close()
        
        response.raise_for_status()
        result = response.json()
        
        print(f"✅ Successful: {result['successful']}/{result['total_files']}")
        if result['failed'] > 0:
            print(f"❌ Failed: {result['failed']}")
            for error in result['errors']:
                print(f"   {error}")
        
        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Average time per image: {elapsed/len(image_paths):.2f}s")
        
        return result
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return None


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("🫁 Pneumonia Detection API Test Suite")
    print("=" * 60)
    
    # Check if server is running
    print("\n🔗 Checking server connection...")
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=2)
        print(f"✅ Connected to {API_BASE_URL}")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {API_BASE_URL}")
        print("   Make sure to run: python app.py")
        sys.exit(1)
    
    # Run tests
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Health check
    tests_total += 1
    if test_health():
        tests_passed += 1
    
    # Test 2: Model info
    tests_total += 1
    if test_model_info():
        tests_passed += 1
    
    # Test 3: Find test images
    print("\n🔍 Looking for test images...")
    print("-" * 60)
    test_images = find_test_images(TEST_IMAGES_COUNT)
    
    if not test_images:
        print("❌ No test images found in dataset")
        print(f"   Expected path: {DATASET_PATH}")
        sys.exit(1)
    
    print(f"✅ Found {len(test_images)} test images")
    for img in test_images:
        print(f"   - {Path(img).name}")
    
    # Test 4: Single predictions
    print("\n🤖 Testing PyTorch Model (Single Predictions)...")
    print("=" * 60)
    tests_total += 1
    pytorch_results = []
    for img_path in test_images[:2]:
        result = test_single_prediction(img_path, "pytorch")
        if result:
            pytorch_results.append(result)
    
    if len(pytorch_results) > 0:
        tests_passed += 1
    
    # Test 5: TensorFlow predictions
    print("\n🤖 Testing TensorFlow Model (Single Predictions)...")
    print("=" * 60)
    tests_total += 1
    tf_results = []
    for img_path in test_images[:2]:
        result = test_single_prediction(img_path, "tensorflow")
        if result:
            tf_results.append(result)
    
    if len(tf_results) > 0:
        tests_passed += 1
    
    # Test 6: Batch prediction
    print("\n📦 Testing Batch Prediction...")
    print("=" * 60)
    tests_total += 1
    batch_result = test_batch_prediction(test_images, "pytorch")
    if batch_result and batch_result['successful'] > 0:
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✅ All tests passed! API is working correctly.\n")
    else:
        print(f"⚠️  {tests_total - tests_passed} test(s) failed.\n")


if __name__ == "__main__":
    main()
