#!/usr/bin/env python3
"""
Detailed model testing script to identify prediction issues
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_model_with_different_images():
    """Test model with various types of images"""
    print("=== Detailed Model Testing ===")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8s.pt')
        
        # Test 1: Real image with people
        print("\n1. Testing with real image (people)...")
        test_image = download_test_image()
        if test_image is not None:
            results = model(test_image)
            print(f"   Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for i, (box, conf, cls) in enumerate(zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls)):
                    class_name = results[0].names[int(cls)]
                    print(f"   - {class_name}: {conf:.3f} confidence")
        
        # Test 2: Synthetic image with shapes
        print("\n2. Testing with synthetic shapes...")
        synthetic_image = create_synthetic_image()
        results = model(synthetic_image)
        print(f"   Detections: {len(results[0].boxes) if results[0].boxes is not None else 0}")
        
        # Test 3: Different confidence thresholds
        print("\n3. Testing different confidence thresholds...")
        if test_image is not None:
            for conf_threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                results = model(test_image, conf=conf_threshold)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"   Confidence {conf_threshold}: {detections} detections")
        
        # Test 4: Different image sizes
        print("\n4. Testing different image sizes...")
        if test_image is not None:
            for size in [(320, 320), (640, 640), (1280, 1280)]:
                resized = cv2.resize(test_image, size)
                results = model(resized)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"   Size {size}: {detections} detections")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_test_image():
    """Download a test image"""
    url = "https://ultralytics.com/images/zidane.jpg"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        return image_bgr
    except Exception as e:
        print(f"Failed to download test image: {e}")
        return None

def create_synthetic_image():
    """Create a synthetic test image"""
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.rectangle(image, (200, 150), (400, 450), (0, 0, 255), -1)
    cv2.circle(image, (500, 300), 50, (255, 0, 0), -1)
    return image

def test_custom_model():
    """Test if there's a custom model in the models directory"""
    print("\n=== Testing Custom Models ===")
    
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
        print(f"Found {len(model_files)} model files:")
        
        for model_file in model_files:
            print(f"  - {model_file.name}")
            
            try:
                from ultralytics import YOLO
                model = YOLO(str(model_file))
                
                # Test with synthetic image
                test_image = create_synthetic_image()
                results = model(test_image)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                print(f"    Detections: {detections}")
                
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    print(f"    Classes: {set(results[0].boxes.cls.cpu().numpy().astype(int))}")
                    print(f"    Average confidence: {results[0].boxes.conf.cpu().numpy().mean():.3f}")
                
            except Exception as e:
                print(f"    Error loading model: {e}")
    else:
        print("No models directory found")

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    print("\n=== Testing Complete Pipeline ===")
    
    try:
        from utils.predictor import Predictor
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('yolov8s.pt')
        predictor = Predictor(model, 'yolov8')
        
        # Test image
        test_image = download_test_image()
        if test_image is None:
            test_image = create_synthetic_image()
        
        # Predict
        predictions = predictor.predict_image_from_array(test_image)
        print(f"Pipeline predictions: {len(predictions.get('boxes', []))} detections")
        
        # Save labels
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        label_path, vis_path = predictor.save_predictions(
            Path("test_image.jpg"), 
            predictions, 
            output_dir,
            "test_object"
        )
        
        print(f"Label saved to: {label_path}")
        print(f"Visualization saved to: {vis_path}")
        
        return True
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test different scenarios
    test_model_with_different_images()
    test_custom_model()
    test_prediction_pipeline()
    
    print("\n=== Testing Complete ===")
    print("\nIf you're having issues with your specific model:")
    print("1. Check if your model file is in the 'models' directory")
    print("2. Verify the model format (should be .pt or .pth)")
    print("3. Try different confidence thresholds")
    print("4. Check image quality and size")
    print("5. Ensure the model is trained for the objects you're trying to detect") 