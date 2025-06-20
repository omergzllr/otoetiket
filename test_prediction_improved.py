#!/usr/bin/env python3
"""
Improved test script to debug prediction issues with real images
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

from utils.model_loader import ModelLoader
from utils.predictor import Predictor

def download_test_image():
    """Download a test image with known objects"""
    # Download a test image with people (COCO dataset style)
    url = "https://ultralytics.com/images/zidane.jpg"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to numpy array
        image = Image.open(BytesIO(response.content))
        image_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Save locally for debugging
        cv2.imwrite("test_image.jpg", image_bgr)
        print(f"Downloaded test image: {image_bgr.shape}")
        return image_bgr
        
    except Exception as e:
        print(f"Failed to download test image: {e}")
        return None

def create_synthetic_test_image():
    """Create a synthetic test image with simple shapes"""
    # Create a 640x640 image with a white background
    image = np.ones((640, 640, 3), dtype=np.uint8) * 255
    
    # Add a red rectangle (simulating a person)
    cv2.rectangle(image, (200, 150), (400, 450), (0, 0, 255), -1)
    
    # Add a blue circle (simulating a ball)
    cv2.circle(image, (500, 300), 50, (255, 0, 0), -1)
    
    # Add some text
    cv2.putText(image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    # Save for debugging
    cv2.imwrite("synthetic_test_image.jpg", image)
    print(f"Created synthetic test image: {image.shape}")
    return image

def test_prediction_with_real_image():
    """Test prediction with a real image"""
    print("Testing prediction with real image...")
    
    try:
        # Try to download a real test image
        test_image = download_test_image()
        if test_image is None:
            print("Using synthetic test image instead...")
            test_image = create_synthetic_test_image()
        
        # Load YOLOv8 model
        from ultralytics import YOLO
        print("Loading YOLOv8s model...")
        model = YOLO('yolov8s.pt')
        
        # Test direct model prediction first
        print("Testing direct model prediction...")
        results = model(test_image)
        print(f"Direct results: {results}")
        
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            print(f"Direct detection count: {len(results[0].boxes)}")
            if len(results[0].boxes) > 0:
                print(f"Direct detections: {results[0].boxes.xyxy}")
                print(f"Direct scores: {results[0].boxes.conf}")
                print(f"Direct classes: {results[0].boxes.cls}")
        else:
            print("No direct detections found")
        
        # Test through our Predictor class
        print("\nTesting through Predictor class...")
        predictor = Predictor(model, 'yolov8')
        predictions = predictor.predict_image_from_array(test_image)
        
        print(f"Predictor results: {predictions}")
        print(f"Number of detections: {len(predictions.get('boxes', []))}")
        
        if len(predictions.get('boxes', [])) > 0:
            print("✓ Prediction successful with detections")
            
            # Test label saving
            test_label_path = Path("test_label.txt")
            predictor._save_yolo_labels(predictions, test_label_path, Path("test_image.jpg"), "test_object")
            
            # Check if file was created and has content
            if test_label_path.exists():
                with open(test_label_path, 'r') as f:
                    content = f.read()
                print(f"✓ Label file created with content: {content}")
                test_label_path.unlink()  # Clean up
            else:
                print("✗ Label file was not created")
        else:
            print("⚠ No detections found - this might indicate an issue")
            
            # Try with different confidence threshold
            print("\nTrying with lower confidence threshold...")
            results = model(test_image, conf=0.1)  # Lower confidence threshold
            if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                print(f"Found {len(results[0].boxes)} detections with lower confidence")
                print(f"Detections: {results[0].boxes.xyxy}")
                print(f"Scores: {results[0].boxes.conf}")
            else:
                print("Still no detections even with lower confidence")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test model loading functionality"""
    print("\nTesting model loading...")
    
    try:
        loader = ModelLoader()
        
        # Test loading YOLOv8
        print("Testing YOLOv8 loading...")
        model = loader.load_yolo_model("yolov8s.pt")
        print("✓ YOLOv8 model loaded successfully")
        
        # Test prediction with loaded model
        test_image = create_synthetic_test_image()
        predictor = Predictor(model, 'yolov8')
        predictions = predictor.predict_image_from_array(test_image)
        print(f"✓ Model prediction successful: {len(predictions.get('boxes', []))} detections")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== Improved Prediction Test ===")
    
    # Test model loading
    test_model_loading()
    
    # Test prediction with real image
    test_prediction_with_real_image()
    
    print("\n=== Test Complete ===") 