#!/usr/bin/env python3
"""
Test script to debug prediction issues
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from utils.model_loader import ModelLoader
from utils.predictor import Predictor

def test_prediction():
    """Test prediction functionality"""
    
    print("Testing prediction functionality...")
    
    try:
        # Test with default YOLOv8 model
        from ultralytics import YOLO
        print("Loading default YOLOv8s model...")
        model = YOLO('yolov8s.pt')
        
        # Create a test image with a simple object
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Add a simple rectangle to make it more likely to detect something
        cv2.rectangle(test_image, (100, 100), (300, 300), (255, 0, 0), 10)
        
        print(f"Test image shape: {test_image.shape}")
        
        # Test prediction
        predictor = Predictor(model, 'yolov8')
        predictions = predictor.predict_image_from_array(test_image)
        
        print(f"Predictions: {predictions}")
        print(f"Number of detections: {len(predictions.get('boxes', []))}")
        
        if len(predictions.get('boxes', [])) > 0:
            print("✓ Prediction successful with detections")
            
            # Test label saving
            test_label_path = Path("test_label.txt")
            predictor._save_yolo_labels(predictions, test_label_path, None, "test_object")
            
            # Check if file was created and has content
            if test_label_path.exists():
                with open(test_label_path, 'r') as f:
                    content = f.read()
                print(f"✓ Label file created with content: {content}")
                test_label_path.unlink()  # Clean up
            else:
                print("✗ Label file was not created")
        else:
            print("⚠ No detections found in test image")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_prediction() 