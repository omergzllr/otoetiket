#!/usr/bin/env python3
"""
Simple test to verify model loading with default models
"""

import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_default_model_loading():
    """Test loading default YOLO models"""
    
    print("Testing default model loading...")
    
    try:
        from ultralytics import YOLO
        
        # Test loading default YOLOv8s model
        print("Loading default YOLOv8s model...")
        model = YOLO('yolov8s.pt')
        print("✓ Default YOLOv8s model loaded successfully")
        
        # Test a simple prediction
        import numpy as np
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        results = model(test_image)
        print("✓ Model prediction successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    test_default_model_loading() 