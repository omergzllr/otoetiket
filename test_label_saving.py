#!/usr/bin/env python3
"""
Test script to verify label saving functionality
"""

import os
import cv2
import numpy as np
from pathlib import Path
from utils.predictor import Predictor
from utils.model_loader import ModelLoader

def create_test_image():
    """Create a simple test image with a colored rectangle"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw a red rectangle in the center
    cv2.rectangle(img, (200, 150), (440, 330), (0, 0, 255), -1)
    return img

def test_label_saving():
    """Test the label saving functionality"""
    print("=== Testing Label Saving ===")
    
    try:
        # Create test image
        test_image = create_test_image()
        test_image_path = "test_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        print(f"Created test image: {test_image_path}")
        
        # Try to load a model (use a simple one if available)
        model_path = "models"  # Check if any model exists
        if os.path.exists(model_path) and any(os.listdir(model_path)):
            model_files = [f for f in os.listdir(model_path) if f.endswith(('.pt', '.pth', '.onnx'))]
            if model_files:
                model_file = os.path.join(model_path, model_files[0])
                print(f"Using model: {model_file}")
                
                # Load model
                model_loader = ModelLoader(model_file)
                model = model_loader.load_model()
                model_type = model_loader.get_model_type()
                
                # Setup predictor
                predictor = Predictor(model, model_type)
                
                # Make prediction
                predictions = predictor.predict_image(test_image_path)
                print(f"Predictions: {predictions}")
                
                # Test label saving
                output_dir = Path("test_output")
                output_dir.mkdir(exist_ok=True)
                
                # Test save_predictions
                label_path, vis_path = predictor.save_predictions(
                    test_image_path, 
                    predictions, 
                    output_dir, 
                    "test_object"
                )
                
                print(f"Label saved to: {label_path}")
                print(f"Visualization saved to: {vis_path}")
                
                # Check if label file exists and has content
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    print(f"Label content: '{content}'")
                    if content:
                        print("✓ Label file has content")
                    else:
                        print("⚠ Label file is empty")
                else:
                    print("✗ Label file was not created")
                
                # Test save_labels_only
                label_only_path = output_dir / "test_labels_only.txt"
                predictor.save_labels_only(predictions, label_only_path, "test_object")
                
                if os.path.exists(label_only_path):
                    with open(label_only_path, 'r') as f:
                        content = f.read().strip()
                    print(f"Labels-only content: '{content}'")
                else:
                    print("✗ Labels-only file was not created")
                
            else:
                print("No model files found in models directory")
        else:
            print("Models directory not found or empty")
            
        # Clean up
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
        print("=== Test completed ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_label_saving() 