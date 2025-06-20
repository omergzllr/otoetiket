#!/usr/bin/env python3
"""
Debug script to test label saving with realistic scenarios
"""

import os
import cv2
import numpy as np
from pathlib import Path
from utils.predictor import Predictor
from utils.model_loader import ModelLoader

def create_test_image_with_objects():
    """Create a test image with objects that YOLO can detect"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw a person-like shape (rectangle)
    cv2.rectangle(img, (200, 100), (300, 400), (255, 255, 255), -1)
    # Draw a head (circle)
    cv2.circle(img, (250, 80), 30, (255, 255, 255), -1)
    
    # Draw a car-like shape
    cv2.rectangle(img, (400, 200), (550, 300), (0, 255, 0), -1)
    cv2.rectangle(img, (420, 180), (530, 200), (0, 255, 0), -1)
    
    return img

def test_with_lower_confidence():
    """Test with lower confidence threshold to get more detections"""
    print("=== Testing with Lower Confidence ===")
    
    try:
        # Create test image
        test_image = create_test_image_with_objects()
        test_image_path = "debug_test_image.jpg"
        cv2.imwrite(test_image_path, test_image)
        print(f"Created test image: {test_image_path}")
        
        # Load model
        model_path = "models/yolov8m.pt"
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return
            
        print(f"Using model: {model_path}")
        model_loader = ModelLoader(model_path)
        model = model_loader.load_model()
        model_type = model_loader.get_model_type()
        
        # Setup predictor
        predictor = Predictor(model, model_type)
        
        # Test with different confidence thresholds
        for conf_threshold in [0.1, 0.05, 0.01]:
            print(f"\n--- Testing with confidence threshold: {conf_threshold} ---")
            
            # Make prediction with lower confidence
            results = model(test_image, conf=conf_threshold)[0]
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                
                print(f"Found {len(boxes)} detections")
                for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                    print(f"  Detection {i}: class {class_id}, score {score:.3f}, box {box}")
                
                predictions = {
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids,
                    'type': 'detection'
                }
                
                # Test label saving
                output_dir = Path("debug_output")
                output_dir.mkdir(exist_ok=True)
                
                label_path, vis_path = predictor.save_predictions(
                    test_image_path, 
                    predictions, 
                    output_dir, 
                    "test_object"
                )
                
                print(f"Label saved to: {label_path}")
                
                # Check label content
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    print(f"Label content: '{content}'")
                    if content:
                        print("✓ SUCCESS: Label file has content!")
                        return True
                    else:
                        print("⚠ Label file is empty")
                else:
                    print("✗ Label file was not created")
            else:
                print("No detections found even with low confidence")
        
        print("\n=== No successful detections found ===")
        return False
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_real_image():
    """Test with a real image if available"""
    print("\n=== Testing with Real Image ===")
    
    # Look for any image in uploads/extracted
    uploads_dir = "uploads/extracted"
    if os.path.exists(uploads_dir):
        image_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            image_files.extend(Path(uploads_dir).rglob(f'*.{ext}'))
        
        if image_files:
            test_image_path = str(image_files[0])
            print(f"Using real image: {test_image_path}")
            
            try:
                # Load model
                model_path = "models/yolov8m.pt"
                model_loader = ModelLoader(model_path)
                model = model_loader.load_model()
                model_type = model_loader.get_model_type()
                
                # Setup predictor
                predictor = Predictor(model, model_type)
                
                # Make prediction
                predictions = predictor.predict_image(test_image_path)
                print(f"Predictions: {len(predictions.get('boxes', []))} detections")
                
                if len(predictions.get('boxes', [])) > 0:
                    # Test label saving
                    output_dir = Path("debug_output")
                    output_dir.mkdir(exist_ok=True)
                    
                    label_path, vis_path = predictor.save_predictions(
                        test_image_path, 
                        predictions, 
                        output_dir, 
                        "test_object"
                    )
                    
                    print(f"Label saved to: {label_path}")
                    
                    # Check label content
                    if os.path.exists(label_path):
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                        print(f"Label content: '{content}'")
                        if content:
                            print("✓ SUCCESS: Label file has content!")
                            return True
                        else:
                            print("⚠ Label file is empty")
                    else:
                        print("✗ Label file was not created")
                else:
                    print("No detections found in real image")
                    
            except Exception as e:
                print(f"Error testing with real image: {e}")
                import traceback
                traceback.print_exc()
    
    print("No real images found for testing")
    return False

if __name__ == "__main__":
    print("=== Debug Label Saving ===")
    
    # Clean up previous test files
    for file in ["debug_test_image.jpg", "debug_output"]:
        if os.path.exists(file):
            if os.path.isdir(file):
                import shutil
                shutil.rmtree(file)
            else:
                os.remove(file)
    
    # Test with synthetic image first
    success = test_with_lower_confidence()
    
    if not success:
        # Try with real image
        success = test_with_real_image()
    
    if success:
        print("\n🎉 SUCCESS: Label saving is working!")
    else:
        print("\n❌ FAILED: Label saving is not working properly")
    
    print("\n=== Debug completed ===") 