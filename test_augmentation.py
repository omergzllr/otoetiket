#!/usr/bin/env python3
"""
Test script for augmentation with proper label generation
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append('utils')

from model_loader import ModelLoader
from predictor import Predictor
from augmentor import Augmentor

def test_augmentation_with_labels():
    """Test augmentation with proper label generation"""
    
    # Setup paths
    model_path = "models/yolov8s.pt"
    test_image_path = "uploads/extracted/fabrika_cam/capture_20250502_123756.jpg"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        return
    
    print("=== Testing Augmentation with Labels ===")
    
    try:
        # Load model
        print("Loading model...")
        model_loader = ModelLoader(model_path)
        model = model_loader.load_model()
        model_type = model_loader.get_model_type()
        print(f"Model loaded: {model_type}")
        
        # Setup predictor and augmentor
        predictor = Predictor(model, model_type)
        augmentor = Augmentor()
        
        # Make prediction
        print("Making prediction...")
        predictions = predictor.predict_image(test_image_path)
        print(f"Predictions: {predictions}")
        
        if len(predictions.get('boxes', [])) == 0:
            print("No detections found - cannot test augmentation")
            return
        
        # Load original image
        original_image = cv2.imread(test_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Convert pixel coordinates to YOLO format for augmentation
        img_height, img_width = original_image.shape[:2]
        yolo_boxes = []
        for box in predictions['boxes']:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_boxes.append([x_center, y_center, width, height])
        
        yolo_boxes = np.array(yolo_boxes)
        print(f"Converted {len(yolo_boxes)} boxes to YOLO format:")
        for i, box in enumerate(yolo_boxes):
            print(f"  Box {i}: {box}")
        
        # Perform augmentation
        print("Performing augmentation...")
        aug_results = augmentor.augment_detection(
            original_image,
            yolo_boxes,
            predictions['class_ids'],
            num_augmentations=2
        )
        
        print(f"Generated {len(aug_results)} augmented versions")
        
        # Save results
        output_dir = Path("test_output")
        output_dir.mkdir(exist_ok=True)
        
        aug_paths = augmentor.save_augmented_results(
            test_image_path, aug_results, output_dir, predictions['type']
        )
        
        print("Saved augmented results:")
        for img_path, label_path in aug_paths:
            print(f"  Image: {img_path}")
            print(f"  Label: {label_path}")
            
            # Check label file content
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    print(f"  Label content: {content}")
            else:
                print(f"  Label file not found!")
        
        print("=== Test completed successfully ===")
        
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_augmentation_with_labels() 