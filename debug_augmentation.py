#!/usr/bin/env python3
"""
Debug script for augmentation process
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

def debug_augmentation():
    """Debug augmentation process step by step"""
    
    # Setup paths
    model_path = "models/yolov8s.pt"
    test_image_path = "uploads/extracted/fabrika_cam/capture_20250502_123756.jpg"
    
    print("=== DEBUG: Augmentation Process ===")
    
    try:
        # 1. Load model
        print("1. Loading model...")
        model_loader = ModelLoader(model_path)
        model = model_loader.load_model()
        model_type = model_loader.get_model_type()
        print(f"   Model loaded: {model_type}")
        
        # 2. Make prediction
        print("2. Making prediction...")
        predictor = Predictor(model, model_type)
        predictions = predictor.predict_image(test_image_path)
        print(f"   Predictions: {predictions}")
        print(f"   Boxes count: {len(predictions.get('boxes', []))}")
        
        if len(predictions.get('boxes', [])) == 0:
            print("   ERROR: No detections found!")
            return
        
        # 3. Load original image
        print("3. Loading original image...")
        original_image = cv2.imread(test_image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        print(f"   Image shape: {original_image.shape}")
        
        # 4. Convert to YOLO format
        print("4. Converting boxes to YOLO format...")
        img_height, img_width = original_image.shape[:2]
        yolo_boxes = []
        for i, box in enumerate(predictions['boxes']):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_boxes.append([x_center, y_center, width, height])
            print(f"   Box {i}: pixel({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) -> YOLO({x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f})")
        
        yolo_boxes = np.array(yolo_boxes)
        print(f"   Converted {len(yolo_boxes)} boxes")
        
        # 5. Perform augmentation
        print("5. Performing augmentation...")
        augmentor = Augmentor()
        aug_results = augmentor.augment_detection(
            original_image,
            yolo_boxes,
            predictions['class_ids'],
            num_augmentations=2
        )
        print(f"   Generated {len(aug_results)} augmented versions")
        
        # 6. Debug each augmented result
        print("6. Debugging augmented results...")
        for i, result in enumerate(aug_results):
            print(f"   Aug {i}:")
            print(f"     Image shape: {result['image'].shape}")
            print(f"     Boxes count: {len(result['boxes'])}")
            print(f"     Class IDs: {result['class_ids']}")
            if len(result['boxes']) > 0:
                for j, box in enumerate(result['boxes']):
                    print(f"       Box {j}: {box}")
            else:
                print(f"       WARNING: No boxes in augmented result {i}!")
        
        # 7. Save results
        print("7. Saving augmented results...")
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)
        
        aug_paths = augmentor.save_augmented_results(
            test_image_path, aug_results, output_dir, predictions['type']
        )
        
        print("8. Checking saved files...")
        for img_path, label_path in aug_paths:
            print(f"   Image: {img_path}")
            print(f"   Label: {label_path}")
            
            # Check if files exist
            if os.path.exists(img_path):
                print(f"     Image exists: {os.path.getsize(img_path)} bytes")
            else:
                print(f"     ERROR: Image file not found!")
                
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                print(f"     Label exists: {len(content)} chars, content: {content}")
            else:
                print(f"     ERROR: Label file not found!")
        
        print("=== Debug completed ===")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_augmentation() 