#!/usr/bin/env python3
"""
Test script for the Accuracy Improvement System
This script demonstrates how the accuracy improvement system works
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add utils to path
sys.path.append('utils')

from accuracy_improver import AccuracyImprover
from model_loader import ModelLoader
from predictor import Predictor

def test_accuracy_improvement():
    """Test the accuracy improvement system"""
    
    print("=== Testing Accuracy Improvement System ===")
    
    # Initialize accuracy improver
    accuracy_improver = AccuracyImprover(
        min_size_ratio=0.01,  # Minimum box size as 1% of image
        max_size_ratio=0.8,   # Maximum box size as 80% of image
        iou_threshold=0.5     # IoU threshold for NMS
    )
    
    # Test image path
    test_image_path = "test_image.jpg"
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found: {test_image_path}")
        print("Please place a test image named 'test_image.jpg' in the current directory")
        return
    
    # Load model (you'll need to provide a model file)
    model_path = "models/best.pt"  # Adjust path as needed
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Please place a YOLO model file in the models directory")
        return
    
    try:
        # Load model
        print("Loading model...")
        model_loader = ModelLoader(model_path)
        model = model_loader.load_model()
        model_type = model_loader.get_model_type()
        print(f"Model loaded successfully. Type: {model_type}")
        
        # Test regular predictions
        print("\n=== Testing Regular Predictions ===")
        predictor = Predictor(model, model_type)
        regular_predictions = predictor.predict_image(test_image_path)
        print(f"Regular predictions: {len(regular_predictions.get('boxes', []))} detections")
        
        if regular_predictions.get('boxes') is not None and len(regular_predictions['boxes']) > 0:
            print(f"Confidence scores: {regular_predictions['scores']}")
            print(f"Average confidence: {np.mean(regular_predictions['scores']):.3f}")
        
        # Test accuracy improvement
        print("\n=== Testing Accuracy Improvement ===")
        improved_predictions = accuracy_improver.improve_predictions(
            model, 
            test_image_path, 
            base_confidence=0.3
        )
        print(f"Improved predictions: {len(improved_predictions.get('boxes', []))} detections")
        
        if improved_predictions.get('boxes') is not None and len(improved_predictions['boxes']) > 0:
            print(f"Improved confidence scores: {improved_predictions['scores']}")
            print(f"Average improved confidence: {np.mean(improved_predictions['scores']):.3f}")
        
        # Test ensemble predictions
        print("\n=== Testing Ensemble Predictions ===")
        ensemble_predictions = accuracy_improver.ensemble_predictions(
            model, 
            test_image_path, 
            confidence_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        print(f"Ensemble predictions: {len(ensemble_predictions.get('boxes', []))} detections")
        
        # Test label validation
        print("\n=== Testing Label Validation ===")
        
        # Create a test label file
        test_label_path = "test_labels.txt"
        with open(test_label_path, 'w') as f:
            f.write("0 0.5 0.5 0.2 0.3\n")  # Valid label
            f.write("0 0.8 0.8 0.1 0.1\n")  # Small box (warning)
            f.write("0 1.5 0.5 0.2 0.3\n")  # Invalid coordinates (error)
        
        validation_result = accuracy_improver.validate_labels(test_label_path, test_image_path)
        print(f"Validation result: {validation_result}")
        
        # Clean up test file
        if os.path.exists(test_label_path):
            os.remove(test_label_path)
        
        print("\n=== Accuracy Improvement Test Complete ===")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_accuracy_improvement_with_sample_data():
    """Test with sample data to show the benefits"""
    
    print("\n=== Sample Data Accuracy Improvement Demo ===")
    
    # Create sample predictions (simulating model output)
    sample_predictions = {
        'boxes': np.array([
            [100, 100, 200, 200],  # Good detection
            [105, 105, 195, 195],  # Duplicate (high IoU)
            [50, 50, 60, 60],      # Too small
            [10, 10, 800, 600],    # Too large
            [300, 300, 400, 400],  # Good detection
        ]),
        'scores': np.array([0.8, 0.7, 0.6, 0.5, 0.9]),
        'class_ids': np.array([0, 0, 0, 0, 0]),
        'type': 'detection'
    }
    
    print("Sample predictions before improvement:")
    print(f"Number of detections: {len(sample_predictions['boxes'])}")
    print(f"Confidence scores: {sample_predictions['scores']}")
    
    # Simulate accuracy improvement
    accuracy_improver = AccuracyImprover()
    
    # Apply NMS manually
    print("\nApplying Non-Maximum Suppression...")
    # This would normally be done within the ensemble system
    # For demo purposes, we'll show the concept
    
    # Filter by size
    print("Filtering by size constraints...")
    filtered_boxes = []
    filtered_scores = []
    
    for box, score in zip(sample_predictions['boxes'], sample_predictions['scores']):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Check size constraints (assuming 640x640 image)
        width_ratio = width / 640
        height_ratio = height / 640
        
        if (width_ratio >= 0.01 and width_ratio <= 0.8 and 
            height_ratio >= 0.01 and height_ratio <= 0.8):
            filtered_boxes.append(box)
            filtered_scores.append(score)
    
    print(f"After size filtering: {len(filtered_boxes)} detections")
    
    # Calibrate confidence scores
    print("Calibrating confidence scores...")
    calibrated_scores = []
    for box, score in zip(filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Size-based calibration
        size_factor = min(box_width * box_height / 1000, 1.0)
        
        # Position-based calibration
        center_x = (x1 + x2) / 2 / 640  # Normalize
        center_y = (y1 + y2) / 2 / 640
        edge_factor = 1.0 - min(abs(center_x - 0.5) + abs(center_y - 0.5), 0.5)
        
        calibrated_score = score * (0.7 + 0.3 * size_factor * edge_factor)
        calibrated_scores.append(min(calibrated_score, 1.0))
    
    print("Sample predictions after improvement:")
    print(f"Number of detections: {len(filtered_boxes)}")
    print(f"Original scores: {filtered_scores}")
    print(f"Calibrated scores: {[f'{s:.3f}' for s in calibrated_scores]}")

if __name__ == "__main__":
    test_accuracy_improvement()
    test_accuracy_improvement_with_sample_data() 