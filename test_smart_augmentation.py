#!/usr/bin/env python3
"""
Test script for smart augmentation system
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add utils to path
sys.path.append('utils')

from smart_augmentor import SmartAugmentor

def create_test_image(width=640, height=480):
    """Create a test image with a simple rectangle"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a rectangle in the center
    center_x, center_y = width // 2, height // 2
    rect_width, rect_height = 100, 80
    
    x1 = center_x - rect_width // 2
    y1 = center_y - rect_height // 2
    x2 = center_x + rect_width // 2
    y2 = center_y + rect_height // 2
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Add text
    cv2.putText(image, "TEST", (center_x - 20, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return image, (x1, y1, x2, y2)

def test_augmentation():
    """Test the smart augmentation system"""
    print("=== Testing Smart Augmentation System ===")
    
    # Create test image
    test_image, bbox = create_test_image()
    test_image_path = "test_image.jpg"
    cv2.imwrite(test_image_path, test_image)
    
    print(f"Created test image: {test_image_path}")
    print(f"Original bbox: {bbox}")
    
    # Create smart augmentor
    augmentor = SmartAugmentor()
    
    # Test augmentation
    results = augmentor.augment_image_with_tracking(
        test_image_path, 
        num_augmentations=3,
        output_dir="test_output"
    )
    
    print(f"\nCreated {len(results)} augmented images:")
    
    for i, result in enumerate(results):
        print(f"\n--- Augmented Image {i+1} ---")
        print(f"Filename: {result['filename']}")
        print(f"Transforms: {result['metadata']['transform_sequence']}")
        print(f"Applied transforms: {result['metadata']['applied_transforms']}")
        
        # Test coordinate transformation
        # Create a dummy box in YOLO format (normalized)
        original_bbox = bbox
        x1, y1, x2, y2 = original_bbox
        
        # Convert to YOLO format
        img_height, img_width = test_image.shape[:2]
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        yolo_box = np.array([[x_center, y_center, width, height]])
        
        print(f"Original YOLO box: {yolo_box[0]}")
        
        # Transform coordinates
        transformed_box = augmentor.transform_coordinates(yolo_box, result['metadata'])
        print(f"Transformed YOLO box: {transformed_box[0]}")
        
        # Convert back to pixel coordinates
        transformed_pixel = []
        for box in transformed_box:
            x_center, y_center, width, height = box
            x1_new = (x_center - width/2) * img_width
            y1_new = (y_center - height/2) * img_height
            x2_new = (x_center + width/2) * img_width
            y2_new = (y_center + height/2) * img_height
            transformed_pixel.append([x1_new, y1_new, x2_new, y2_new])
        
        print(f"Transformed pixel box: {transformed_pixel[0]}")
        
        # Load augmented image and draw the transformed box
        aug_image = cv2.imread(result['image_path'])
        if aug_image is not None:
            x1, y1, x2, y2 = map(int, transformed_pixel[0])
            cv2.rectangle(aug_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Save with box drawn
            output_path = result['image_path'].replace('.jpg', '_with_box.jpg')
            cv2.imwrite(output_path, aug_image)
            print(f"Saved image with box: {output_path}")
    
    print(f"\n=== Test Complete ===")
    print(f"Check the 'test_output' directory for results")

if __name__ == "__main__":
    test_augmentation() 