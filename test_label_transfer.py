#!/usr/bin/env python3
"""
Test script for label transfer system
"""

import cv2
import numpy as np
import os
import sys
import json
from pathlib import Path

# Add utils to path
sys.path.append('utils')

from smart_augmentor import SmartAugmentor
from label_transfer import LabelTransfer

def create_test_image_with_labels(width=640, height=480):
    """Create a test image with known labels"""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw multiple rectangles with known positions
    labels = []
    
    # Rectangle 1: Center
    center_x, center_y = width // 2, height // 2
    rect_width, rect_height = 100, 80
    x1 = center_x - rect_width // 2
    y1 = center_y - rect_height // 2
    x2 = center_x + rect_width // 2
    y2 = center_y + rect_height // 2
    
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, "1", (center_x - 5, center_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Convert to YOLO format
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    labels.append([0, x_center, y_center, w, h])
    
    # Rectangle 2: Top left
    x1, y1 = 50, 50
    x2, y2 = 150, 150
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, "2", (x1 + 40, y1 + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    labels.append([0, x_center, y_center, w, h])
    
    # Rectangle 3: Bottom right
    x1, y1 = width - 150, height - 150
    x2, y2 = width - 50, height - 50
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, "3", (x1 + 40, y1 + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    w = (x2 - x1) / width
    h = (y2 - y1) / height
    labels.append([0, x_center, y_center, w, h])
    
    return image, labels

def test_label_transfer():
    """Test the label transfer system"""
    print("=== Testing Label Transfer System ===")
    
    # Create test image and labels
    test_image, original_labels = create_test_image_with_labels()
    test_image_path = "test_image_with_labels.jpg"
    test_label_path = "test_image_with_labels.txt"
    
    cv2.imwrite(test_image_path, test_image)
    
    # Save original labels
    with open(test_label_path, 'w') as f:
        for label in original_labels:
            class_id, x_center, y_center, width, height = label
            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    print(f"Created test image: {test_image_path}")
    print(f"Original labels: {len(original_labels)} objects")
    for i, label in enumerate(original_labels):
        print(f"  Label {i+1}: class={label[0]}, center=({label[1]:.3f}, {label[2]:.3f}), size=({label[3]:.3f}, {label[4]:.3f})")
    
    # Create smart augmentor and augment image
    augmentor = SmartAugmentor()
    augmentation_results = augmentor.augment_image_with_tracking(
        test_image_path, 
        num_augmentations=2,
        output_dir="test_transfer_output"
    )
    
    print(f"\nCreated {len(augmentation_results)} augmented images")
    
    # Test label transfer for each augmented image
    label_transfer = LabelTransfer()
    
    for i, result in enumerate(augmentation_results):
        print(f"\n--- Testing Label Transfer for Augmented Image {i+1} ---")
        print(f"Filename: {result['filename']}")
        print(f"Transforms: {result['metadata']['transform_sequence']}")
        
        # Transfer labels
        transferred_labels = label_transfer.transfer_labels_from_original(
            test_image_path,
            test_label_path,
            result['image_path'],
            result['metadata_path']
        )
        
        print(f"Transferred {len(transferred_labels)} labels")
        
        # Save transferred labels
        transferred_label_path = result['image_path'].replace('.jpg', '.txt')
        label_transfer.save_transferred_labels(transferred_labels, transferred_label_path)
        
        # Display transferred labels
        for j, label in enumerate(transferred_labels):
            print(f"  Transferred Label {j+1}: class={label[0]}, center=({label[1]:.3f}, {label[2]:.3f}), size=({label[3]:.3f}, {label[4]:.3f})")
        
        # Load augmented image and draw transferred boxes
        aug_image = cv2.imread(result['image_path'])
        if aug_image is not None:
            img_height, img_width = aug_image.shape[:2]
            
            for label in transferred_labels:
                class_id, x_center, y_center, width, height = label
                
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width/2) * img_width)
                y1 = int((y_center - height/2) * img_height)
                x2 = int((x_center + width/2) * img_width)
                y2 = int((y_center + height/2) * img_height)
                
                # Draw box
                cv2.rectangle(aug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(aug_image, f"T{int(class_id)}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save image with transferred boxes
            output_path = result['image_path'].replace('.jpg', '_with_transferred_boxes.jpg')
            cv2.imwrite(output_path, aug_image)
            print(f"Saved image with transferred boxes: {output_path}")
    
    print(f"\n=== Label Transfer Test Complete ===")
    print(f"Check the 'test_transfer_output' directory for results")

if __name__ == "__main__":
    test_label_transfer() 