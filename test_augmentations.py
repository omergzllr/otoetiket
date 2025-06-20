#!/usr/bin/env python3
"""
Test script to verify albumentations functionality
"""

import numpy as np
import cv2
from pathlib import Path

# Add the current directory to Python path
import sys
sys.path.append(str(Path(__file__).parent))

from utils.augmentor import Augmentor

def test_augmentations():
    """Test augmentation functionality"""
    
    print("Testing albumentations functionality...")
    
    # Create a simple test image
    test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    test_boxes = np.array([[0.1, 0.1, 0.2, 0.2]])  # YOLO format
    test_class_ids = np.array([0])
    
    try:
        # Test detection augmentation
        augmentor = Augmentor()
        aug_results = augmentor.augment_detection(test_image, test_boxes, test_class_ids, num_augmentations=2)
        
        print(f"✓ Detection augmentation successful: {len(aug_results)} results")
        
        # Test segmentation augmentation
        test_masks = np.random.randint(0, 2, (2, 100, 100), dtype=np.uint8)
        seg_results = augmentor.augment_segmentation(test_image, test_masks, num_augmentations=2)
        
        print(f"✓ Segmentation augmentation successful: {len(seg_results)} results")
        
        print("✓ All augmentation tests passed!")
        
    except Exception as e:
        print(f"✗ Augmentation test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_augmentations() 