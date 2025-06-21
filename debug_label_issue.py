#!/usr/bin/env python3
"""
Debug script to analyze label issues
"""

import os
import sys
from pathlib import Path

def analyze_outputs():
    """Analyze the current output directory structure"""
    print("=== Analyzing Output Directory ===")
    
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
    
    # Check original labels
    original_labels_dir = os.path.join(output_dir, "original_labels")
    if os.path.exists(original_labels_dir):
        original_files = list(Path(original_labels_dir).glob("*.txt"))
        print(f"Original labels directory: {len(original_files)} files")
        
        for i, file_path in enumerate(original_files[:5]):  # Show first 5
            with open(file_path, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n') if content else []
                print(f"  {file_path.name}: {len(lines)} labels")
                if lines:
                    print(f"    Sample: {lines[0]}")
    else:
        print("Original labels directory does not exist")
    
    # Check augmented labels
    augmented_labels_dir = os.path.join(output_dir, "augmented_labels")
    if os.path.exists(augmented_labels_dir):
        augmented_files = list(Path(augmented_labels_dir).glob("*.txt"))
        print(f"Augmented labels directory: {len(augmented_files)} files")
        
        for i, file_path in enumerate(augmented_files[:5]):  # Show first 5
            with open(file_path, 'r') as f:
                content = f.read().strip()
                lines = content.split('\n') if content else []
                print(f"  {file_path.name}: {len(lines)} labels")
                if lines:
                    print(f"    Sample: {lines[0]}")
    else:
        print("Augmented labels directory does not exist")
    
    # Check augmented images
    augmented_images_dir = os.path.join(output_dir, "augmented_images")
    if os.path.exists(augmented_images_dir):
        image_files = list(Path(augmented_images_dir).glob("*.jpg"))
        print(f"Augmented images directory: {len(image_files)} files")
        
        for i, file_path in enumerate(image_files[:5]):  # Show first 5
            print(f"  {file_path.name}")
    else:
        print("Augmented images directory does not exist")
    
    # Check metadata files
    if os.path.exists(augmented_images_dir):
        metadata_files = list(Path(augmented_images_dir).glob("*_metadata.json"))
        print(f"Metadata files: {len(metadata_files)} files")
        
        if metadata_files:
            # Show sample metadata
            with open(metadata_files[0], 'r') as f:
                import json
                metadata = json.load(f)
                print(f"Sample metadata structure: {list(metadata.keys())}")
                print(f"Transform sequence: {metadata.get('transform_sequence', [])}")

def analyze_file_naming():
    """Analyze file naming patterns"""
    print("\n=== Analyzing File Naming Patterns ===")
    
    # Check uploads/extracted
    extract_dir = os.path.join("uploads", "extracted")
    if os.path.exists(extract_dir):
        image_files = []
        for ext in ['png', 'jpg', 'jpeg']:
            image_files.extend(Path(extract_dir).glob(f"*.{ext}"))
        
        print(f"Original images in uploads/extracted: {len(image_files)} files")
        
        for i, file_path in enumerate(image_files[:5]):  # Show first 5
            print(f"  {file_path.name}")
            
            # Analyze naming pattern
            name = file_path.stem
            if '.rf.' in name:
                parts = name.split('.rf.')
                print(f"    Base: {parts[0]}, Hash: {parts[1]}")
    
    # Check augmented images naming
    aug_dir = os.path.join("outputs", "augmented_images")
    if os.path.exists(aug_dir):
        aug_files = list(Path(aug_dir).glob("*.jpg"))
        if aug_files:
            print(f"\nAugmented image naming patterns:")
            for i, file_path in enumerate(aug_files[:3]):  # Show first 3
                name = file_path.stem
                print(f"  {name}")
                
                # Try to extract base name
                if '_aug' in name:
                    base = name.split('_aug')[0]
                    print(f"    Extracted base: {base}")

def test_model_prediction():
    """Test if model can make predictions"""
    print("\n=== Testing Model Prediction ===")
    
    # Check if there are any models
    model_dir = "models"
    if os.path.exists(model_dir):
        model_files = list(Path(model_dir).glob("*"))
        print(f"Models found: {len(model_files)} files")
        
        for model_file in model_files:
            print(f"  {model_file.name}")
    
    # Check if there are any test images
    extract_dir = os.path.join("uploads", "extracted")
    if os.path.exists(extract_dir):
        test_images = list(Path(extract_dir).glob("*.jpg"))[:1]  # Take first image
        if test_images:
            print(f"Test image: {test_images[0].name}")
            
            # Try to load and predict (if model exists)
            try:
                sys.path.append('utils')
                from model_loader import ModelLoader
                from predictor import Predictor
                
                # Find first model
                model_files = list(Path("models").glob("*"))
                if model_files:
                    model_path = str(model_files[0])
                    print(f"Loading model: {model_path}")
                    
                    model_loader = ModelLoader(model_path)
                    model = model_loader.load_model()
                    model_type = model_loader.get_model_type()
                    
                    predictor = Predictor(model, model_type)
                    predictions = predictor.predict_image(str(test_images[0]))
                    
                    print(f"Model type: {model_type}")
                    print(f"Predictions: {len(predictions.get('boxes', []))} detections")
                    
                    if 'scores' in predictions and len(predictions['scores']) > 0:
                        print(f"Confidence scores: {predictions['scores']}")
                        print(f"Average confidence: {predictions['scores'].mean():.3f}")
                    
            except Exception as e:
                print(f"Error testing model: {e}")

if __name__ == "__main__":
    analyze_outputs()
    analyze_file_naming()
    test_model_prediction() 