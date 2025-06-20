#!/usr/bin/env python3
"""
Test script to verify model loading functionality
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from utils.model_loader import ModelLoader

def test_model_loading():
    """Test model loading with different approaches"""
    
    # Check if there are any model files in the models directory
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found. Creating one...")
        models_dir.mkdir(exist_ok=True)
        print("Please place your model files (.pt, .pth, .pkl, .onnx) in the models/ directory")
        return
    
    model_files = list(models_dir.glob("*.pt")) + list(models_dir.glob("*.pth"))
    
    if not model_files:
        print("No model files found in models/ directory")
        print("Please place your model files (.pt, .pth, .pkl, .onnx) in the models/ directory")
        return
    
    print(f"Found {len(model_files)} model files:")
    for model_file in model_files:
        print(f"  - {model_file.name}")
    
    # Test each model file
    for model_file in model_files:
        print(f"\nTesting model: {model_file.name}")
        try:
            # Test model identification
            loader = ModelLoader(model_file)
            model_type, version = loader.identify_model_type()
            print(f"  ✓ Identified as: {model_type} {version if version else ''}")
            
            # Test model loading (optional - comment out if you don't want to load the full model)
            # model = loader.load_model()
            # print(f"  ✓ Model loaded successfully")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    test_model_loading() 