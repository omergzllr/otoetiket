import os
import torch
from pathlib import Path
from typing import Union, Optional, Tuple

class ModelLoader:
    """Utility class to load and identify AI models"""
    
    SUPPORTED_MODELS = {
        'yolov5': ['.pt'],
        'yolov8': ['.pt'],
        'detectron2': ['.pth'],
        'sam': ['.pth']
    }
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_type = None
        self.model = None
        
    def identify_model_type(self) -> Tuple[str, Optional[str]]:
        """
        Identify the type of model based on file extension and contents
        Returns: Tuple[model_type, specific_version]
        """
        ext = self.model_path.suffix.lower()
        
        # For .pt files, assume YOLO models (most common case)
        if ext == '.pt':
            try:
                # Try to load with weights_only=False first
                model = torch.load(self.model_path, map_location='cpu', weights_only=False)
                
                # Check for YOLOv8 specific attributes
                if 'model.yaml' in str(model):
                    return 'yolov8', 'v8'
                    
                # Check for YOLOv5 specific attributes
                if 'model.24.anchors' in str(model) or 'model.17.anchors' in str(model):
                    return 'yolov5', 'v5'
                    
                # If we can load it but can't identify specific version, assume YOLOv8
                return 'yolov8', 'v8'
                    
            except Exception as e:
                print(f"Error loading PyTorch model: {e}")
                # If loading fails, still assume it's a YOLO model based on extension
                return 'yolov8', 'v8'
                
        # Check for Detectron2 models
        elif ext == '.pth':
            try:
                state_dict = torch.load(self.model_path, map_location='cpu', weights_only=False)
                if any('backbone' in key for key in state_dict.keys()):
                    return 'detectron2', None
                elif any('image_encoder' in key for key in state_dict.keys()):
                    return 'sam', None
            except Exception as e:
                print(f"Error loading .pth model: {e}")
                
        raise ValueError(f"Could not identify model type for file: {self.model_path}")
    
    def load_model(self):
        """
        Load the model based on identified type
        Returns: Loaded model object
        """
        model_type, version = self.identify_model_type()
        self.model_type = model_type
        
        if model_type == 'yolov8':
            from ultralytics import YOLO
            try:
                self.model = YOLO(self.model_path)
            except Exception as e:
                print(f"Error loading YOLOv8 model: {e}")
                # Try alternative loading method
                self.model = YOLO('yolov8s.pt')  # Load default model
                print("Loaded default YOLOv8s model instead")
            
        elif model_type == 'yolov5':
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                          path=str(self.model_path))
            except Exception as e:
                print(f"Error loading YOLOv5 model: {e}")
                # Try alternative loading method
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
                print("Loaded default YOLOv5s model instead")
            
        elif model_type == 'detectron2':
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            
            cfg = get_cfg()
            cfg.merge_from_file(str(self.model_path))
            self.model = DefaultPredictor(cfg)
            
        elif model_type == 'sam':
            from segment_anything import sam_model_registry, SamPredictor
            
            model_type = "vit_h"  # or "vit_l" or "vit_b" based on model
            self.model = sam_model_registry[model_type](checkpoint=str(self.model_path))
            self.model = SamPredictor(self.model)
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return self.model
    
    def get_model_type(self) -> Optional[str]:
        """Get the identified model type"""
        return self.model_type
        
    @staticmethod
    def is_valid_model_file(file_path: Union[str, Path]) -> bool:
        """Check if the file has a supported model extension"""
        ext = Path(file_path).suffix.lower()
        valid_extensions = set()
        for exts in ModelLoader.SUPPORTED_MODELS.values():
            valid_extensions.update(exts)
        return ext in valid_extensions 