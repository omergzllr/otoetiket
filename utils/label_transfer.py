import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Dict, Union, Tuple, Optional

class LabelTransfer:
    """Transfer labels from original images to augmented images using coordinate transformations"""
    
    def __init__(self):
        pass
    
    def transfer_labels_from_original(self, 
                                    original_image_path: str,
                                    original_label_path: str,
                                    augmented_image_path: str,
                                    metadata_path: str) -> List[List[float]]:
        """
        Transfer labels from original image to augmented image using metadata
        Returns: List of YOLO format labels for augmented image
        """
        
        # Load original labels
        original_labels = self._load_yolo_labels(original_label_path)
        if not original_labels:
            return []
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Transform coordinates
        transformed_labels = []
        for label in original_labels:
            class_id, x_center, y_center, width, height = label
            
            # Convert to YOLO format array
            yolo_box = np.array([[x_center, y_center, width, height]])
            
            # Apply transformations
            transformed_box = self._apply_transformations(yolo_box, metadata)
            
            # Add class_id back
            transformed_labels.append([class_id] + transformed_box[0].tolist())
        
        return transformed_labels
    
    def _load_yolo_labels(self, label_path: str) -> List[List[float]]:
        """Load YOLO format labels from file"""
        labels = []
        
        if not os.path.exists(label_path):
            return labels
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            labels.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"Error loading labels from {label_path}: {e}")
        
        return labels
    
    def _apply_transformations(self, boxes: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply coordinate transformations based on augmentation metadata"""
        if not boxes.size:
            return boxes
        
        transformed_boxes = boxes.copy()
        
        # Apply transformations in reverse order (as they were applied to the image)
        for transform_info in reversed(metadata.get('applied_transforms', [])):
            transform_name = transform_info['name']
            
            if transform_name == 'horizontal_flip':
                # Flip x coordinates
                transformed_boxes[:, 0] = 1.0 - transformed_boxes[:, 0]
                
            elif transform_name == 'rotation':
                # Apply rotation based on angle
                angle = transform_info['angle']
                
                if angle == 90:
                    # Rotate 90 degrees clockwise
                    x_center = transformed_boxes[:, 0]
                    y_center = transformed_boxes[:, 1]
                    width = transformed_boxes[:, 2]
                    height = transformed_boxes[:, 3]
                    
                    new_x_center = y_center
                    new_y_center = 1.0 - x_center
                    new_width = height
                    new_height = width
                    
                    transformed_boxes[:, 0] = new_x_center
                    transformed_boxes[:, 1] = new_y_center
                    transformed_boxes[:, 2] = new_width
                    transformed_boxes[:, 3] = new_height
                    
                elif angle == 180:
                    # Rotate 180 degrees
                    transformed_boxes[:, 0] = 1.0 - transformed_boxes[:, 0]
                    transformed_boxes[:, 1] = 1.0 - transformed_boxes[:, 1]
                    
                elif angle == 270:
                    # Rotate 270 degrees clockwise (90 degrees counter-clockwise)
                    x_center = transformed_boxes[:, 0]
                    y_center = transformed_boxes[:, 1]
                    width = transformed_boxes[:, 2]
                    height = transformed_boxes[:, 3]
                    
                    new_x_center = 1.0 - y_center
                    new_y_center = x_center
                    new_width = height
                    new_height = width
                    
                    transformed_boxes[:, 0] = new_x_center
                    transformed_boxes[:, 1] = new_y_center
                    transformed_boxes[:, 2] = new_width
                    transformed_boxes[:, 3] = new_height
                
                # angle == 0 means no rotation, so no changes needed
        
        return transformed_boxes
    
    def save_transferred_labels(self, labels: List[List[float]], output_path: str):
        """Save transferred labels to file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            for label in labels:
                class_id, x_center, y_center, width, height = label
                f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    def compare_labels(self, 
                      original_labels: List[List[float]], 
                      predicted_labels: List[List[float]],
                      threshold: float = 0.1) -> Dict:
        """Compare original and predicted labels for consistency"""
        
        if not original_labels and not predicted_labels:
            return {'consistent': True, 'score': 1.0, 'details': 'No labels to compare'}
        
        if not original_labels or not predicted_labels:
            return {'consistent': False, 'score': 0.0, 'details': 'Missing labels'}
        
        # Convert to numpy arrays for easier comparison
        orig_boxes = np.array([label[1:] for label in original_labels])  # Skip class_id
        pred_boxes = np.array([label[1:] for label in predicted_labels])  # Skip class_id
        
        # Calculate IoU for each pair
        ious = []
        for orig_box in orig_boxes:
            for pred_box in pred_boxes:
                iou = self._calculate_iou(orig_box, pred_box)
                ious.append(iou)
        
        if not ious:
            return {'consistent': False, 'score': 0.0, 'details': 'No overlapping boxes'}
        
        max_iou = max(ious)
        avg_iou = np.mean(ious)
        
        # Determine consistency
        consistent = max_iou >= threshold
        
        details = f"Max IoU: {max_iou:.3f}, Avg IoU: {avg_iou:.3f}, Threshold: {threshold}"
        
        return {
            'consistent': consistent,
            'score': max_iou,
            'avg_score': avg_iou,
            'details': details,
            'original_count': len(original_labels),
            'predicted_count': len(predicted_labels)
        }
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two YOLO format boxes"""
        # Convert YOLO format to corner format
        x1_1, y1_1, w1, h1 = box1
        x1_2, y1_2, w2, h2 = box2
        
        # Calculate corners
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0 