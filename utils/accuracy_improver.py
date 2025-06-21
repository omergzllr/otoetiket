import cv2
import numpy as np
import os
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import logging

class AccuracyImprover:
    """
    Advanced system to improve label accuracy using multiple strategies:
    1. Ensemble predictions from multiple confidence thresholds
    2. Non-maximum suppression (NMS) to remove duplicate detections
    3. Confidence score calibration
    4. Size-based filtering for unrealistic detections
    5. Position-based validation
    """
    
    def __init__(self, min_size_ratio=0.01, max_size_ratio=0.8, iou_threshold=0.5):
        self.min_size_ratio = min_size_ratio  # Minimum box size as ratio of image
        self.max_size_ratio = max_size_ratio  # Maximum box size as ratio of image
        self.iou_threshold = iou_threshold    # IoU threshold for NMS
        self.logger = logging.getLogger(__name__)
        
    def ensemble_predictions(self, model, image_path: str, confidence_thresholds: List[float] = None) -> Dict:
        """
        Generate ensemble predictions using multiple confidence thresholds
        """
        if confidence_thresholds is None:
            confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        image = cv2.imread(image_path)
        if image is None:
            return {'boxes': [], 'scores': [], 'class_ids': [], 'type': 'detection'}
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        all_predictions = []
        
        # Get predictions for each confidence threshold
        for conf_threshold in confidence_thresholds:
            try:
                results = model(image, conf=conf_threshold)
                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy()
                    
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        all_predictions.append({
                            'box': box,
                            'score': score,
                            'class_id': int(class_id),
                            'conf_threshold': conf_threshold
                        })
            except Exception as e:
                self.logger.warning(f"Error with confidence threshold {conf_threshold}: {e}")
                continue
        
        if not all_predictions:
            return {'boxes': [], 'scores': [], 'class_ids': [], 'type': 'detection'}
        
        # Apply ensemble strategies
        ensemble_boxes = self._apply_ensemble_strategies(all_predictions, width, height)
        
        return {
            'boxes': ensemble_boxes['boxes'],
            'scores': ensemble_boxes['scores'],
            'class_ids': ensemble_boxes['class_ids'],
            'type': 'detection'
        }
    
    def _apply_ensemble_strategies(self, all_predictions: List[Dict], width: int, height: int) -> Dict:
        """
        Apply ensemble strategies to combine predictions
        """
        # Group predictions by class and location
        class_groups = {}
        
        for pred in all_predictions:
            class_id = pred['class_id']
            if class_id not in class_groups:
                class_groups[class_id] = []
            class_groups[class_id].append(pred)
        
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        for class_id, predictions in class_groups.items():
            # Apply NMS for each class
            class_boxes = self._apply_nms(predictions)
            
            # Filter by size and position
            filtered_boxes = self._filter_by_size_and_position(class_boxes, width, height)
            
            # Add to final results
            for box_data in filtered_boxes:
                final_boxes.append(box_data['box'])
                final_scores.append(box_data['score'])
                final_class_ids.append(class_id)
        
        return {
            'boxes': np.array(final_boxes) if final_boxes else np.array([]),
            'scores': np.array(final_scores) if final_scores else np.array([]),
            'class_ids': np.array(final_class_ids) if final_class_ids else np.array([])
        }
    
    def _apply_nms(self, predictions: List[Dict]) -> List[Dict]:
        """
        Apply Non-Maximum Suppression to remove duplicate detections
        """
        if len(predictions) <= 1:
            return predictions
        
        # Sort by confidence score
        predictions.sort(key=lambda x: x['score'], reverse=True)
        
        kept_predictions = []
        
        for i, pred in enumerate(predictions):
            if pred is None:
                continue
                
            kept_predictions.append(pred)
            
            # Check IoU with remaining predictions
            for j in range(i + 1, len(predictions)):
                if predictions[j] is None:
                    continue
                    
                iou = self._calculate_iou(pred['box'], predictions[j]['box'])
                if iou > self.iou_threshold:
                    predictions[j] = None  # Mark for removal
        
        return kept_predictions
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two bounding boxes
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _filter_by_size_and_position(self, predictions: List[Dict], width: int, height: int) -> List[Dict]:
        """
        Filter predictions based on size and position constraints
        """
        filtered = []
        
        for pred in predictions:
            box = pred['box']
            x1, y1, x2, y2 = box
            
            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate ratios
            width_ratio = box_width / width
            height_ratio = box_height / height
            
            # Check size constraints
            if (width_ratio < self.min_size_ratio or width_ratio > self.max_size_ratio or
                height_ratio < self.min_size_ratio or height_ratio > self.max_size_ratio):
                continue
            
            # Check position constraints (box should be within image bounds)
            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                continue
            
            # Check aspect ratio (avoid extremely thin or wide boxes)
            aspect_ratio = box_width / box_height if box_height > 0 else 0
            if aspect_ratio < 0.1 or aspect_ratio > 10:
                continue
            
            filtered.append(pred)
        
        return filtered
    
    def calibrate_confidence_scores(self, predictions: Dict) -> Dict:
        """
        Calibrate confidence scores based on detection quality
        """
        if not predictions['boxes'].size:
            return predictions
        
        calibrated_scores = []
        
        for i, (box, score) in enumerate(zip(predictions['boxes'], predictions['scores'])):
            # Calculate box quality metrics
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Size-based calibration
            size_factor = min(box_width * box_height / 1000, 1.0)  # Normalize by expected size
            
            # Position-based calibration (prefer boxes not at edges)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            edge_factor = 1.0 - min(abs(center_x - 0.5) + abs(center_y - 0.5), 0.5)
            
            # Apply calibration
            calibrated_score = score * (0.7 + 0.3 * size_factor * edge_factor)
            calibrated_score = min(calibrated_score, 1.0)  # Cap at 1.0
            
            calibrated_scores.append(calibrated_score)
        
        predictions['scores'] = np.array(calibrated_scores)
        return predictions
    
    def improve_predictions(self, model, image_path: str, base_confidence: float = 0.3) -> Dict:
        """
        Main method to improve predictions using all strategies
        """
        self.logger.info(f"Improving predictions for {image_path}")
        
        # Generate ensemble predictions
        ensemble_pred = self.ensemble_predictions(model, image_path)
        
        if not ensemble_pred['boxes'].size:
            self.logger.warning("No predictions found in ensemble")
            return ensemble_pred
        
        # Calibrate confidence scores
        improved_pred = self.calibrate_confidence_scores(ensemble_pred)
        
        # Apply final confidence threshold
        mask = improved_pred['scores'] >= base_confidence
        improved_pred['boxes'] = improved_pred['boxes'][mask]
        improved_pred['scores'] = improved_pred['scores'][mask]
        improved_pred['class_ids'] = improved_pred['class_ids'][mask]
        
        self.logger.info(f"Improved predictions: {len(improved_pred['boxes'])} detections")
        return improved_pred
    
    def validate_labels(self, label_path: str, image_path: str) -> Dict:
        """
        Validate existing labels and suggest improvements
        """
        if not os.path.exists(label_path) or not os.path.exists(image_path):
            return {'valid': False, 'issues': ['File not found']}
        
        image = cv2.imread(image_path)
        if image is None:
            return {'valid': False, 'issues': ['Cannot read image']}
        
        height, width = image.shape[:2]
        issues = []
        warnings = []
        
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    issues.append(f"Line {i+1}: Invalid format")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                    
                    # Validate coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                        issues.append(f"Line {i+1}: Center coordinates out of bounds")
                    
                    if not (0 < box_width <= 1 and 0 < box_height <= 1):
                        issues.append(f"Line {i+1}: Box dimensions invalid")
                    
                    # Check for unrealistic sizes
                    if box_width < self.min_size_ratio or box_height < self.min_size_ratio:
                        warnings.append(f"Line {i+1}: Very small box detected")
                    
                    if box_width > self.max_size_ratio or box_height > self.max_size_ratio:
                        warnings.append(f"Line {i+1}: Very large box detected")
                    
                    # Check aspect ratio
                    aspect_ratio = box_width / box_height if box_height > 0 else 0
                    if aspect_ratio < 0.1 or aspect_ratio > 10:
                        warnings.append(f"Line {i+1}: Extreme aspect ratio")
                    
                except ValueError:
                    issues.append(f"Line {i+1}: Invalid numeric values")
            
            return {
                'valid': len(issues) == 0,
                'issues': issues,
                'warnings': warnings,
                'total_labels': len(lines)
            }
            
        except Exception as e:
            return {'valid': False, 'issues': [f'Error reading file: {str(e)}']} 