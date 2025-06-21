import os
import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Union, Tuple

class Predictor:
    """Handles predictions for different model types"""
    
    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        
    def predict_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Predict on a single image
        Returns: Dict with predictions and metadata
        """
        image_path = Path(image_path)
        print(f"Processing image: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image.shape}")
        
        if self.model_type in ['yolov5', 'yolov8']:
            return self._predict_yolo(image)
        elif self.model_type == 'detectron2':
            return self._predict_detectron2(image)
        elif self.model_type == 'sam':
            return self._predict_sam(image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def predict_image_with_augmentation_correction(self, 
                                                 image_path: Union[str, Path],
                                                 metadata_path: Union[str, Path] = None) -> Dict:
        """
        Predict on an image and correct coordinates if it's an augmented image
        Returns: Dict with predictions and metadata
        """
        image_path = Path(image_path)
        
        # Get predictions
        predictions = self.predict_image(image_path)
        
        # Check if this is an augmented image and load metadata
        if metadata_path and Path(metadata_path).exists():
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Apply coordinate transformations based on augmentation metadata
                if predictions['type'] in ['detection', 'instance_segmentation'] and len(predictions['boxes']) > 0:
                    predictions['boxes'] = self._apply_augmentation_corrections(
                        predictions['boxes'], metadata
                    )
                    print(f"Applied augmentation corrections to {len(predictions['boxes'])} boxes")
                    
            except Exception as e:
                print(f"Warning: Could not apply augmentation corrections: {e}")
        
        return predictions
    
    def _apply_augmentation_corrections(self, boxes: np.ndarray, metadata: Dict) -> np.ndarray:
        """Apply coordinate corrections based on augmentation metadata"""
        if not boxes.size:
            return boxes
        
        # Convert boxes from absolute coordinates to normalized YOLO format first
        # Assuming boxes are in [x1, y1, x2, y2] format
        original_shape = metadata.get('original_shape', [480, 640, 3])
        final_shape = metadata.get('final_shape', [480, 640, 3])
        
        orig_height, orig_width = original_shape[:2]
        final_height, final_width = final_shape[:2]
        
        # Convert to YOLO format
        yolo_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2 / final_width
            y_center = (y1 + y2) / 2 / final_height
            width = (x2 - x1) / final_width
            height = (y2 - y1) / final_height
            yolo_boxes.append([x_center, y_center, width, height])
        
        yolo_boxes = np.array(yolo_boxes)
        
        # Apply transformations in reverse order
        for transform_info in reversed(metadata.get('applied_transforms', [])):
            transform_name = transform_info['name']
            
            if transform_name == 'horizontal_flip':
                # Flip x coordinates
                yolo_boxes[:, 0] = 1.0 - yolo_boxes[:, 0]
                
            elif transform_name == 'rotation':
                # Apply rotation based on angle
                angle = transform_info['angle']
                
                if angle == 90:
                    # Rotate 90 degrees clockwise
                    x_center = yolo_boxes[:, 0]
                    y_center = yolo_boxes[:, 1]
                    width = yolo_boxes[:, 2]
                    height = yolo_boxes[:, 3]
                    
                    new_x_center = y_center
                    new_y_center = 1.0 - x_center
                    new_width = height
                    new_height = width
                    
                    yolo_boxes[:, 0] = new_x_center
                    yolo_boxes[:, 1] = new_y_center
                    yolo_boxes[:, 2] = new_width
                    yolo_boxes[:, 3] = new_height
                    
                elif angle == 180:
                    # Rotate 180 degrees
                    yolo_boxes[:, 0] = 1.0 - yolo_boxes[:, 0]
                    yolo_boxes[:, 1] = 1.0 - yolo_boxes[:, 1]
                    
                elif angle == 270:
                    # Rotate 270 degrees clockwise (90 degrees counter-clockwise)
                    x_center = yolo_boxes[:, 0]
                    y_center = yolo_boxes[:, 1]
                    width = yolo_boxes[:, 2]
                    height = yolo_boxes[:, 3]
                    
                    new_x_center = 1.0 - y_center
                    new_y_center = x_center
                    new_width = height
                    new_height = width
                    
                    yolo_boxes[:, 0] = new_x_center
                    yolo_boxes[:, 1] = new_y_center
                    yolo_boxes[:, 2] = new_width
                    yolo_boxes[:, 3] = new_height
                
                # angle == 0 means no rotation, so no changes needed
        
        # Convert back to absolute coordinates for the original image
        corrected_boxes = []
        for box in yolo_boxes:
            x_center, y_center, width, height = box
            x1 = (x_center - width/2) * orig_width
            y1 = (y_center - height/2) * orig_height
            x2 = (x_center + width/2) * orig_width
            y2 = (y_center + height/2) * orig_height
            corrected_boxes.append([x1, y1, x2, y2])
        
        return np.array(corrected_boxes)
    
    def predict_image_from_array(self, image: np.ndarray) -> Dict:
        """
        Predict on a numpy array image (for testing)
        Returns: Dict with predictions and metadata
        """
        print(f"Processing image array with shape: {image.shape}")
        
        if self.model_type in ['yolov5', 'yolov8']:
            return self._predict_yolo(image)
        elif self.model_type == 'detectron2':
            return self._predict_detectron2(image)
        elif self.model_type == 'sam':
            return self._predict_sam(image)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _predict_yolo(self, image: np.ndarray) -> Dict:
        """Handle YOLO model predictions"""
        print(f"Running YOLO prediction with model type: {self.model_type}")
        
        if self.model_type == 'yolov8':
            results = self.model(image)[0]
            print(f"YOLOv8 results: {results}")
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                print(f"YOLOv8 detections: {len(boxes)} objects")
            else:
                print("No detections found in YOLOv8 results")
                boxes = np.array([])
                scores = np.array([])
                class_ids = np.array([])
                
        else:  # YOLOv5
            results = self.model(image)
            print(f"YOLOv5 results shape: {results.xyxy[0].shape if len(results.xyxy) > 0 else 'No detections'}")
            
            if len(results.xyxy) > 0 and len(results.xyxy[0]) > 0:
                boxes = results.xyxy[0].cpu().numpy()
                scores = boxes[:, 4]
                class_ids = boxes[:, 5].astype(int)
                boxes = boxes[:, :4]
                print(f"YOLOv5 detections: {len(boxes)} objects")
            else:
                print("No detections found in YOLOv5 results")
                boxes = np.array([])
                scores = np.array([])
                class_ids = np.array([])
            
        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'type': 'detection'
        }
    
    def _predict_detectron2(self, image: np.ndarray) -> Dict:
        """Handle Detectron2 model predictions"""
        outputs = self.model(image)
        
        if "instances" in outputs:
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            class_ids = instances.pred_classes.numpy()
            
            # If segmentation masks are available
            if instances.has("pred_masks"):
                masks = instances.pred_masks.numpy()
                return {
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids,
                    'masks': masks,
                    'type': 'instance_segmentation'
                }
            
            return {
                'boxes': boxes,
                'scores': scores,
                'class_ids': class_ids,
                'type': 'detection'
            }
        
        return {}
    
    def _predict_sam(self, image: np.ndarray) -> Dict:
        """Handle SAM model predictions"""
        self.model.set_image(image)
        
        # Generate automatic mask predictions
        masks, scores, logits = self.model.predict()
        
        return {
            'masks': masks,
            'scores': scores,
            'type': 'segmentation'
        }
    
    def save_predictions(self, 
                        image_path: Union[str, Path],
                        predictions: Dict,
                        output_dir: Union[str, Path],
                        label_name: str = 'object',
                        save_visualizations: bool = True) -> Tuple[str, str]:
        """
        Save predictions as YOLO format labels and optionally save visualization
        Returns: Tuple[label_path, vis_path]
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save label file
        label_path = output_dir / f"{image_path.stem}.txt"
        
        if predictions['type'] in ['detection', 'instance_segmentation']:
            self._save_yolo_labels(predictions, label_path, image_path, label_name)
        
        # Save visualization if requested
        vis_path = None
        if save_visualizations:
            vis_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
            self._save_visualization(image_path, predictions, vis_path)
            
        return str(label_path), str(vis_path) if vis_path else None
    
    def save_labels_only(self, 
                        predictions: Dict,
                        label_path: Union[str, Path],
                        label_name: str = 'object') -> str:
        """
        Save only the label file without creating visualizations
        Returns: label_path
        """
        label_path = Path(label_path)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        
        if predictions['type'] in ['detection', 'instance_segmentation']:
            # For augmented images, we need to get image dimensions from the predictions
            # or use a default size since we don't have the original image path
            self._save_yolo_labels_only(predictions, label_path, label_name)
            
        return str(label_path)
    
    def _save_yolo_labels(self, predictions: Dict, label_path: Path, image_path: Path, label_name: str = 'object') -> None:
        """Save predictions in YOLO format with custom label name"""
        print(f"Saving labels to: {label_path}")
        print(f"Predictions: {predictions}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not load image for label saving: {image_path}")
            # Create empty file if image cannot be loaded
            with open(label_path, 'w') as f:
                pass
            return
            
        img_height, img_width = image.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
        
        boxes = predictions.get('boxes', [])
        class_ids = predictions.get('class_ids', [])
        scores = predictions.get('scores', [])
        
        print(f"Number of detections: {len(boxes)}")
        print(f"Class IDs: {class_ids}")
        print(f"Scores: {scores}")
        
        # Always create the label file, even if empty
        with open(label_path, 'w') as f:
            if len(boxes) > 0:
                for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
                    # Convert box to YOLO format (x_center, y_center, width, height)
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write YOLO format line: class x_center y_center width height confidence
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                    f.write(line)
                    print(f"Written line {i+1}: {line.strip()}")
                print(f"Label file saved with {len(boxes)} detections")
            else:
                print("No detections found - created empty label file")
        
        print(f"Label file created: {label_path}")
    
    def _save_yolo_labels_only(self, predictions: Dict, label_path: Path, label_name: str = 'object') -> None:
        """Save predictions in YOLO format without requiring original image"""
        print(f"Saving labels only to: {label_path}")
        print(f"Predictions: {predictions}")
        
        boxes = predictions.get('boxes', [])
        class_ids = predictions.get('class_ids', [])
        scores = predictions.get('scores', [])
        
        print(f"Number of detections: {len(boxes)}")
        print(f"Class IDs: {class_ids}")
        print(f"Scores: {scores}")
        
        # For augmented images, we assume the image dimensions are the same as the augmented image
        # This is a reasonable assumption since augmentation typically preserves image size
        # If needed, this can be made configurable
        img_width = 640  # Default width, can be made configurable
        img_height = 480  # Default height, can be made configurable
        
        # Always create the label file, even if empty
        with open(label_path, 'w') as f:
            if len(boxes) > 0:
                for i, (box, class_id, score) in enumerate(zip(boxes, class_ids, scores)):
                    # Convert box to YOLO format (x_center, y_center, width, height)
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2 / img_width
                    y_center = (y1 + y2) / 2 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Write YOLO format line: class x_center y_center width height confidence
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                    f.write(line)
                    print(f"Written line {i+1}: {line.strip()}")
                print(f"Label file saved with {len(boxes)} detections")
            else:
                print("No detections found - created empty label file")
        
        print(f"Label file created: {label_path}")
    
    def _save_visualization(self, image_path: Path, predictions: Dict, output_path: Path) -> None:
        """Save visualization of predictions"""
        image = cv2.imread(str(image_path))
        
        if predictions['type'] == 'detection':
            # Draw boxes
            for box, score, class_id in zip(predictions['boxes'], 
                                          predictions['scores'],
                                          predictions['class_ids']):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"Class {class_id} ({score:.2f})",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, (0, 255, 0), 2)
                
        elif predictions['type'] == 'instance_segmentation':
            # Draw masks and boxes
            for mask, box, score, class_id in zip(predictions['masks'],
                                                predictions['boxes'],
                                                predictions['scores'],
                                                predictions['class_ids']):
                color = np.random.randint(0, 255, 3).tolist()
                image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
                
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"Class {class_id} ({score:.2f})",
                          (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                          0.5, color, 2)
                
        elif predictions['type'] == 'segmentation':
            # Draw SAM masks
            for mask, score in zip(predictions['masks'], predictions['scores']):
                color = np.random.randint(0, 255, 3).tolist()
                image[mask] = image[mask] * 0.5 + np.array(color) * 0.5
                
        cv2.imwrite(str(output_path), image) 