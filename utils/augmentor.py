import cv2
import numpy as np
import albumentations as A
from pathlib import Path
from typing import List, Dict, Union, Tuple

class Augmentor:
    """Handles image augmentation with bounding box/mask support"""
    
    def __init__(self):
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.RGBShift(p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_ids']))
        
    def augment_detection(self, 
                         image: np.ndarray,
                         boxes: np.ndarray,
                         class_ids: np.ndarray,
                         num_augmentations: int = 3) -> List[Dict]:
        """
        Augment image with bounding boxes
        Returns: List of dicts with augmented images and boxes
        """
        results = []
        
        for _ in range(num_augmentations):
            # Convert boxes from YOLO format to albumentations format
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_ids=class_ids.tolist()
            )
            
            results.append({
                'image': transformed['image'],
                'boxes': np.array(transformed['bboxes']),
                'class_ids': np.array(transformed['class_ids'])
            })
            
        return results
    
    def augment_segmentation(self,
                           image: np.ndarray,
                           masks: np.ndarray,
                           num_augmentations: int = 3) -> List[Dict]:
        """
        Augment image with segmentation masks
        Returns: List of dicts with augmented images and masks
        """
        # Create a transform that includes mask augmentation
        transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussNoise(p=0.2),
            A.RGBShift(p=0.2),
            A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        ])
        
        results = []
        
        for _ in range(num_augmentations):
            transformed = transform(image=image, masks=[m.astype(np.uint8) for m in masks])
            
            results.append({
                'image': transformed['image'],
                'masks': np.array(transformed['masks'])
            })
            
        return results
    
    def save_augmented_results(self,
                             base_image_path: Union[str, Path],
                             augmented_results: List[Dict],
                             output_dir: Union[str, Path],
                             prediction_type: str) -> List[Tuple[str, str]]:
        """
        Save augmented images and their corresponding labels/masks
        Returns: List of tuples (image_path, label_path)
        """
        base_image_path = Path(base_image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for i, result in enumerate(augmented_results):
            # Save augmented image
            image_filename = f"{base_image_path.stem}_aug{i}{base_image_path.suffix}"
            image_path = output_dir / image_filename
            cv2.imwrite(str(image_path), cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR))
            
            # Save corresponding labels/masks
            if prediction_type in ['detection', 'instance_segmentation']:
                label_filename = f"{base_image_path.stem}_aug{i}.txt"
                label_path = output_dir / label_filename
                
                # Get image dimensions for coordinate conversion
                img_height, img_width = result['image'].shape[:2]
                
                # Debug: Print box info
                print(f"[AUG] {label_path}: {len(result['boxes'])} boxes")
                if len(result['boxes']) == 0:
                    print(f"[AUG] No boxes for {label_path}, skipping label file.")
                    continue
                
                # Save YOLO format labels with confidence scores
                with open(label_path, 'w') as f:
                    for box, class_id in zip(result['boxes'], result['class_ids']):
                        f.write(f"{class_id} {' '.join(map(str, box))} 1.0\n")  # Default confidence 1.0 for augmented
                print(f"[AUG] Label file created: {label_path}")
                saved_paths.append((str(image_path), str(label_path)))
                
            elif prediction_type == 'segmentation':
                mask_filename = f"{base_image_path.stem}_aug{i}_mask.npz"
                mask_path = output_dir / mask_filename
                
                # Save masks as compressed numpy array
                np.savez_compressed(str(mask_path), masks=result['masks'])
                
                saved_paths.append((str(image_path), str(mask_path)))
                
        return saved_paths 