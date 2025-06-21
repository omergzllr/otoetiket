import cv2
import numpy as np
import albumentations as A
from pathlib import Path
import json
import os
from typing import List, Dict, Union, Tuple, Any

class SmartAugmentor:
    """Smart augmentation system that tracks transformations and applies them to labels"""
    
    def __init__(self):
        # Define individual transforms that we can track
        self.transforms = {
            'horizontal_flip': A.HorizontalFlip(p=0.5),
            'brightness_contrast': A.RandomBrightnessContrast(p=0.2),
            'gauss_noise': A.GaussNoise(p=0.2),
            'rgb_shift': A.RGBShift(p=0.2),
            'blur': A.OneOf([
                A.Blur(blur_limit=3, p=0.5),
                A.MedianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
        }
        
    def create_augmentation_chain(self, num_augmentations: int = 3) -> List[Dict]:
        """Create a list of augmentation chains with tracked transformations"""
        chains = []
        
        for i in range(num_augmentations):
            # Randomly select which transformations to apply
            applied_transforms = []
            
            # Apply rotation (0, 90, 180, 270 degrees)
            rotation_angle = np.random.choice([0, 90, 180, 270])
            applied_transforms.append(('rotation', rotation_angle))
            
            # Apply horizontal flip
            if np.random.random() < 0.5:
                applied_transforms.append(('horizontal_flip', self.transforms['horizontal_flip']))
            
            # Apply color/quality transformations
            if np.random.random() < 0.3:
                applied_transforms.append(('brightness_contrast', self.transforms['brightness_contrast']))
            
            if np.random.random() < 0.3:
                applied_transforms.append(('gauss_noise', self.transforms['gauss_noise']))
            
            if np.random.random() < 0.3:
                applied_transforms.append(('rgb_shift', self.transforms['rgb_shift']))
            
            if np.random.random() < 0.2:
                applied_transforms.append(('blur', self.transforms['blur']))
            
            chains.append({
                'chain_id': i,
                'transforms': applied_transforms,
                'transform_names': [name for name, _ in applied_transforms]
            })
        
        return chains
    
    def apply_augmentation_chain(self, image: np.ndarray, chain: Dict) -> Tuple[np.ndarray, Dict]:
        """Apply a specific augmentation chain and return transformed image with metadata"""
        transformed_image = image.copy()
        metadata = {
            'original_shape': image.shape,
            'applied_transforms': [],
            'transform_sequence': []
        }
        
        for transform_name, transform in chain['transforms']:
            # Apply the transformation
            if transform_name == 'rotation':
                # Apply specific rotation angle
                angle = transform
                if angle == 90:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_180)
                elif angle == 270:
                    transformed_image = cv2.rotate(transformed_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                # angle == 0 means no rotation
                
                metadata['applied_transforms'].append({
                    'name': 'rotation',
                    'angle': angle
                })
                metadata['transform_sequence'].append(f'rotation{angle}')
                
            elif transform_name == 'horizontal_flip':
                result = transform(image=transformed_image)
                transformed_image = result['image']
                metadata['applied_transforms'].append({
                    'name': 'horizontal_flip',
                    'applied': True
                })
                metadata['transform_sequence'].append('horizontal_flip')
                
            else:
                # For other transformations, just apply them
                result = transform(image=transformed_image)
                transformed_image = result['image']
                metadata['applied_transforms'].append({
                    'name': transform_name,
                    'applied': True
                })
                metadata['transform_sequence'].append(transform_name)
        
        metadata['final_shape'] = transformed_image.shape
        return transformed_image, metadata
    
    def augment_image_with_tracking(self, 
                                  image_path: str, 
                                  num_augmentations: int = 3,
                                  output_dir: str = None) -> List[Dict]:
        """Augment an image and track all transformations"""
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create augmentation chains
        chains = self.create_augmentation_chain(num_augmentations)
        
        results = []
        base_name = Path(image_path).stem
        
        for chain in chains:
            # Apply augmentation chain
            augmented_image, metadata = self.apply_augmentation_chain(image, chain)
            
            # Create descriptive filename
            transform_suffix = '_'.join(metadata['transform_sequence'])
            if not transform_suffix:
                transform_suffix = 'no_transform'
            
            filename = f"{base_name}_aug{chain['chain_id']}_{transform_suffix}"
            
            # Save augmented image
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                image_filename = f"{filename}.jpg"
                image_path_out = os.path.join(output_dir, image_filename)
                
                # Convert back to BGR for saving
                augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path_out, augmented_image_bgr)
                
                # Save metadata (convert numpy types to Python types for JSON serialization)
                metadata_filename = f"{filename}_metadata.json"
                metadata_path = os.path.join(output_dir, metadata_filename)
                
                # Convert metadata to JSON-serializable format
                json_metadata = self._convert_metadata_to_json(metadata)
                
                with open(metadata_path, 'w') as f:
                    json.dump(json_metadata, f, indent=2)
                
                results.append({
                    'image_path': image_path_out,
                    'metadata_path': metadata_path,
                    'filename': filename,
                    'metadata': metadata,
                    'augmented_image': augmented_image
                })
            else:
                results.append({
                    'filename': filename,
                    'metadata': metadata,
                    'augmented_image': augmented_image
                })
        
        return results
    
    def _convert_metadata_to_json(self, metadata: Dict) -> Dict:
        """Convert metadata to JSON-serializable format"""
        json_metadata = {}
        
        for key, value in metadata.items():
            if key == 'original_shape':
                json_metadata[key] = [int(x) for x in value]
            elif key == 'final_shape':
                json_metadata[key] = [int(x) for x in value]
            elif key == 'applied_transforms':
                json_metadata[key] = []
                for transform in value:
                    json_transform = {}
                    for t_key, t_value in transform.items():
                        if isinstance(t_value, (np.integer, np.floating)):
                            json_transform[t_key] = t_value.item()
                        else:
                            json_transform[t_key] = t_value
                    json_metadata[key].append(json_transform)
            elif key == 'transform_sequence':
                json_metadata[key] = [str(x) for x in value]
            else:
                json_metadata[key] = value
        
        return json_metadata
    
    def transform_coordinates(self, boxes: np.ndarray, metadata: Dict) -> np.ndarray:
        """Transform bounding box coordinates based on augmentation metadata"""
        if not boxes.size:
            return boxes
        
        transformed_boxes = boxes.copy()
        original_shape = metadata['original_shape']
        final_shape = metadata['final_shape']
        
        # Apply transformations in reverse order (as they were applied to the image)
        for transform_info in reversed(metadata['applied_transforms']):
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
    
    def load_metadata(self, metadata_path: str) -> Dict:
        """Load metadata from JSON file"""
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def get_augmentation_info_from_filename(self, filename: str) -> Dict:
        """Extract augmentation information from filename"""
        # Expected format: base_augN_transform1_transform2_...
        parts = filename.split('_')
        
        info = {
            'is_augmented': False,
            'augmentation_id': None,
            'transforms': []
        }
        
        for i, part in enumerate(parts):
            if part.startswith('aug') and part[3:].isdigit():
                info['is_augmented'] = True
                info['augmentation_id'] = int(part[3:])
                # Get transforms after this part
                info['transforms'] = parts[i+1:]
                break
        
        return info 