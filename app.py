import os
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import zipfile
import rarfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import uuid
import albumentations as A
import traceback
import yaml
import yaml
import random
import json

from utils.model_loader import ModelLoader
from utils.predictor import Predictor
from utils.augmentor import Augmentor
from utils.smart_augmentor import SmartAugmentor
from utils.label_transfer import LabelTransfer
from utils.accuracy_improver import AccuracyImprover


print("deneme commit")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.secret_key = 'your-secret-key-here'  # Required for session

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'model': {'pt', 'pth', 'pkl', 'onnx'},
    'image': {'png', 'jpg', 'jpeg'},
    'archive': {'zip', 'rar'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def extract_archive(file_path):
    """Extract uploaded ZIP/RAR archive to uploads folder"""
    extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
    os.makedirs(extract_path, exist_ok=True)
    
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    elif file_path.endswith('.rar'):
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(extract_path)
            
    return extract_path

def get_image_files(folder_path):
    """Get all image files from extracted folder"""
    image_files = []
    for ext in ALLOWED_EXTENSIONS['image']:
        image_files.extend(Path(folder_path).rglob(f'*.{ext}'))
    return [str(f) for f in image_files]

def create_zip_archive(directory, output_path):
    """Create a ZIP archive of a directory"""
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory)
                zipf.write(file_path, arcname)

def test_model_performance(model, test_image_path):
    """Test model performance with different confidence thresholds"""
    print("=== Testing Model Performance ===")
    
    try:
        from ultralytics import YOLO
        
        # Load test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print("Warning: Could not load test image for performance testing")
            return {}
        
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        # Test different confidence thresholds
        performance_results = {}
        for conf_threshold in [0.1, 0.3, 0.5, 0.7]:
            try:
                results = model(test_image, conf=conf_threshold)
                detections = len(results[0].boxes) if results[0].boxes is not None else 0
                performance_results[f'conf_{conf_threshold}'] = detections
                print(f"Confidence {conf_threshold}: {detections} detections")
            except Exception as e:
                print(f"Error testing confidence {conf_threshold}: {e}")
                performance_results[f'conf_{conf_threshold}'] = 0
        
        return performance_results
        
    except Exception as e:
        print(f"Error in model performance testing: {e}")
        return {}

def optimize_confidence_threshold(model, image_files, initial_threshold=0.5):
    """Automatically optimize confidence threshold based on detection results"""
    print("=== Optimizing Confidence Threshold ===")
    
    if not image_files:
        return initial_threshold
    
    # Test with a sample image
    sample_image = image_files[0]
    test_image = cv2.imread(sample_image)
    if test_image is None:
        return initial_threshold
    
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
    
    # Test different thresholds
    best_threshold = initial_threshold
    best_detections = 0
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        try:
            results = model(test_image, conf=threshold)
            detections = len(results[0].boxes) if results[0].boxes is not None else 0
            
            if detections > best_detections:
                best_detections = detections
                best_threshold = threshold
                
            print(f"Threshold {threshold}: {detections} detections")
            
        except Exception as e:
            print(f"Error testing threshold {threshold}: {e}")
            continue
    
    print(f"Optimized threshold: {best_threshold} with {best_detections} detections")
    return best_threshold

@app.route('/', methods=['GET', 'POST'])
def index():
    # Always clear ALL session data on GET requests to prevent automatic processing
    if request.method == 'GET':
        session.clear()  # Clear all session data
        return render_template('index.html')
    
    # POST request - only process if form is actually submitted
    if request.method == 'POST':
        try:
            # Check if we're already processing to prevent loops
            if session.get('processing'):
                return render_template('index.html', 
                                     messages=[{'type': 'warning', 'text': 'Processing already in progress. Please wait.'}])
            
            # Check if form data is actually present
            model_file = request.files.get('model_file')
            data_file = request.files.get('data_file')
            
            if not model_file or not data_file:
                return render_template('index.html', 
                                     messages=[{'type': 'danger', 'text': 'Please upload both model and data files.'}])
            
            # Set processing flag
            session['processing'] = True
            
            # Get form data
            model_type = request.form.get('model_type', 'auto')
            label_name = request.form.get('label_name', 'object')
            confidence_threshold = float(request.form.get('confidence_threshold', 0.3))
            augmentation_count = int(request.form.get('augmentation_count', 3))
            do_augmentation = request.form.get('do_augmentation') == 'on'
            auto_optimize = request.form.get('auto_optimize') == 'on'  # New option
            use_accuracy_improvement = request.form.get('use_accuracy_improvement') == 'on'  # New accuracy improvement option
            
            # Generate unique session ID for this processing job
            session_id = str(uuid.uuid4())
            session['processing_id'] = session_id
            
            print(f"=== Starting Processing Session {session_id} ===")
            print(f"Model type: {model_type}")
            print(f"Label name: {label_name}")
            print(f"Initial confidence threshold: {confidence_threshold}")
            print(f"Augmentation: {do_augmentation} ({augmentation_count} images)")
            print(f"Auto optimize: {auto_optimize}")
            print(f"Accuracy improvement: {use_accuracy_improvement}")
            
            # Save model file
            model_filename = secure_filename(model_file.filename)
            os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
            model_path = os.path.join(app.config['MODEL_FOLDER'], model_filename)
            model_file.save(model_path)
            print(f"Model saved: {model_path}")
            
            # Save and extract data file
            data_filename = secure_filename(data_file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            data_path = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
            data_file.save(data_path)
            print(f"Data saved: {data_path}")
            
            # Extract archive if it's a ZIP/RAR file
            if data_filename.endswith(('.zip', '.rar')):
                extract_path = extract_archive(data_path)
                print(f"Archive extracted to: {extract_path}")
            else:
                # Single file upload
                extract_path = os.path.join(app.config['UPLOAD_FOLDER'], 'extracted')
                os.makedirs(extract_path, exist_ok=True)
                shutil.copy2(data_path, extract_path)
                print(f"Single file copied to: {extract_path}")
            
            # Get image files
            image_files = get_image_files(extract_path)
            print(f"Found {len(image_files)} image files")
            
            if not image_files:
                session.pop('processing', None)  # Clear processing flag
                return render_template('index.html', 
                                     messages=[{'type': 'danger', 'text': 'No image files found in uploaded data.'}])
            
            # Load model with enhanced error handling
            try:
                print("Loading model...")
                model_loader = ModelLoader(model_path)
                model = model_loader.load_model()
                detected_model_type = model_loader.get_model_type()
                print(f"Model loaded successfully. Type: {detected_model_type}")
                
                # Test model performance
                if len(image_files) > 0:
                    performance_results = test_model_performance(model, image_files[0])
                    print(f"Performance test results: {performance_results}")
                
                # Auto-optimize confidence threshold if requested
                if auto_optimize:
                    optimized_threshold = optimize_confidence_threshold(model, image_files, confidence_threshold)
                    if optimized_threshold != confidence_threshold:
                        print(f"Auto-optimized threshold: {confidence_threshold} -> {optimized_threshold}")
                        confidence_threshold = optimized_threshold
                
            except Exception as model_error:
                print(f"Error loading model: {model_error}")
                traceback.print_exc()
                session.pop('processing', None)
                return render_template('index.html', 
                                     messages=[{'type': 'danger', 'text': f'Error loading model: {str(model_error)}'}])
            
            # Setup predictor and augmentor
            predictor = Predictor(model, detected_model_type)
            smart_augmentor = SmartAugmentor() if do_augmentation else None
            label_transfer = LabelTransfer() if do_augmentation else None
            accuracy_improver = AccuracyImprover() if use_accuracy_improvement else None
            
            # Create output directories
            aug_images_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_images')
            aug_labels_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels')
            os.makedirs(aug_images_dir, exist_ok=True)
            os.makedirs(aug_labels_dir, exist_ok=True)
            
            # Step 1: Perform smart augmentation on all images first
            all_augmented_images = []
            all_augmented_metadata = {}  # Store metadata for each augmented image
            total_augmented = 0
            
            if do_augmentation:
                print("=== STEP 1: PERFORMING SMART AUGMENTATION ===")
                for image_file in image_files:
                    try:
                        print(f"\n--- Smart Augmenting {image_file} ---")
                        
                        # Use smart augmentation with tracking
                        augmentation_results = smart_augmentor.augment_image_with_tracking(
                            image_file, 
                            augmentation_count, 
                            aug_images_dir
                        )
                        
                        for result in augmentation_results:
                            all_augmented_images.append(result['image_path'])
                            all_augmented_metadata[result['image_path']] = result['metadata_path']
                            total_augmented += 1
                            
                            print(f"Created: {result['filename']} with transforms: {result['metadata']['transform_sequence']}")
                            
                        print(f"Created {len(augmentation_results)} augmented images for {os.path.basename(image_file)}")
                        
                    except Exception as e:
                        print(f"Error during smart augmentation of {image_file}: {e}")
                        traceback.print_exc()
                        continue
                
                print(f"\n=== SMART AUGMENTATION COMPLETE: {total_augmented} images created ===")
            
            # Step 2: First process original images to get their labels
            print(f"\n=== STEP 2: PROCESSING ORIGINAL IMAGES ===")
            original_labels_map = {}  # Map original image paths to their label paths
            
            for image_file in image_files:
                try:
                    print(f"\n--- Processing Original Image: {os.path.basename(image_file)} ---")
                    
                    # Make predictions on original image
                    if use_accuracy_improvement and accuracy_improver:
                        print("Using accuracy improvement for predictions...")
                        predictions = accuracy_improver.improve_predictions(model, image_file, confidence_threshold)
                    else:
                        predictions = predictor.predict_image(image_file)
                    print(f"Raw predictions: {len(predictions.get('boxes', []))} detections")
                    
                    # Filter predictions based on confidence threshold
                    if predictions['type'] in ['detection', 'instance_segmentation']:
                        original_count = len(predictions['boxes'])
                        mask = predictions['scores'] >= confidence_threshold
                        predictions['boxes'] = predictions['boxes'][mask]
                        predictions['scores'] = predictions['scores'][mask]
                        predictions['class_ids'] = predictions['class_ids'][mask]
                        if 'masks' in predictions:
                            predictions['masks'] = predictions['masks'][mask]
                        
                        filtered_count = len(predictions['boxes'])
                        print(f"After confidence filtering: {filtered_count}/{original_count} detections")
                        
                        # Log confidence scores for debugging
                        if len(predictions['scores']) > 0:
                            print(f"Confidence scores: {predictions['scores']}")
                            print(f"Average confidence: {np.mean(predictions['scores']):.3f}")
                    
                    # Save original image predictions
                    output_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels')
                    os.makedirs(output_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(image_file))[0]
                    pred_images_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'original_images')
                    os.makedirs(pred_images_dir, exist_ok=True)
                    pred_image_path = os.path.join(pred_images_dir, f"{base_name}_pred.jpg")
                    pred_label_path = os.path.join(output_dir, f"{base_name}.txt")
                    
                    # Save predictions
                    try:
                        predictor.save_predictions(image_file, predictions, pred_images_dir, label_name, True)
                        print(f"Saved predictions to: {pred_images_dir}")
                    except Exception as save_error:
                        print(f"Error saving predictions: {save_error}")
                        # Try to save labels only as fallback
                        try:
                            predictor.save_labels_only(predictions, pred_label_path, label_name)
                            print(f"Saved labels only as fallback: {pred_label_path}")
                        except Exception as label_error:
                            print(f"Error saving labels only: {label_error}")
                            # Create empty label file if all else fails
                            with open(pred_label_path, 'w') as f:
                                pass
                            print(f"Created empty label file: {pred_label_path}")
                    
                    # Verify that label file was actually created
                    if os.path.exists(pred_label_path):
                        # Count lines in label file
                        with open(pred_label_path, 'r') as f:
                            lines = f.readlines()
                            label_count = len([line for line in lines if line.strip()])
                        print(f"Label file created with {label_count} labels")
                        
                        # If no labels were found, try with lower confidence threshold
                        if label_count == 0 and len(predictions.get('boxes', [])) > 0:
                            print("No labels saved, trying with lower confidence threshold...")
                            # Use a very low threshold to ensure we get some labels
                            low_threshold = 0.1
                            mask = predictions['scores'] >= low_threshold
                            low_conf_predictions = {
                                'boxes': predictions['boxes'][mask],
                                'scores': predictions['scores'][mask],
                                'class_ids': predictions['class_ids'][mask],
                                'type': predictions['type']
                            }
                            
                            try:
                                predictor.save_labels_only(low_conf_predictions, pred_label_path, label_name)
                                with open(pred_label_path, 'r') as f:
                                    lines = f.readlines()
                                    label_count = len([line for line in lines if line.strip()])
                                print(f"Retried with low threshold: {label_count} labels")
                            except Exception as retry_error:
                                print(f"Retry failed: {retry_error}")
                    else:
                        print(f"Warning: Label file was not created: {pred_label_path}")
                        # Create empty file to prevent errors
                        with open(pred_label_path, 'w') as f:
                            pass
                        print(f"Created empty label file as fallback")
                    
                    # Store label path for later use
                    original_labels_map[image_file] = pred_label_path
                    
                    # Validate labels if accuracy improvement is enabled
                    if use_accuracy_improvement and accuracy_improver:
                        validation_result = accuracy_improver.validate_labels(pred_label_path, image_file)
                        if not validation_result['valid']:
                            print(f"Label validation issues: {validation_result['issues']}")
                        if validation_result['warnings']:
                            print(f"Label validation warnings: {validation_result['warnings']}")
                        print(f"Label validation: {validation_result['total_labels']} labels validated")
                    
                    print(f"Saved original labels: {pred_label_path}")
                    
                except Exception as e:
                    print(f"Error processing original image {image_file}: {e}")
                    traceback.print_exc()
                    continue
                    
            print(f"\nOriginal labels map contains {len(original_labels_map)} entries")
            for img_path, label_path in list(original_labels_map.items())[:3]:  # Show first 3
                print(f"  {os.path.basename(img_path)} -> {os.path.basename(label_path)}")
            
            # Step 3: Process augmented images using label transfer
            print(f"\n=== STEP 3: PROCESSING AUGMENTED IMAGES WITH LABEL TRANSFER ===")
            
            if do_augmentation:
                for aug_image_path in all_augmented_images:
                    try:
                        print(f"\n--- Processing Augmented Image: {os.path.basename(aug_image_path)} ---")
                        
                        # Find corresponding original image
                        aug_filename = os.path.basename(aug_image_path)
                        
                        # Extract base name more robustly
                        # Handle different naming patterns
                        if '_aug' in aug_filename:
                            base_name = aug_filename.split('_aug')[0]
                        else:
                            # Fallback: try to extract base name before any augmentation indicators
                            base_name = aug_filename
                            for suffix in ['_rotation', '_horizontal_flip', '_brightness_contrast', '_gauss_noise', '_rgb_shift', '_blur']:
                                if suffix in base_name:
                                    base_name = base_name.split(suffix)[0]
                                    break
                        
                        print(f"Looking for original image with base name: {base_name}")
                        
                        # Find original image path - try multiple matching strategies
                        original_image_path = None
                        
                        # Strategy 1: Exact match
                        for orig_path in image_files:
                            orig_basename = os.path.splitext(os.path.basename(orig_path))[0]
                            if orig_basename == base_name:
                                original_image_path = orig_path
                                print(f"Found exact match: {orig_basename}")
                                break
                        
                        # Strategy 2: Partial match (if exact match fails)
                        if not original_image_path:
                            for orig_path in image_files:
                                orig_basename = os.path.splitext(os.path.basename(orig_path))[0]
                                # Check if base_name is contained in original basename
                                if base_name in orig_basename or orig_basename in base_name:
                                    original_image_path = orig_path
                                    print(f"Found partial match: {orig_basename} (base: {base_name})")
                                    break
                        
                        # Strategy 3: Try to match by removing common suffixes
                        if not original_image_path:
                            # Remove common suffixes from base_name
                            clean_base = base_name
                            for suffix in ['.rf.', '.png', '.jpg', '.jpeg']:
                                if suffix in clean_base:
                                    clean_base = clean_base.split(suffix)[0]
                            
                            for orig_path in image_files:
                                orig_basename = os.path.splitext(os.path.basename(orig_path))[0]
                                orig_clean = orig_basename
                                for suffix in ['.rf.', '.png', '.jpg', '.jpeg']:
                                    if suffix in orig_clean:
                                        orig_clean = orig_clean.split(suffix)[0]
                                
                                if clean_base == orig_clean:
                                    original_image_path = orig_path
                                    print(f"Found cleaned match: {orig_basename} (clean: {clean_base})")
                                    break
                        
                        if not original_image_path:
                            print(f"Warning: Could not find original image for {aug_filename}")
                            print(f"Available original images: {[os.path.basename(f) for f in image_files[:5]]}...")
                            continue
                        
                        # Get original label path
                        original_label_path = original_labels_map.get(original_image_path)
                        if not original_label_path or not os.path.exists(original_label_path):
                            print(f"Warning: No original labels found for {aug_filename}")
                            print(f"Original image: {original_image_path}")
                            print(f"Expected label path: {original_label_path}")
                            
                            # Fallback: Use model predictions directly for augmented image
                            print(f"Using fallback: direct model prediction for augmented image")
                            model_predictions = predictor.predict_image(aug_image_path)
                            if model_predictions['type'] in ['detection', 'instance_segmentation']:
                                original_count = len(model_predictions['boxes'])
                                mask = model_predictions['scores'] >= confidence_threshold
                                model_predictions['boxes'] = model_predictions['boxes'][mask]
                                model_predictions['scores'] = model_predictions['scores'][mask]
                                model_predictions['class_ids'] = model_predictions['class_ids'][mask]
                                
                                # Convert model predictions to label format
                                model_labels = []
                                for box, class_id, score in zip(model_predictions['boxes'], 
                                                              model_predictions['class_ids'], 
                                                              model_predictions['scores']):
                                    x1, y1, x2, y2 = box
                                    img_height, img_width = cv2.imread(aug_image_path).shape[:2]
                                    x_center = (x1 + x2) / 2 / img_width
                                    y_center = (y1 + y2) / 2 / img_height
                                    width = (x2 - x1) / img_width
                                    height = (y2 - y1) / img_height
                                    model_labels.append([class_id, x_center, y_center, width, height])
                                
                                # Save model predictions as labels
                                aug_base_name = os.path.splitext(os.path.basename(aug_image_path))[0]
                                aug_label_path = os.path.join(aug_labels_dir, f"{aug_base_name}.txt")
                                label_transfer.save_transferred_labels(model_labels, aug_label_path)
                                
                                print(f"Saved {len(model_labels)} model predictions as labels: {aug_label_path}")
                            else:
                                print(f"No valid predictions from model for {aug_filename}")
                                continue
                        else:
                            # Get metadata path
                            metadata_path = all_augmented_metadata.get(aug_image_path)
                            if not metadata_path or not os.path.exists(metadata_path):
                                print(f"Warning: No metadata found for {aug_filename}")
                                continue
                            
                            # Transfer labels from original to augmented image
                            print(f"Transferring labels from {os.path.basename(original_image_path)}")
                            transferred_labels = label_transfer.transfer_labels_from_original(
                                original_image_path,
                                original_label_path,
                                aug_image_path,
                                metadata_path
                            )
                            
                            # Save transferred labels
                            aug_base_name = os.path.splitext(os.path.basename(aug_image_path))[0]
                            aug_label_path = os.path.join(aug_labels_dir, f"{aug_base_name}.txt")
                            label_transfer.save_transferred_labels(transferred_labels, aug_label_path)
                            
                            print(f"Transferred {len(transferred_labels)} labels to: {aug_label_path}")
                            
                            # Also get model predictions for comparison
                            model_predictions = predictor.predict_image(aug_image_path)
                            if model_predictions['type'] in ['detection', 'instance_segmentation']:
                                original_count = len(model_predictions['boxes'])
                                mask = model_predictions['scores'] >= confidence_threshold
                                model_predictions['boxes'] = model_predictions['boxes'][mask]
                                model_predictions['scores'] = model_predictions['scores'][mask]
                                model_predictions['class_ids'] = model_predictions['class_ids'][mask]
                                
                                # Convert model predictions to label format for comparison
                                model_labels = []
                                for box, class_id, score in zip(model_predictions['boxes'], 
                                                              model_predictions['class_ids'], 
                                                              model_predictions['scores']):
                                    x1, y1, x2, y2 = box
                                    img_height, img_width = cv2.imread(aug_image_path).shape[:2]
                                    x_center = (x1 + x2) / 2 / img_width
                                    y_center = (y1 + y2) / 2 / img_height
                                    width = (x2 - x1) / img_width
                                    height = (y2 - y1) / img_height
                                    model_labels.append([class_id, x_center, y_center, width, height])
                                
                                # Compare transferred vs model predictions
                                comparison = label_transfer.compare_labels(transferred_labels, model_labels)
                                print(f"Label consistency: {comparison['details']}")
                                
                                # If model predictions are significantly better, use them instead
                                if comparison['score'] < 0.3 and len(model_labels) > 0:
                                    print(f"Using model predictions instead of transferred labels (better consistency)")
                                    label_transfer.save_transferred_labels(model_labels, aug_label_path)
                    except Exception as e:
                        print(f"Error processing augmented image {aug_image_path}: {e}")
                        traceback.print_exc()
                        continue
            
            # Step 4: Calculate final statistics and create processed files list
            print(f"\n=== STEP 4: CALCULATING FINAL STATISTICS ===")
            
            processed_images = 0
            total_objects = 0
            total_confidence = 0.0
            detection_stats = {
                'high_confidence': 0,  # > 0.8
                'medium_confidence': 0,  # 0.5-0.8
                'low_confidence': 0,  # < 0.5
                'no_detections': 0
            }
            
            # This list will store paths of successfully processed files
            processed_files_list = []
            # This set will store all unique detected class names
            all_class_names = set()
            
            # Count original images
            for image_file in image_files:
                label_path = original_labels_map.get(image_file)
                if label_path and os.path.exists(label_path):
                    processed_images += 1
                    
                    # Count objects in label file
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            total_objects += len(lines)
                    except:
                        pass
                    
                    # Add to processed files list
                    processed_files_list.append({
                        'image_path': image_file,
                        'label_path': label_path,
                        'type': 'original'
                    })
            
            # Count augmented images
            if do_augmentation:
                for aug_image_path in all_augmented_images:
                    aug_base_name = os.path.splitext(os.path.basename(aug_image_path))[0]
                    aug_label_path = os.path.join(aug_labels_dir, f"{aug_base_name}.txt")
                    
                    if os.path.exists(aug_label_path):
                        processed_images += 1
                        
                        # Count objects in label file
                        try:
                            with open(aug_label_path, 'r') as f:
                                lines = f.readlines()
                                total_objects += len(lines)
                        except:
                            pass
                        
                        # Add to processed files list
                        processed_files_list.append({
                            'image_path': aug_image_path,
                            'label_path': aug_label_path,
                            'type': 'augmented'
                        })
            
            # Calculate average confidence (approximate)
            avg_confidence = 0.7  # Default confidence for transferred labels
            
            # Create ZIP files
            try:
                if do_augmentation:
                    aug_images_zip = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_images.zip')
                    aug_labels_zip = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels.zip')
                    
                    # Only create ZIP if directory exists and has files
                    if os.path.exists(aug_images_dir) and any(os.listdir(aug_images_dir)):
                        create_zip_archive(aug_images_dir, aug_images_zip)
                    if os.path.exists(aug_labels_dir) and any(os.listdir(aug_labels_dir)):
                        create_zip_archive(aug_labels_dir, aug_labels_zip)
                
                # Create original labels ZIP
                original_labels_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels')
                if os.path.exists(original_labels_dir) and any(os.listdir(original_labels_dir)):
                    original_labels_zip = os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels.zip')
                    create_zip_archive(original_labels_dir, original_labels_zip)
                    
            except Exception as zip_error:
                print(f"Warning: Error creating ZIP files: {zip_error}")
                traceback.print_exc()
                # Continue processing even if ZIP creation fails
            
<<<<<<< HEAD
            # Calculate results
            avg_confidence = total_confidence / processed_images if processed_images > 0 else 0.0
            
            # After loop, store results and file paths in session (reduced data to prevent cookie overflow)
=======
            # After loop, store results and file paths in session
>>>>>>> 209e6a65475c0d3ae1acacd96fdece629a3cb288
            session['results'] = {
                'total_images': int(len(image_files) + (len(all_augmented_images) if do_augmentation else 0)),
                'processed_images': int(processed_images),
                'total_objects': int(total_objects),
                'avg_confidence': float(avg_confidence),
                'total_augmented': int(total_augmented) if do_augmentation else 0,
                'confidence_threshold': float(confidence_threshold),
                'detection_stats': detection_stats,
                'performance_results': performance_results if 'performance_results' in locals() else {},
                # Store only file counts instead of full paths to reduce session size
                'original_files_count': len([f for f in processed_files_list if 'original' in f['label_path']]),
                'augmented_files_count': len([f for f in processed_files_list if 'augmented' in f['label_path']])
            }
            # Store unique class names in session
            session['class_names'] = sorted(list(all_class_names))
            
            # Store processed files list in a temporary file instead of session
            temp_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f'processed_files_{session_id}.json')
            with open(temp_file_path, 'w') as f:
                json.dump(processed_files_list, f)
            session['processed_files_temp'] = temp_file_path
            
            print(f"\n=== PROCESSING COMPLETE ===")
            print(f"Results: {session['results']}")
            
            # Store results in session and redirect
            session['message'] = {'type': 'success', 'text': f'Successfully processed {processed_images} images with {total_objects} objects detected.'}
            session.pop('processing', None)  # Clear processing flag
            
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Error in form processing: {e}")
            traceback.print_exc()
            session.pop('processing', None)  # Clear processing flag
            session['message'] = {'type': 'danger', 'text': f'Error processing request: {str(e)}'}
            return redirect(url_for('index'))

@app.route('/clear_session', methods=['POST'])
def clear_session():
    """Clear all session data and temporary files"""
    # Clean up temporary files
    if 'processed_files_temp' in session:
        temp_file = session['processed_files_temp']
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass  # Ignore errors if file can't be deleted
    
    session.clear()
    return redirect(url_for('index'))

@app.route('/validate_labels', methods=['POST'])
def validate_labels():
    """Validate the generated labels and provide quality metrics"""
    try:
        validation_data = []
        
        # Check both original and augmented label directories
        label_dirs = [
            os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels'),
            os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels')
        ]
        
        found_files = False
        
        for label_dir in label_dirs:
            if not os.path.exists(label_dir):
                continue
                
            # Analyze each prediction file in this directory
            for txt_file in Path(label_dir).glob('*.txt'):
                try:
                    found_files = True
                    image_name = txt_file.stem
                    
                    # Read the label file
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                    
                    object_count = len(lines)
                    confidences = []
                    
                    # Parse each detection
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # YOLO format: class x_center y_center width height [confidence]
                            if len(parts) > 5:
                                confidences.append(float(parts[5]))
                            else:
                                confidences.append(1.0)  # Default confidence if not provided
                    
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
                    
                    # Determine status based on confidence and object count
                    if avg_confidence > 0.8:
                        status = "Excellent"
                    elif avg_confidence > 0.6:
                        status = "Good"
                    elif avg_confidence > 0.4:
                        status = "Fair"
                    else:
                        status = "Poor"
                    
                    details = f"Detected {object_count} objects with average confidence {avg_confidence:.2f}"
                    
                    validation_data.append({
                        'image': image_name,
                        'object_count': object_count,
                        'confidence': avg_confidence,
                        'status': status,
                        'details': details,
                        'type': 'augmented' if 'augmented' in label_dir else 'original'
                    })
                    
                except Exception as e:
                    validation_data.append({
                        'image': txt_file.stem,
                        'object_count': 0,
                        'confidence': 0.0,
                        'status': 'Error',
                        'details': f'Error reading file: {str(e)}',
                        'type': 'augmented' if 'augmented' in label_dir else 'original'
                    })
        
        if not found_files:
            return jsonify({'error': 'No label files found. Please process images first.'}), 400
        
        return jsonify({
            'message': 'Label validation completed',
            'validation_data': validation_data
        })
        
    except Exception as e:
        return jsonify({'error': f'Validation error: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download a file from the output folder"""
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': f'File {filename} not found. Please process images first.'}), 404
        
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            return jsonify({'error': f'File {filename} is empty.'}), 400
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/download_individual/<file_type>/<filename>')
def download_individual_file(file_type, filename):
    """Download individual files from specific directories"""
    try:
        if file_type == 'original_labels':
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels', filename)
        elif file_type == 'augmented_labels':
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels', filename)
        elif file_type == 'augmented_images':
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_images', filename)
        elif file_type == 'original_images':
            file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original_images', filename)
        else:
            return jsonify({'error': f'Invalid file type: {file_type}'}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': f'File {filename} not found in {file_type}.'}), 404
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': f'Download error: {str(e)}'}), 500

@app.route('/list_files')
def list_available_files():
    """List all available files for download"""
    try:
        files = {
            'zip_files': [],
            'original_labels': [],
            'augmented_labels': [],
            'augmented_images': [],
            'original_images': []
        }
        
        # Check ZIP files
        output_dir = app.config['OUTPUT_FOLDER']
        for file in os.listdir(output_dir):
            if file.endswith('.zip'):
                file_path = os.path.join(output_dir, file)
                if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                    files['zip_files'].append({
                        'name': file,
                        'size': os.path.getsize(file_path),
                        'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    })
        
        # Check individual files
        dirs_to_check = [
            ('original_labels', 'original_labels'),
            ('augmented_labels', 'augmented_labels'),
            ('augmented_images', 'augmented_images'),
            ('original_images', 'original_images')
        ]
        
        for dir_name, key in dirs_to_check:
            dir_path = os.path.join(output_dir, dir_name)
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                        files[key].append({
                            'name': file,
                            'size': os.path.getsize(file_path),
                            'size_mb': round(os.path.getsize(file_path) / (1024 * 1024), 2)
                        })
        
        return jsonify(files)
        
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@app.route('/statistics')
def get_statistics():
    """Get detailed statistics about processed data"""
    try:
        stats = {
            'file_counts': {},
            'total_sizes': {},
            'processing_summary': session.get('results', {}),
            'available_files': {}
        }
        
        output_dir = app.config['OUTPUT_FOLDER']
        
        # Count files in each directory
        dirs_to_check = [
            'original_labels',
            'augmented_labels', 
            'augmented_images',
            'original_images'
        ]
        
        for dir_name in dirs_to_check:
            dir_path = os.path.join(output_dir, dir_name)
            if os.path.exists(dir_path):
                files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
                total_size = sum(os.path.getsize(os.path.join(dir_path, f)) for f in files)
                
                stats['file_counts'][dir_name] = len(files)
                stats['total_sizes'][dir_name] = {
                    'bytes': total_size,
                    'mb': round(total_size / (1024 * 1024), 2)
                }
                stats['available_files'][dir_name] = files[:10]  # First 10 files
        
        # Count ZIP files
        zip_files = [f for f in os.listdir(output_dir) if f.endswith('.zip')]
        stats['file_counts']['zip_files'] = len(zip_files)
        stats['available_files']['zip_files'] = zip_files
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': f'Error getting statistics: {str(e)}'}), 500

@app.route('/download_all')
def download_all_files():
    """Create and download a comprehensive ZIP with all results"""
    try:
        import tempfile
        import zipfile
        
        # Create temporary ZIP file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            output_dir = app.config['OUTPUT_FOLDER']
            
            # Add all directories
            dirs_to_add = [
                'original_labels',
                'augmented_labels',
                'augmented_images', 
                'original_images'
            ]
            
            for dir_name in dirs_to_add:
                dir_path = os.path.join(output_dir, dir_name)
                if os.path.exists(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, output_dir)
                            zipf.write(file_path, arcname)
            
            # Add ZIP files
            for file in os.listdir(output_dir):
                if file.endswith('.zip'):
                    file_path = os.path.join(output_dir, file)
                    zipf.write(file_path, file)
        
        # Send the ZIP file
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name='all_results.zip',
            mimetype='application/zip'
        )
        
    except Exception as e:
        return jsonify({'error': f'Error creating comprehensive ZIP: {str(e)}'}), 500

@app.route('/preview_file/<file_type>/<filename>')
def preview_file(file_type, filename):
    """Preview file contents (for text files)"""
    try:
        if file_type in ['original_labels', 'augmented_labels']:
            if file_type == 'original_labels':
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'original_labels', filename)
            else:
                file_path = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels', filename)
            
            if not os.path.exists(file_path):
                return jsonify({'error': 'File not found'}), 404
            
            # Read first few lines for preview
            with open(file_path, 'r') as f:
                lines = f.readlines()[:10]  # First 10 lines
            
            return jsonify({
                'filename': filename,
                'preview': lines,
                'total_lines': len(lines),
                'file_size': os.path.getsize(file_path)
            })
        
        else:
            return jsonify({'error': 'Preview not available for this file type'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Error previewing file: {str(e)}'}), 500

@app.route('/create_yolo_dataset', methods=['POST'])
def create_and_download_yolo_dataset():
    """Create and download YOLO format dataset"""
    try:
        data = request.json
        image_paths = data['image_paths']
        label_paths = data['label_paths']
        class_names = data['class_names']
        split_ratios = {
            'train': float(data['train']) / 100,
            'valid': float(data['valid']) / 100,
            'test': float(data['test']) / 100
        }

        zip_path = create_yolo_dataset(image_paths, label_paths, class_names, split_ratios)
        
        return send_file(zip_path, as_attachment=True, download_name='yolo_dataset.zip')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error creating YOLO dataset: {str(e)}'}), 500

def create_yolo_dataset(image_paths, label_paths, class_names, split_ratios, zip_basename='yolo_dataset_final'):
    """Create YOLO format dataset structure with specified splits."""
    base_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'yolo_dataset_split')
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # Create YOLO directory structure
    sub_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
    for sub_dir in sub_dirs:
        os.makedirs(os.path.join(base_dir, sub_dir), exist_ok=True)

    # Shuffle image and label paths together
    combined = list(zip(image_paths, label_paths))
    random.shuffle(combined)
    
    if not combined: # Handle case with no files
        image_paths, label_paths = [], []
    else:
        image_paths[:], label_paths[:] = zip(*combined)

    # Calculate split indices
    total_files = len(image_paths)
    train_end = int(total_files * split_ratios['train'])
    valid_end = train_end + int(total_files * split_ratios['valid'])

    # Define splits
    splits = {
        'train': (image_paths[:train_end], label_paths[:train_end]),
        'valid': (image_paths[train_end:valid_end], label_paths[train_end:valid_end]),
        'test': (image_paths[valid_end:], label_paths[valid_end:])
    }

    # Copy files to new directories
    for split_name, (img_files, lbl_files) in splits.items():
        # Ensure source paths exist before copying
        valid_img_files = [p for p in img_files if os.path.exists(p)]
        valid_lbl_files = [p for p in lbl_files if os.path.exists(p)]
        
        for img_path in valid_img_files:
            shutil.copy(img_path, os.path.join(base_dir, f'{split_name}/images'))
        for lbl_path in valid_lbl_files:
            shutil.copy(lbl_path, os.path.join(base_dir, f'{split_name}/labels'))

    # Create dataset.yaml
    yaml_content = {
        'path': '../',
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names
    }
    with open(os.path.join(base_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

    # Create ZIP file
    zip_output_path = os.path.join(app.config['OUTPUT_FOLDER'], zip_basename)
    shutil.make_archive(zip_output_path, 'zip', base_dir)
    
    return f"{zip_output_path}.zip"

@app.route('/create_split_dataset', methods=['POST'])
def create_split_dataset_route():
    """Create and download a YOLO dataset with specified splits."""
    try:
        data = request.json
        split_ratios = {
            'train': float(data.get('train', 70)) / 100,
            'valid': float(data.get('valid', 10)) / 100,
            'test': float(data.get('test', 20)) / 100
        }

        if 'results' not in session or 'processed_files_temp' not in session:
            return jsonify({'error': 'No processed data found in session. Please process images first.'}), 404

        # Load processed files from temporary file
        temp_file_path = session['processed_files_temp']
        if not os.path.exists(temp_file_path):
            return jsonify({'error': 'Processed files data not found. Please process images first.'}), 404
            
        with open(temp_file_path, 'r') as f:
            processed_files = json.load(f)
        
        image_paths = [item['image_path'] for item in processed_files]
        label_paths = [item['label_path'] for item in processed_files]
        
        class_names = session.get('class_names', ['object'])

        zip_basename = 'yolo_dataset_all_files'
        zip_path = create_yolo_dataset(image_paths, label_paths, class_names, split_ratios, zip_basename)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{zip_basename}.zip')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error creating split dataset: {str(e)}'}), 500

@app.route('/create_yolo_from_augmented', methods=['POST'])
def create_yolo_from_augmented_route():
    """Create and download a YOLO dataset exclusively from augmented data."""
    try:
        data = request.json
        split_ratios = {
            'train': float(data.get('train', 70)) / 100,
            'valid': float(data.get('valid', 20)) / 100,
            'test': float(data.get('test', 10)) / 100
        }

        aug_images_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_images')
        aug_labels_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'augmented_labels')

        if not os.path.exists(aug_images_dir) or not os.path.exists(aug_labels_dir):
            return jsonify({'error': 'Augmented data folders not found. Please process images with augmentation first.'}), 404

        # Find all images and their corresponding labels
        image_paths = []
        label_paths = []
        
        all_aug_images = [os.path.join(aug_images_dir, f) for f in os.listdir(aug_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_path in all_aug_images:
            # Find corresponding label file
            base_name = Path(img_path).stem
            label_file = f"{base_name}.txt"
            label_path = os.path.join(aug_labels_dir, label_file)
            
            if os.path.exists(label_path):
                image_paths.append(img_path)
                label_paths.append(label_path)

        if not image_paths:
            return jsonify({'error': 'No matching image and label pairs found in augmented directories.'}), 404
            
        class_names = session.get('class_names', ['object'])

        zip_basename = 'yolo_dataset_augmented_files'
        zip_path = create_yolo_dataset(image_paths, label_paths, class_names, split_ratios, zip_basename)
        
        return send_file(zip_path, as_attachment=True, download_name=f'{zip_basename}.zip')

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Error creating split dataset from augmented data: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    
    # Disable debug mode to prevent auto-restart
    app.run(debug=False, host='127.0.0.1', port=5000) 