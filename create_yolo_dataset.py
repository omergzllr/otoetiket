import os
import shutil
import zipfile
from pathlib import Path

def create_yolo_structure():
    # Define the directory structure
    directories = [
        'test/images',
        'test/labels',
        'train/images',
        'train/labels',
        'valid/images',
        'valid/labels'
    ]
    
    # Create base directory
    base_dir = 'yolo_dataset'
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)
    
    # Create all directories
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path)
    
    # Create and write dataset.yaml
    yaml_content = """path: .
train: train/images
val: valid/images
test: test/images

# Classes
nc: 1  # number of classes
names: ['etiket']  # class names"""
    
    with open(os.path.join(base_dir, 'dataset.yaml'), 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    # Create ZIP file
    shutil.make_archive('yolo_dataset', 'zip', base_dir)
    
    print("YOLO dataset structure created successfully!")
    print("Files created:")
    print(f"- {base_dir}/ (directory with YOLO structure)")
    print("- yolo_dataset.zip (compressed file for download)")

if __name__ == '__main__':
    create_yolo_structure() 