#!/usr/bin/env python3
"""
Check ZIP file contents
"""

import zipfile
import os

def check_zip_contents():
    zip_path = "outputs/results.zip"
    
    if not os.path.exists(zip_path):
        print(f"ZIP file not found: {zip_path}")
        return
    
    print(f"Checking ZIP file: {zip_path}")
    print(f"File size: {os.path.getsize(zip_path)} bytes")
    
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        # List all files
        all_files = zipf.namelist()
        print(f"\nTotal files in ZIP: {len(all_files)}")
        
        # Count augmented files
        augmented_files = [f for f in all_files if 'aug' in f]
        print(f"Augmented files: {len(augmented_files)}")
        
        # Show first 10 augmented files
        print("\nFirst 10 augmented files:")
        for i, file in enumerate(augmented_files[:10]):
            info = zipf.getinfo(file)
            print(f"  {i+1}. {file} ({info.file_size} bytes)")
        
        # Count by type
        aug_images = [f for f in augmented_files if f.endswith('.jpg')]
        aug_labels = [f for f in augmented_files if f.endswith('.txt')]
        
        print(f"\nAugmented images: {len(aug_images)}")
        print(f"Augmented labels: {len(aug_labels)}")
        
        # Show some label contents
        print("\nSample augmented label contents:")
        for i, label_file in enumerate(aug_labels[:3]):
            with zipf.open(label_file) as f:
                content = f.read().decode('utf-8').strip()
                print(f"  {label_file}: {content}")

if __name__ == "__main__":
    check_zip_contents() 