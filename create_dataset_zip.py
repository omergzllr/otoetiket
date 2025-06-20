import os
import zipfile

def create_dataset_zip():
    # Define the directories to include
    dirs = [
        'train/images', 'train/labels',
        'valid/images', 'valid/labels',
        'test/images', 'test/labels'
    ]
    
    # Create dataset.zip
    with zipfile.ZipFile('dataset.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add data.yaml
        zipf.write('data.yaml')
        
        # Add all directories (even if empty)
        for dir_path in dirs:
            # Create a ZipInfo object for the directory
            dir_info = zipfile.ZipInfo(dir_path + '/')
            dir_info.external_attr = 0o40775 << 16  # Unix directory permissions
            zipf.writestr(dir_info, '')

if __name__ == '__main__':
    create_dataset_zip()
    print("Dataset ZIP file created successfully!") 