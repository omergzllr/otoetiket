#!/usr/bin/env python3
"""
Label Accuracy Checker
Displays images with their YOLO labels overlaid for visual verification
"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog, messagebox

class LabelChecker:
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.current_index = 0
        self.image_files = []
        self.label_files = []
        
        # Get all image files
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list(self.images_dir.glob(ext)))
        
        self.image_files.sort()
        print(f"Found {len(self.image_files)} images")
        
    def load_yolo_label(self, label_path):
        """Load YOLO format label file"""
        if not os.path.exists(label_path):
            return []
        
        boxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    confidence = float(parts[5]) if len(parts) > 5 else 1.0
                    
                    boxes.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height,
                        'confidence': confidence
                    })
        return boxes
    
    def yolo_to_pixel(self, yolo_box, img_width, img_height):
        """Convert YOLO format to pixel coordinates"""
        x_center = yolo_box['x_center'] * img_width
        y_center = yolo_box['y_center'] * img_height
        width = yolo_box['width'] * img_width
        height = yolo_box['height'] * img_height
        
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        return int(x1), int(y1), int(x2), int(y2)
    
    def display_image_with_labels(self, image_path, label_path):
        """Display image with bounding boxes"""
        # Load image
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load labels
        boxes = self.load_yolo_label(label_path)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = self.yolo_to_pixel(box, width, height)
            
            # Create rectangle
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2,
                edgecolor=colors[box['class_id'] % len(colors)],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label_text = f"Class {box['class_id']} ({box['confidence']:.2f})"
            ax.text(x1, y1-5, label_text, 
                   color=colors[box['class_id'] % len(colors)],
                   fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
        
        ax.set_title(f"Image: {image_path.name}\nLabels: {len(boxes)} objects")
        ax.axis('off')
        
        return fig
    
    def interactive_viewer(self):
        """Interactive viewer with navigation buttons"""
        if not self.image_files:
            print("No images found!")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        plt.subplots_adjust(bottom=0.15)
        
        def update_display():
            ax.clear()
            
            image_path = self.image_files[self.current_index]
            label_path = self.labels_dir / f"{image_path.stem}.txt"
            
            # Load and display image
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            ax.imshow(image)
            
            # Load and draw labels
            if os.path.exists(label_path):
                boxes = self.load_yolo_label(label_path)
                colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'cyan']
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = self.yolo_to_pixel(box, width, height)
                    
                    rect = patches.Rectangle(
                        (x1, y1), x2-x1, y2-y1,
                        linewidth=2,
                        edgecolor=colors[box['class_id'] % len(colors)],
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    label_text = f"Class {box['class_id']} ({box['confidence']:.2f})"
                    ax.text(x1, y1-5, label_text, 
                           color=colors[box['class_id'] % len(colors)],
                           fontsize=10, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
                
                ax.set_title(f"Image {self.current_index + 1}/{len(self.image_files)}: {image_path.name}\n"
                           f"Labels: {len(boxes)} objects")
            else:
                ax.set_title(f"Image {self.current_index + 1}/{len(self.image_files)}: {image_path.name}\n"
                           f"No label file found")
            
            ax.axis('off')
            plt.draw()
        
        def next_image(event):
            self.current_index = (self.current_index + 1) % len(self.image_files)
            update_display()
        
        def prev_image(event):
            self.current_index = (self.current_index - 1) % len(self.image_files)
            update_display()
        
        def mark_correct(event):
            print(f"Marked image {self.current_index + 1} as CORRECT")
            # You can add logic here to save the verification status
        
        def mark_incorrect(event):
            print(f"Marked image {self.current_index + 1} as INCORRECT")
            # You can add logic here to save the verification status
        
        # Add buttons
        ax_prev = plt.axes([0.1, 0.05, 0.1, 0.04])
        ax_next = plt.axes([0.25, 0.05, 0.1, 0.04])
        ax_correct = plt.axes([0.4, 0.05, 0.1, 0.04])
        ax_incorrect = plt.axes([0.55, 0.05, 0.1, 0.04])
        
        btn_prev = Button(ax_prev, 'Previous')
        btn_next = Button(ax_next, 'Next')
        btn_correct = Button(ax_correct, 'Correct')
        btn_incorrect = Button(ax_incorrect, 'Incorrect')
        
        btn_prev.on_clicked(prev_image)
        btn_next.on_clicked(next_image)
        btn_correct.on_clicked(mark_correct)
        btn_incorrect.on_clicked(mark_incorrect)
        
        # Initial display
        update_display()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Check YOLO label accuracy')
    parser.add_argument('--images', type=str, default='outputs/predictions',
                       help='Directory containing images')
    parser.add_argument('--labels', type=str, default='outputs/predictions',
                       help='Directory containing label files')
    parser.add_argument('--interactive', action='store_true',
                       help='Use interactive viewer')
    
    args = parser.parse_args()
    
    checker = LabelChecker(args.images, args.labels)
    
    if args.interactive:
        checker.interactive_viewer()
    else:
        # Display first image
        if checker.image_files:
            image_path = checker.image_files[0]
            label_path = checker.labels_dir / f"{image_path.stem}.txt"
            fig = checker.display_image_with_labels(image_path, label_path)
            plt.show()

if __name__ == "__main__":
    main() 