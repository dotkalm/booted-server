#!/usr/bin/env python3
"""
Train YOLOv8 wheel detection model on street-level car images from Roboflow
"""

from ultralytics import YOLO
import os
from pathlib import Path

def train_street_wheel_detector():
    """Train wheel detector on street-level car dataset."""

    # Get project root (two levels up from services/train/)
    project_root = Path(__file__).parent.parent.parent

    # Path to Roboflow exported dataset
    # Update this path after downloading from Roboflow
    dataset_yaml = project_root / "models/street-car-wheels-1/data.yaml"

    if not dataset_yaml.exists():
        print(f"ERROR: Dataset not found at {dataset_yaml}")
        print("Download from Roboflow first:")
        print("   1. Go to your Roboflow project")
        print("   2. Generate version with augmentations")
        print("   3. Export as YOLOv8 format")
        print("   4. Download and place in models/ directory")
        return

    # Load pretrained YOLOv8n as starting point
    yolov8_path = project_root / 'yolov8n.pt'
    model = YOLO(str(yolov8_path))

    print("Starting training on street-level wheel dataset...")
    print(f"Dataset: {dataset_yaml}")

    # Train with street-specific augmentations
    results_dir = project_root / 'results'

    results = model.train(
        data=str(dataset_yaml),

        # Training parameters
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust based on GPU memory
        patience=20,  # Early stopping

        # Augmentation (Roboflow already augments, but YOLO adds more)
        hsv_h=0.015,       # Hue variation
        hsv_s=0.7,         # Saturation
        hsv_v=0.4,         # Brightness
        degrees=10,        # Rotation for angled cars
        translate=0.1,     # Position shift
        scale=0.5,         # Scale variation (near/far cars)
        perspective=0.0002,  # Perspective transform
        flipud=0.0,        # Don't flip vertically (wheels at bottom)
        fliplr=0.5,        # Horizontal flip OK
        mosaic=1.0,        # Mix multiple images
        mixup=0.1,         # Blend images

        # Optimizer
        optimizer='AdamW',
        lr0=0.001,         # Learning rate
        lrf=0.01,          # Final learning rate factor
        weight_decay=0.0005,

        # Validation & saving
        val=True,
        save=True,
        save_period=10,    # Save checkpoint every 10 epochs
        plots=True,        # Generate training plots

        # Output
        project=str(results_dir),
        name='street_wheel_detection_v1',
        exist_ok=False,

        # Performance
        device=0,  # GPU 0, or 'cpu' for CPU training
        workers=8,
        verbose=True
    )

    output_path = results_dir / 'street_wheel_detection_v1'
    print("\nTraining complete!")
    print(f"Results saved to: {output_path}/")
    print(f"Best weights: {output_path}/weights/best.pt")

    # Run validation
    print("\nRunning validation...")
    metrics = model.val()

    print(f"\nFinal Metrics:")
    print(f"   mAP@50: {metrics.box.map50:.3f}")
    print(f"   mAP@50-95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")

    return results

if __name__ == "__main__":
    train_street_wheel_detector()
