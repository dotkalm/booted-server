# Training Scripts for Street-Level Wheel Detection

This directory contains scripts for training custom YOLOv8 wheel detection models using Roboflow for dataset management.

## Workflow Overview

```
1. Collect Images → 2. Upload to Roboflow → 3. Label → 4. Download → 5. Train
```

## Scripts

### 1. `upload_to_roboflow.py`

Upload street-level car images to Roboflow for labeling.

**Usage:**
```bash
export ROBOFLOW_IMAGE_PATH="/path/to/street/car/images"
export ROBOFLOW_PRIVATE_API_KEY="your_api_key"
export ROBOFLOW_WORKSPACE="joelml"
export ROBOFLOW_PROJECT="street-car-wheels"

python services/train/upload_to_roboflow.py
```

**What it does:**
- Validates image directory and API credentials
- Uploads all images (jpg, png, etc.) to Roboflow
- Creates timestamped batches for organization
- Shows progress and summary

### 2. Label in Roboflow (Web Interface)

After uploading:

1. Go to https://app.roboflow.com
2. Open your project
3. Click **"Annotate"**
4. Use **Box tool (B key)** to label wheels
5. Label all visible wheels in each image
6. When done, click **"Generate"**:
   - Apply preprocessing (640×640 resize)
   - Apply augmentations (flip, rotate, brightness, etc.)
   - Split: 70% train / 20% valid / 10% test
   - Export format: **YOLOv8**

### 3. `download_from_roboflow.py`

Download the labeled and augmented dataset.

**Usage:**
```bash
export ROBOFLOW_PRIVATE_API_KEY="your_api_key"
export ROBOFLOW_WORKSPACE="joelml"
export ROBOFLOW_PROJECT="street-car-wheels"
export ROBOFLOW_VERSION="1"  # Optional, defaults to 1

python services/train/download_from_roboflow.py
```

**What it does:**
- Downloads annotated dataset from Roboflow
- Saves to `models/street-car-wheels-1/`
- Shows dataset configuration and stats

### 4. `train_street_wheels.py`

Train YOLOv8 model on the downloaded dataset.

**Usage:**
```bash
# Update dataset path in script if needed, then:
python services/train/train_street_wheels.py
```

**What it does:**
- Loads YOLOv8n pretrained weights
- Trains on your street-level wheel dataset
- Applies street-specific augmentations
- Saves results to `results/street_wheel_detection_v1/`
- Generates training plots and metrics

**Training parameters:**
- 100 epochs with early stopping (patience=20)
- AdamW optimizer
- Street-specific augmentations (scale, perspective, etc.)
- GPU training (set `device='cpu'` if no GPU)

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ROBOFLOW_PRIVATE_API_KEY` | ✓ | - | Your Roboflow private API key (preferred) |
| `ROBOFLOW_PUBLIC_API_KEY` | - | - | Fallback if private key not set |
| `ROBOFLOW_IMAGE_PATH` | ✓ (upload) | - | Path to images for upload |
| `ROBOFLOW_PROJECT` | ✓ | `street-car-wheels` | Project ID from URL |
| `ROBOFLOW_WORKSPACE` | ✓ | - | Workspace name from URL |
| `ROBOFLOW_VERSION` | - | `1` | Dataset version to download |

## Full Workflow Example

```bash
# Step 1: Upload images
export ROBOFLOW_IMAGE_PATH="./street_images"
export ROBOFLOW_PRIVATE_API_KEY="abc123..."  # Use private key for API operations
export ROBOFLOW_WORKSPACE="joelml"
export ROBOFLOW_PROJECT="street-car-wheels"

python services/train/upload_to_roboflow.py

# Step 2: Label in Roboflow web interface
# - Go to https://app.roboflow.com
# - Label wheels in all images
# - Generate dataset version with augmentations

# Step 3: Download labeled dataset
python services/train/download_from_roboflow.py

# Step 4: Train model
python services/train/train_street_wheels.py

# Step 5: Update detector to use new model
# Edit services/models/detector.py:
#   wheel_model_path = base_path / "results/street_wheel_detection_v1/weights/best.pt"

# Step 6: Test
python -m uvicorn main:app --reload
```

## Tips

### Dataset Quality
- **500-1000 images** of street-level cars
- **Various angles**: side, 3/4, front
- **Different car types**: sedan, SUV, truck, van
- **Lighting variety**: day, night, shadows
- **Distance**: 10-50 meters from camera

### Labeling Strategy
- Label **all visible wheels** (2-4 per car typically)
- Include **partially occluded** wheels (50%+ visible)
- Skip wheels that are <20% visible
- Be consistent with wheel boundaries

### Sources for Images
- **Pexels/Unsplash**: Free stock photos
- **COCO dataset**: Filter for car images
- **Dashcam footage**: Extract frames with ffmpeg
- **Your own photos**: Drive around and capture

### Roboflow Tips
- Use **Label Assist** after ~50 manual labels
- Check **Health Check** before generating
- Use **3× augmentation** for more training data
- Save multiple versions to compare

## Troubleshooting

### Upload fails
- Check API key is correct
- Verify project name matches
- Ensure image path exists

### Download fails
- Make sure dataset version is generated
- Check you selected YOLOv8 export format
- Verify API key permissions

### Training fails
- Check dataset path in `train_street_wheels.py`
- Verify `data.yaml` exists
- Reduce batch size if GPU memory error
- Set `device='cpu'` if no GPU available

## Output Structure

```
models/
  street-car-wheels-1/
    ├── data.yaml          # Dataset configuration
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/

results/
  street_wheel_detection_v1/
    ├── weights/
    │   ├── best.pt        # Best model weights
    │   └── last.pt        # Last epoch weights
    ├── results.csv        # Training metrics
    ├── results.png        # Training plots
    └── ...
```
