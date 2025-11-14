#!/usr/bin/env python3
"""
Download labeled dataset from Roboflow after annotation is complete

Usage:
    export ROBOFLOW_PUBLIC_API_KEY="your_api_key"
    export ROBOFLOW_PROJECT="street-car-wheels"  # Optional
    export ROBOFLOW_VERSION="1"  # Optional, defaults to 1
    python services/train/download_from_roboflow.py
"""

import os
import sys
from pathlib import Path
from roboflow import Roboflow

def download_dataset_from_roboflow():
    """Download annotated dataset from Roboflow project."""

    # Get project root (two levels up from services/train/)
    project_root = Path(__file__).parent.parent.parent

    # Get environment variables
    private_api_key = os.getenv("ROBOFLOW_PRIVATE_API_KEY")
    public_api_key = os.getenv("ROBOFLOW_PUBLIC_API_KEY")

    # Try private key first (usually needed for API operations)
    api_key = private_api_key or public_api_key

    workspace_name = os.getenv("ROBOFLOW_WORKSPACE", None)
    project_name = os.getenv("ROBOFLOW_PROJECT", "street-car-wheels")
    version = int(os.getenv("ROBOFLOW_VERSION", "1"))

    # Validate API key
    if not api_key:
        print("ERROR: No Roboflow API key found")
        print("   Set ROBOFLOW_PRIVATE_API_KEY or ROBOFLOW_PUBLIC_API_KEY")
        print("   Get your API key from: https://app.roboflow.com/settings/api")
        sys.exit(1)

    print(f"Using API key: {api_key[:10]}...")
    print(f"Project: {project_name}")
    print(f"Version: {version}")

    # Initialize Roboflow
    try:
        rf = Roboflow(api_key=api_key)
        print("Connected to Roboflow")
    except Exception as e:
        print(f"ERROR: Failed to connect to Roboflow: {e}")
        sys.exit(1)

    # Get workspace and project
    try:
        if workspace_name:
            workspace = rf.workspace(workspace_name)
        else:
            workspace = rf.workspace()

        print(f"Workspace: {workspace.name}")

        project = workspace.project(project_name)
        print(f"Project: {project.name}")

    except Exception as e:
        print(f"ERROR: Failed to access project: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure you created a project named '{project_name}'")
        print(f"  2. Or set ROBOFLOW_PROJECT env var to your project name")
        print(f"  3. Check project settings -> API to see correct project ID")
        sys.exit(1)

    # Get dataset version
    try:
        dataset_version = project.version(version)
        print(f"Dataset version {version} found")
    except Exception as e:
        print(f"ERROR: Failed to access version {version}: {e}")
        print(f"\nMake sure you:")
        print(f"  1. Finished labeling images")
        print(f"  2. Generated a dataset version")
        print(f"  3. Applied preprocessing/augmentations")
        sys.exit(1)

    # Download to models directory
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    print(f"\nDownloading dataset to {models_dir}/")
    print(f"   Format: YOLOv8")

    try:
        dataset = dataset_version.download(
            model_format="yolov8",
            location=str(models_dir)
        )

        print(f"\nDownload complete!")
        print(f"Dataset saved to: {models_dir}/{project_name}-{version}/")

        # Show dataset info
        data_yaml = models_dir / f"{project_name}-{version}" / "data.yaml"
        if data_yaml.exists():
            print(f"\nDataset configuration: {data_yaml}")
            print(f"\nContents:")
            with open(data_yaml, 'r') as f:
                print(f.read())

        print(f"\nNext steps:")
        print(f"   1. Review the dataset in {models_dir}/{project_name}-{version}/")
        print(f"   2. Update train_street_wheels.py if needed:")
        print(f"      dataset_yaml = project_root / 'models/{project_name}-{version}/data.yaml'")
        print(f"   3. Run training:")
        print(f"      python services/train/train_street_wheels.py")

        return dataset

    except Exception as e:
        print(f"\nERROR: Download failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check your internet connection")
        print(f"  2. Verify API key permissions")
        print(f"  3. Make sure version {version} was generated with YOLOv8 format")
        sys.exit(1)

if __name__ == "__main__":
    print("=" * 60)
    print("Roboflow Dataset Download Script")
    print("=" * 60)
    print()

    try:
        download_dataset_from_roboflow()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\nWARNING: Download cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")
        sys.exit(1)
