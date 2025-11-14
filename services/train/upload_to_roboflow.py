#!/usr/bin/env python3
"""
Upload images to Roboflow project for labeling

Usage:
    export ROBOFLOW_IMAGE_PATH="/path/to/images"
    export ROBOFLOW_PUBLIC_API_KEY="your_public_key"
    export ROBOFLOW_PRIVATE_API_KEY="your_private_key"
    python upload_to_roboflow.py
"""

import os
import sys
from pathlib import Path
from roboflow import Roboflow
from datetime import datetime

def upload_images_to_roboflow():
    """Upload images from ROBOFLOW_IMAGE_PATH to Roboflow project."""

    # Get environment variables
    image_path = os.getenv("ROBOFLOW_IMAGE_PATH")
    public_api_key = os.getenv("ROBOFLOW_PUBLIC_API_KEY")
    private_api_key = os.getenv("ROBOFLOW_PRIVATE_API_KEY")

    # Validate environment variables
    if not image_path:
        print("❌ Error: ROBOFLOW_IMAGE_PATH environment variable not set")
        print("   Usage: export ROBOFLOW_IMAGE_PATH='/path/to/images'")
        sys.exit(1)

    if not public_api_key:
        print("❌ Error: ROBOFLOW_PUBLIC_API_KEY environment variable not set")
        print("   Get your API key from: https://app.roboflow.com/settings/api")
        sys.exit(1)

    # Roboflow typically uses just one API key - try public first
    api_key = public_api_key or private_api_key

    # Validate image path exists
    image_dir = Path(image_path)
    if not image_dir.exists():
        print(f"❌ Error: Image path does not exist: {image_path}")
        sys.exit(1)

    if not image_dir.is_dir():
        print(f"❌ Error: Path is not a directory: {image_path}")
        sys.exit(1)

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = [
        f for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ]

    if not image_files:
        print(f"❌ Error: No image files found in {image_path}")
        print(f"   Looking for extensions: {', '.join(image_extensions)}")
        sys.exit(1)

    print(f"📁 Found {len(image_files)} images in {image_path}")
    print(f"🔑 Using API key: {api_key[:10]}...")

    # Initialize Roboflow
    try:
        rf = Roboflow(api_key=api_key)
        print("✓ Connected to Roboflow")
    except Exception as e:
        print(f"❌ Failed to connect to Roboflow: {e}")
        sys.exit(1)

    # Get workspace and project
    # You'll need to update these after creating your project
    workspace_name = os.getenv("ROBOFLOW_WORKSPACE", None)
    project_name = os.getenv("ROBOFLOW_PROJECT", "street-car-wheels")

    try:
        if workspace_name:
            workspace = rf.workspace(workspace_name)
        else:
            workspace = rf.workspace()

        print(f"✓ Workspace: {workspace.name}")

        project = workspace.project(project_name)
        print(f"✓ Project: {project.name}")

    except Exception as e:
        print(f"❌ Failed to access project: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Make sure you created a project named '{project_name}'")
        print(f"  2. Or set ROBOFLOW_PROJECT env var to your project name")
        print(f"  3. Check project settings → API to see correct project ID")
        sys.exit(1)

    # Create batch name with timestamp
    batch_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🚀 Starting upload...")
    print(f"   Batch: {batch_name}")
    print(f"   Images: {len(image_files)}")

    # Upload images
    uploaded_count = 0
    failed_count = 0

    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"   [{i}/{len(image_files)}] Uploading {image_file.name}...", end='\r')

            project.upload(
                image_path=str(image_file),
                batch_name=batch_name,
                num_retry_uploads=3
            )

            uploaded_count += 1

        except Exception as e:
            print(f"\n   ⚠️  Failed to upload {image_file.name}: {e}")
            failed_count += 1
            continue

    # Summary
    print(f"\n\n✅ Upload complete!")
    print(f"   Successfully uploaded: {uploaded_count}/{len(image_files)}")

    if failed_count > 0:
        print(f"   Failed: {failed_count}")

    print(f"\n📋 Next steps:")
    print(f"   1. Go to https://app.roboflow.com")
    print(f"   2. Open your '{project_name}' project")
    print(f"   3. Click 'Annotate' to start labeling")
    print(f"   4. Look for batch: {batch_name}")

    return uploaded_count

if __name__ == "__main__":
    print("=" * 60)
    print("Roboflow Image Upload Script")
    print("=" * 60)
    print()

    try:
        count = upload_images_to_roboflow()
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n\n⚠️  Upload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
