#!/usr/bin/env python3
"""
Test Roboflow connection and verify environment variables

Usage:
    python services/train/test_roboflow_connection.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_roboflow_setup():
    """Verify Roboflow environment variables and connection."""

    # Load .env from project root
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"

    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded .env from {env_path}\n")
    else:
        print(f"WARNING: No .env file found at {env_path}\n")

    # Check environment variables
    print("Environment Variables:")
    print("-" * 60)

    required_vars = {
        'ROBOFLOW_PRIVATE_API_KEY': 'Required for API operations (or use public key)',
        'ROBOFLOW_PROJECT': 'Your project ID',
        'ROBOFLOW_WORKSPACE': 'Your workspace name',
    }

    optional_vars = {
        'ROBOFLOW_PUBLIC_API_KEY': 'Fallback if private key not set',
        'ROBOFLOW_IMAGE_PATH': 'Required for upload only',
        'ROBOFLOW_VERSION': 'Optional (defaults to 1)',
    }

    all_good = True
    has_api_key = False

    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask API keys
            if 'KEY' in var:
                display = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
                has_api_key = True
            else:
                display = value
            print(f"  [OK] {var}: {display}")
            print(f"       {description}")
        else:
            # API key is special - we need either private or public
            if 'KEY' in var:
                print(f"  [ - ] {var}: not set")
                print(f"        {description}")
            else:
                print(f"  [MISSING] {var}: NOT SET")
                print(f"            {description}")
                all_good = False

    print()
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            if 'KEY' in var:
                display = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
                has_api_key = True
            elif 'PATH' in var:
                display = value
                # Check if path exists
                if Path(value).exists():
                    display += " [exists]"
                else:
                    display += " [path doesn't exist]"
            else:
                display = value
            print(f"  [OK] {var}: {display}")
            print(f"       {description}")
        else:
            print(f"  [ - ] {var}: not set (optional)")
            print(f"        {description}")

    print("\n" + "=" * 60)

    if not has_api_key:
        print("ERROR: No API key found (need ROBOFLOW_PRIVATE_API_KEY or ROBOFLOW_PUBLIC_API_KEY)")
        return False

    if not all_good:
        print("ERROR: Missing required environment variables")
        return False

    # Test Roboflow connection
    print("\nTesting Roboflow Connection...")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: Roboflow package not installed")
        print("   Install with: pip install roboflow")
        return False

    try:
        # Try private key first (usually needed for API operations)
        api_key = os.getenv('ROBOFLOW_PRIVATE_API_KEY') or os.getenv('ROBOFLOW_PUBLIC_API_KEY')
        if not api_key:
            print("ERROR: No API key found")
            return False

        rf = Roboflow(api_key=api_key)
        print("Connected to Roboflow API")

        workspace_name = os.getenv('ROBOFLOW_WORKSPACE')
        workspace = rf.workspace(workspace_name)
        print(f"Workspace found: {workspace.name}")

        project_name = os.getenv('ROBOFLOW_PROJECT')
        project = workspace.project(project_name)
        print(f"Project found: {project.name}")

        print(f"\nAll checks passed! Ready to:")
        print(f"   1. Upload images: python services/train/upload_to_roboflow.py")
        print(f"   2. Label in web: https://app.roboflow.com/{workspace_name}/{project_name}")
        print(f"   3. Download dataset: python services/train/download_from_roboflow.py")
        print(f"   4. Train model: python services/train/train_street_wheels.py")

        return True

    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Verify API key at: https://app.roboflow.com/settings/api")
        print(f"  2. Check workspace name: {workspace_name}")
        print(f"  3. Check project name: {project_name}")
        print(f"  4. Make sure project exists in Roboflow")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Roboflow Setup Test")
    print("=" * 60)
    print()

    success = test_roboflow_setup()
    sys.exit(0 if success else 1)
