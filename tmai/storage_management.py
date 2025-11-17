#!/usr/bin/env python
"""
Storage Management Utility
Manages F:\aidata directory structure and cleans up pycache
"""

import os
import shutil
from pathlib import Path
import json
from datetime import datetime

AIDATA_PATH = Path('F:\\aidata')
CACHE_DIR = AIDATA_PATH / 'pycache'
CHECKPOINT_DIR = AIDATA_PATH / 'trackmania_checkpoints'
LOGS_DIR = AIDATA_PATH / 'trackmania_logs'
MODELS_DIR = AIDATA_PATH / 'trackmania_models'

def ensure_aidata_structure():
    """Create F:\\aidata directory structure"""
    dirs = [AIDATA_PATH, CACHE_DIR, CHECKPOINT_DIR, LOGS_DIR, MODELS_DIR]
    
    print("Creating F:\\aidata directory structure...")
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")
    
    # Create README
    readme_path = AIDATA_PATH / 'README.md'
    if not readme_path.exists():
        readme_content = """# Trackmania RL Training Data Storage

This directory stores all training-related data for the Trackmania RL project.

## Directory Structure

- **trackmania_checkpoints/** - Model checkpoints (policy + value networks)
- **trackmania_logs/** - Training logs and analytics
- **trackmania_models/** - Model weights and ONNX exports
- **pycache/** - Python bytecode cache

## Storage Information

All data is stored on the F: drive to conserve local device storage.
This includes:
- Model checkpoints (50-100 MB each)
- Training logs (varies, typically 1-10 MB per session)
- Model weights (5-20 MB each)
- Python cache (automatically managed)

## Usage

The training script automatically manages these directories. No manual intervention needed.

To check disk usage:
```powershell
Get-ChildItem F:\aidata -Recurse | Measure-Object -Property Length -Sum
```

To clean old checkpoints:
```python
from train_utils import TrainingAnalytics
analytics = TrainingAnalytics()
analytics.clean_old_checkpoints(keep_best_n=5)
```
"""
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"  ✓ Created {readme_path}")

def get_storage_stats():
    """Get storage statistics for F:\\aidata"""
    print("\n" + "="*60)
    print("F:\\aidata Storage Statistics")
    print("="*60)
    
    def get_size(path):
        total = 0
        try:
            for entry in os.scandir(path):
                if entry.is_file(follow_symlinks=False):
                    total += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total += get_size(entry.path)
        except PermissionError:
            pass
        return total
    
    for name, path in [
        ("Checkpoints", CHECKPOINT_DIR),
        ("Logs", LOGS_DIR),
        ("Models", MODELS_DIR),
        ("Cache", CACHE_DIR),
    ]:
        if path.exists():
            size = get_size(path)
            size_mb = size / 1024 / 1024
            # Count files
            file_count = len(list(path.glob('**/*')))
            print(f"{name:<15}: {size_mb:>10.2f} MB ({file_count} files)")
    
    total_size = get_size(AIDATA_PATH)
    print("-" * 60)
    print(f"{'TOTAL':<15}: {total_size / 1024 / 1024:>10.2f} MB")
    print("="*60)

def cleanup_pycache():
    """Remove pycache from source directory"""
    source_dir = Path(__file__).parent
    pycache_local = source_dir / '__pycache__'
    
    if pycache_local.exists():
        print(f"\nRemoving local pycache: {pycache_local}")
        shutil.rmtree(pycache_local)
        print("  ✓ Local pycache removed")

def create_config_file():
    """Create storage config file"""
    config = {
        "storage": {
            "base_path": str(AIDATA_PATH),
            "checkpoints": str(CHECKPOINT_DIR),
            "logs": str(LOGS_DIR),
            "models": str(MODELS_DIR),
            "pycache": str(CACHE_DIR),
        },
        "created": datetime.now().isoformat(),
        "description": "Trackmania RL training data storage configuration"
    }
    
    config_path = AIDATA_PATH / 'storage_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Storage configuration saved to: {config_path}")

def main():
    """Main storage management function"""
    print("="*60)
    print("Trackmania RL - Storage Management Utility")
    print("="*60)
    
    # Ensure structure
    ensure_aidata_structure()
    
    # Cleanup local pycache
    cleanup_pycache()
    
    # Create config
    create_config_file()
    
    # Show stats
    get_storage_stats()
    
    print("\n✓ Storage management completed!")
    print("\nNext step: Run START_TRAINING.py to begin training")

if __name__ == "__main__":
    main()
