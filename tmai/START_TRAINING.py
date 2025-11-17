#!/usr/bin/env python
"""
START TRAINING - Simple launcher that uses conda environment
Initializes storage on F: drive before training
Handles ViGEmBus driver detection and pause/resume functionality
"""

import sys
import os
import subprocess
from pathlib import Path

# SET PYCACHE LOCATION TO F-DRIVE IMMEDIATELY (before any imports)
os.environ['PYTHONPYCACHEDIR'] = str(Path('F:\\aidata\\pycache'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '0'  # Allow bytecode but only in F-drive

print("=" * 100)
print("TRACKMANIA RL - TRAINING LAUNCHER")
print("=" * 100)
print()
print("[OK] Python cache directory set to: F:\\aidata\\pycache")
print()

# Check if we're in conda environment
in_conda = os.environ.get('CONDA_DEFAULT_ENV') is not None

if not in_conda:
    print("Not running in conda environment!")
    print("Attempting to switch to conda base...")
    print()
    
    # Re-run with conda
    try:
        result = subprocess.run(
            ['conda', 'run', '-n', 'base', 'python', __file__],
            cwd=os.getcwd()
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"ERROR: Could not activate conda: {e}")
        print("Make sure Anaconda is installed and accessible")
        sys.exit(1)

# Now we're in conda, check dependencies
print(f"Python: {sys.executable}")
print(f"Version: {sys.version.split()[0]}")
print()

# Initialize storage structure on F: drive
print("Initializing storage on F:\\aidata...")
try:
    from storage_management import ensure_aidata_structure, get_storage_stats, cleanup_pycache, create_config_file
    ensure_aidata_structure()
    cleanup_pycache()
    create_config_file()
    get_storage_stats()
except Exception as e:
    print(f"Warning: Storage initialization failed: {e}")

# Check dependencies
print("\nChecking dependencies...")
required = ['torch', 'gymnasium', 'numpy', 'vgamepad', 'tmrl']
all_ok = True

for pkg in required:
    try:
        if pkg == 'vgamepad':
            __import__('vgamepad')
        elif pkg == 'tmrl':
            __import__('tmrl')
        else:
            __import__(pkg)
        print(f"  [OK] {pkg}")
    except ImportError:
        print(f"  [FAIL] {pkg} - NOT INSTALLED")
        all_ok = False

if not all_ok:
    print()
    print("ERROR: Some dependencies are missing!")
    print("Run: conda run -n base pip install -r requirements_rl.txt")
    sys.exit(1)

print()
print("=" * 100)
print("SYSTEM REQUIREMENTS CHECK")
print("=" * 100)
print()

# Check ViGEmBus and environment
try:
    from training_controller import initialize_training_environment
    if not initialize_training_environment():
        print("\n[FAIL] System requirements not met. Please fix errors above and try again.")
        sys.exit(1)
except Exception as e:
    print(f"Warning during environment check: {e}")
    print("Proceeding anyway (may fail when starting game)...")

print()
print("=" * 100)
print("STARTING TRAINING")
print("=" * 100)
print()
print("IMPORTANT:")
print("  1. Open Trackmania 2020")
print("  2. Start an ACTUAL RACE (not replay, not menu!)")
print("  3. Keep Trackmania window open but in background")
print()
print("Training will connect to Trackmania automatically...")
print("Monitor checkpoint progress and rewards below...")
print()
print("HOTKEYS DURING TRAINING:")
print("  Ctrl+P : Pause training")
print("  Ctrl+R : Resume from pause")
print("  Ctrl+S : Save and exit (resume later)")
print("  Ctrl+C : Stop training (graceful shutdown)")
print()

# Import and run training
try:
    from train_tmrl import main
    main()
except ImportError as e:
    print(f"ERROR: Could not import training script: {e}")
    sys.exit(1)
except KeyboardInterrupt:
    print("\n\nTraining stopped by user (Ctrl+C)")
    print("Checkpoints have been saved to: F:\\aidata\\trackmania_checkpoints\\")
    print("Logs saved to: F:\\aidata\\trackmania_logs\\")
    print("Models saved to: F:\\aidata\\trackmania_models\\")
    print("\nTo resume training later, run: python START_TRAINING.py")
    sys.exit(0)
except Exception as e:
    error_msg = str(e).lower()
    print(f"\n\nERROR during training: {e}")
    
    # Check if it's a ViGEmBus error
    if "vigem" in error_msg or "could not connect" in error_msg:
        from training_controller import ViGEmBusManager
        print(ViGEmBusManager.get_fix_instructions())
    else:
        import traceback
        traceback.print_exc()
    
    sys.exit(1)
