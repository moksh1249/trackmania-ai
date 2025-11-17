#!/usr/bin/env python
"""
Log analyzer for checkpoint and track completion detection
Parses training logs and shows checkpoint/completion statistics
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_logs(log_file):
    """Analyze training logs for checkpoint/completion detection"""
    
    if not Path(log_file).exists():
        print(f"Log file not found: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    checkpoints_detected = 0
    completions_detected = 0
    episodes_analyzed = 0
    
    print("\n" + "="*80)
    print("CHECKPOINT & COMPLETION DETECTION ANALYSIS")
    print("="*80)
    print(f"Log File: {log_file}")
    print(f"Total Lines: {len(lines)}")
    print()
    
    # Look for detection markers
    for i, line in enumerate(lines):
        if "[TRACK COMPLETE]" in line:
            completions_detected += 1
            print(f"[{completions_detected}] Track Completion at line {i+1}:")
            print(f"    {line.strip()}")
        
        if "[CHECKPOINT]" in line and "[TRACK COMPLETE]" not in line:
            checkpoints_detected += 1
            print(f"[{checkpoints_detected}] Checkpoint at line {i+1}:")
            print(f"    {line.strip()}")
        
        if "[EPISODE" in line and "SUMMARY" in line:
            episodes_analyzed += 1
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Episodes Analyzed: {episodes_analyzed}")
    print(f"Checkpoints Detected: {checkpoints_detected}")
    print(f"Track Completions Detected: {completions_detected}")
    
    if episodes_analyzed > 0:
        checkpoint_rate = (checkpoints_detected / episodes_analyzed) * 100
        completion_rate = (completions_detected / episodes_analyzed) * 100
        print(f"Checkpoint Detection Rate: {checkpoint_rate:.1f}%")
        print(f"Completion Detection Rate: {completion_rate:.1f}%")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # Get the latest log file from F:\aidata\trackmania_logs
    logs_dir = Path("F:\\aidata\\trackmania_logs")
    
    if not logs_dir.exists():
        print(f"Logs directory not found: {logs_dir}")
        print("\nUsage:")
        print("  python analyze_logs.py")
        print("  (will use latest log from F:\\aidata\\trackmania_logs\\)")
        print("\nOr:")
        print("  python analyze_logs.py <path_to_log_file>")
        sys.exit(1)
    
    # Find latest log file
    log_files = sorted(logs_dir.glob("training_tmrl_*.log"), reverse=True)
    
    if not log_files:
        print(f"No training logs found in {logs_dir}")
        sys.exit(1)
    
    latest_log = log_files[0]
    print(f"\nUsing latest log: {latest_log.name}")
    
    analyze_logs(str(latest_log))
