"""
Training Management Utilities
Checkpoint management, model evaluation, and training analytics
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import os

from models import PolicyNetwork, ValueNetwork, ModelCheckpoint, LOGS_DIR, CHECKPOINT_DIR, MODELS_DIR


os.environ['PYTHONPYCACHEDIR'] = str(Path('F:\\aidata\\pycache'))


logger = logging.getLogger(__name__)


class TrainingAnalytics:
    """Analyze training statistics and logs"""
    
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = LOGS_DIR
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, log_files=None):
        """Plot training reward curves"""
        if log_files is None:
            log_files = sorted(self.log_dir.glob('training_*.log'))
        
        for log_file in log_files:
            rewards = []
            episodes = []
            
            with open(log_file, 'r') as f:
                for line in f:
                    if 'Avg Reward:' in line:
                        try:
                            # Parse: "Avg Reward: X.XX"
                            reward_str = line.split('Avg Reward:')[1].split('|')[0].strip()
                            reward = float(reward_str)
                            rewards.append(reward)
                            episodes.append(len(episodes))
                        except:
                            pass
            
            if rewards:
                plt.figure(figsize=(12, 6))
                plt.plot(episodes, rewards, 'b-', alpha=0.7, label='Average Reward')
                
                # Add moving average
                window = max(1, len(rewards) // 20)
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(episodes)), moving_avg, 'r-', linewidth=2, label='Moving Average')
                
                plt.xlabel('Update Step')
                plt.ylabel('Average Reward')
                plt.title(f'Training Progress - {log_file.name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Save plot
                plot_path = self.log_dir / f'{log_file.stem}_curve.png'
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Saved plot: {plot_path}")
    
    def get_best_checkpoint_stats(self):
        """Get statistics for best checkpoint"""
        checkpoint_dir = CHECKPOINT_DIR
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
        
        if not checkpoints:
            print("No checkpoints found")
            return
        
        best_checkpoint = None
        best_reward = -float('inf')
        
        for cp in checkpoints:
            try:
                reward_str = cp.stem.split('_r')[1]
                reward = float(reward_str)
                
                if reward > best_reward:
                    best_reward = reward
                    best_checkpoint = cp
            except:
                continue
        
        if best_checkpoint:
            checkpoint = torch.load(best_checkpoint, map_location='cpu')
            
            print(f"\n{'='*60}")
            print("BEST CHECKPOINT STATISTICS")
            print(f"{'='*60}")
            print(f"Checkpoint: {best_checkpoint.name}")
            print(f"Episode: {checkpoint['episode']}")
            print(f"Reward: {best_reward:.2f}")
            print(f"File Size: {best_checkpoint.stat().st_size / 1024 / 1024:.2f} MB")
            print(f"{'='*60}\n")
    
    def clean_old_checkpoints(self, keep_best_n=5):
        """Keep only N best checkpoints to save space"""
        checkpoint_dir = CHECKPOINT_DIR
        checkpoints = list(checkpoint_dir.glob('checkpoint_*.pt'))
        
        # Sort by reward
        checkpoint_rewards = []
        for cp in checkpoints:
            try:
                reward_str = cp.stem.split('_r')[1]
                reward = float(reward_str)
                checkpoint_rewards.append((cp, reward))
            except:
                continue
        
        # Sort by reward (descending)
        checkpoint_rewards.sort(key=lambda x: x[1], reverse=True)
        
        # Delete old ones
        deleted_count = 0
        for cp, reward in checkpoint_rewards[keep_best_n:]:
            cp.unlink()
            deleted_count += 1
            print(f"Deleted: {cp.name}")
        
        print(f"\nCleaned up {deleted_count} old checkpoints. Kept {keep_best_n} best.")


class ModelConverter:
    """Convert and optimize models"""
    
    @staticmethod
    def convert_to_onnx(policy_net, output_path=None):
        """Convert PyTorch model to ONNX format"""
        if output_path is None:
            output_path = MODELS_DIR / 'trackmania_model.onnx'
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            import onnx
            
            dummy_input = torch.randn(1, 19)
            torch.onnx.export(
                policy_net,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['lidar_input'],
                output_names=['action'],
                verbose=False
            )
            print(f"✓ Model exported to ONNX: {output_path}")
        except ImportError:
            print("ONNX not installed. Install with: pip install onnx")
    
    @staticmethod
    def quantize_model(policy_net, device='cpu'):
        """Quantize model to reduce size"""
        quantized_model = torch.quantization.quantize_dynamic(
            policy_net,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        return quantized_model


class CheckpointManager:
    """Advanced checkpoint management"""
    
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def list_checkpoints(self):
        """List all checkpoints with stats"""
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_*.pt'))
        
        print(f"\n{'='*80}")
        print(f"{'Episode':<10} {'Reward':<12} {'File Size':<15} {'Date Modified':<20} {'Name'}")
        print(f"{'='*80}")
        
        for cp in checkpoints:
            try:
                checkpoint = torch.load(cp, map_location='cpu')
                episode = checkpoint['episode']
                reward = checkpoint['reward']
                size_mb = cp.stat().st_size / 1024 / 1024
                mtime = datetime.fromtimestamp(cp.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                
                print(f"{episode:<10} {reward:<12.2f} {size_mb:<12.2f} MB {mtime:<20} {cp.name}")
            except Exception as e:
                print(f"Error reading {cp.name}: {e}")
        
        print(f"{'='*80}\n")
    
    def get_checkpoint_info(self, checkpoint_path):
        """Get detailed info about a checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'episode': checkpoint.get('episode', 0),
            'reward': checkpoint.get('reward', 0),
            'file_size_mb': checkpoint_path.stat().st_size / 1024 / 1024,
            'policy_params': sum(p.numel() for p in checkpoint['policy_state'].values()),
            'value_params': sum(p.numel() for p in checkpoint['value_state'].values()),
        }
        
        return info


def main():
    """Main utilities function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trackmania RL Training Utilities')
    parser.add_argument('--analyze', action='store_true', help='Analyze training logs')
    parser.add_argument('--list-checkpoints', action='store_true', help='List all checkpoints')
    parser.add_argument('--best-checkpoint', action='store_true', help='Show best checkpoint')
    parser.add_argument('--clean-checkpoints', type=int, nargs='?', const=5, help='Keep N best checkpoints')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to analyze')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    analytics = TrainingAnalytics()
    checkpoint_mgr = CheckpointManager()
    
    if args.analyze:
        print("Analyzing training logs...")
        analytics.plot_training_curves()
    
    if args.list_checkpoints:
        checkpoint_mgr.list_checkpoints()
    
    if args.best_checkpoint:
        analytics.get_best_checkpoint_stats()
    
    if args.clean_checkpoints:
        print(f"Cleaning checkpoints, keeping {args.clean_checkpoints} best...")
        analytics.clean_old_checkpoints(keep_best_n=args.clean_checkpoints)
    
    if args.checkpoint:
        info = checkpoint_mgr.get_checkpoint_info(args.checkpoint)
        print(f"\n{'='*60}")
        print("CHECKPOINT INFORMATION")
        print(f"{'='*60}")
        for key, value in info.items():
            print(f"{key:<20}: {value}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
