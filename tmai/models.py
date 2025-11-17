"""
Neural Network Models for Trackmania RL
Includes Policy Network and Value Network for Actor-Critic algorithms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os

# Ensure F:\aidata directories exist
AIDATA_PATH = Path('F:\\aidata')
CHECKPOINT_DIR = AIDATA_PATH / 'trackmania_checkpoints'
LOGS_DIR = AIDATA_PATH / 'trackmania_logs'
MODELS_DIR = AIDATA_PATH / 'trackmania_models'
CACHE_DIR = AIDATA_PATH / 'pycache'

# Create directories
for dir_path in [CHECKPOINT_DIR, LOGS_DIR, MODELS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Set Python cache directory
os.environ['PYTHONPYCACHEDIR'] = str(CACHE_DIR)


class PolicyNetwork(nn.Module):
    """
    Actor Network - outputs action mean and std for continuous control
    Input: LIDAR readings (19 sensors)
    Output: steering [-1, 1] and acceleration [0, 1]
    """
    
    def __init__(self, state_dim=19, action_dim=2, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Policy head
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)
        
        # Initialize mean layer with small weights
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        mean = torch.tanh(self.mean(x))  # Steering [-1, 1], Accel [-1, 1] -> will transform
        
        return mean
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy
        state: torch tensor of shape (batch_size, state_dim) or (state_dim,)
        """
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        mean = self.forward(state)
        
        if deterministic:
            return mean.detach()
        
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        
        # Clamp to valid ranges
        action = torch.clamp(action, -1.0, 1.0)
        
        return action.detach(), dist
    
    def evaluate(self, state, action):
        """Evaluate action log probability and entropy"""
        mean = self.forward(state)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().mean()
        
        return log_prob, entropy


class ValueNetwork(nn.Module):
    """
    Critic Network - estimates value of state
    Input: LIDAR readings (19 sensors)
    Output: scalar value estimate
    """
    
    def __init__(self, state_dim=19, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        
        self.state_dim = state_dim
        
        # Feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Value head
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(module.bias, 0)
        
        nn.init.orthogonal_(self.value.weight, gain=0.01)
        nn.init.constant_(self.value.bias, 0)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.value(x)
        
        return value


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network for discrete action spaces (if you want to use it)
    Separates value and advantage streams
    """
    
    def __init__(self, state_dim=19, action_dim=7, hidden_dim=256):
        super(DuelingDQNNetwork, self).__init__()
        
        # Shared feature extraction
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Advantage stream
        self.adv_fc = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        """Forward pass"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)
        
        # Advantage stream
        adv = F.relu(self.adv_fc(x))
        adv = self.advantage(adv)
        
        # Q-value = Value + (Advantage - mean Advantage)
        q_value = value + (adv - adv.mean(dim=1, keepdim=True))
        
        return q_value


class ModelCheckpoint:
    """Utility for saving and loading model checkpoints"""
    
    def __init__(self, checkpoint_dir=None):
        if checkpoint_dir is None:
            checkpoint_dir = CHECKPOINT_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save(self, policy_net, value_net, episode, reward, optimizer_p=None, optimizer_v=None):
        """Save checkpoint"""
        checkpoint = {
            'episode': episode,
            'reward': reward,
            'policy_state': policy_net.state_dict(),
            'value_state': value_net.state_dict(),
        }
        
        if optimizer_p is not None:
            checkpoint['optimizer_p_state'] = optimizer_p.state_dict()
        if optimizer_v is not None:
            checkpoint['optimizer_v_state'] = optimizer_v.state_dict()
        
        path = self.checkpoint_dir / f'checkpoint_ep{episode}_r{reward:.0f}.pt'
        torch.save(checkpoint, path)
        return path
    
    def load(self, policy_net, value_net, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        policy_net.load_state_dict(checkpoint['policy_state'])
        value_net.load_state_dict(checkpoint['value_state'])
        
        return checkpoint.get('episode', 0), checkpoint.get('reward', 0)
    
    def get_latest_checkpoint(self):
        """Get latest checkpoint"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None
        
        # Sort by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest
    
    def get_best_checkpoint(self):
        """Get best checkpoint by reward"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_*.pt'))
        if not checkpoints:
            return None
        
        # Parse reward from filename
        best_checkpoint = None
        best_reward = -float('inf')
        
        for cp in checkpoints:
            try:
                # Extract reward from filename: checkpoint_ep{episode}_r{reward}.pt
                reward_str = cp.stem.split('_r')[1]
                reward = float(reward_str)
                
                if reward > best_reward:
                    best_reward = reward
                    best_checkpoint = cp
            except:
                continue
        
        return best_checkpoint


if __name__ == "__main__":
    # Test the models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dummy input
    batch_size = 4
    state_dim = 19
    action_dim = 2
    
    state = torch.randn(batch_size, state_dim).to(device)
    
    # Test Policy Network
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    action, dist = policy.get_action(state, deterministic=False)
    print(f"Policy Network output shape: {action.shape}")
    print(f"Sample action: {action[0]}")
    
    # Test Value Network
    value_net = ValueNetwork(state_dim).to(device)
    value = value_net(state)
    print(f"Value Network output shape: {value.shape}")
    
    # Test checkpoint
    checkpoint_mgr = ModelCheckpoint()
    print(f"Checkpoint directory: {checkpoint_mgr.checkpoint_dir}")
