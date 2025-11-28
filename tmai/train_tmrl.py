"""
Trackmania RL Training Script using TMRL Official Library
Trains an AI driver using PPO algorithm with TMRL's environment
"""

import sys
import os
from pathlib import Path

# SET PYCACHE LOCATION TO F-DRIVE IMMEDIATELY (before any imports)
os.environ['PYTHONPYCACHEDIR'] = str(Path('F:\\aidata\\pycache'))
os.environ['PYTHONDONTWRITEBYTECODE'] = '0'  # Allow bytecode but only in F-drive

import torch
import torch.optim as optim
import numpy as np
import time
import json
from collections import deque
import logging
from datetime import datetime

# TMRL import (will be done dynamically in main())
from models import PolicyNetwork, ValueNetwork, ModelCheckpoint, LOGS_DIR, CHECKPOINT_DIR, MODELS_DIR
from training_controller import (
    ViGEmBusManager,
    TrainingCheckpoint,
    PauseResumeController,
    initialize_training_environment
)

# Set cache directory
os.environ['PYTHONPYCACHEDIR'] = str(Path('F:\\aidata\\pycache'))

# Setup logging to F drive
log_dir = LOGS_DIR
log_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'training_tmrl_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PPOTrainerTMRL:
    """
    PPO (Proximal Policy Optimization) trainer for Trackmania using TMRL
    With 40-second timeout and checkpoint tracking
    """
    
    def __init__(
        self,
        env,
        policy_net,
        value_net,
        device,
        lr_policy=3e-4,
        lr_value=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        value_loss_coeff=0.5,
        max_grad_norm=0.5,
        batch_size=64,
        n_epochs=3,
        n_steps=2048,
        timeout_seconds=40,
    ):
        """Initialize PPO trainer"""
        self.env = env
        self.policy_net = policy_net.to(device)
        self.value_net = value_net.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_steps = n_steps
        self.timeout_seconds = timeout_seconds

        self.optimizer_policy = optim.Adam(policy_net.parameters(), lr=lr_policy)
        self.optimizer_value = optim.Adam(value_net.parameters(), lr=lr_value)
        

        self.checkpoint_mgr = ModelCheckpoint()

        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.global_step = 0
        self.episode = 0
        self.total_reward = 0
        
        # Episode state tracking
        self.episode_start_time = time.time()
        self.episode_checkpoint_reached = False
        self.episode_track_completion = False
        self.episode_max_reward = 0
        
        logger.info("PPOTrainerTMRL initialized")
        logger.info(f"Device: {device}")
        logger.info(f"Policy Network: {policy_net}")
        logger.info(f"Value Network: {value_net}")
        logger.info(f"Timeout per episode: {timeout_seconds} seconds")
    
    def check_episode_timeout(self):
        """Check if episode exceeded timeout without progress"""
        elapsed = time.time() - self.episode_start_time
        
        # Return True if timeout exceeded with no checkpoint/completion
        if elapsed > self.timeout_seconds:
            if not self.episode_checkpoint_reached and not self.episode_track_completion:
                return True
        
        return False
    
    def reset_car_on_track(self, track_completed=False, timeout=False):
        """
        Reset car position on track using appropriate method
        
        Different reset scenarios:
        1. Track completed: Need to restart from beginning
        2. Timeout/no progress: Need to respawn at current position  
        3. Normal reset: Reset to start
        
        Args:
            track_completed: Whether track was just completed
            timeout: Whether reset is due to timeout (no progress)
        """
        try:
            # Use TMRL's reset mechanism
            # env.reset() communicates with OpenPlanet to reset car position
            obs, info = self.env.reset()
            
            if track_completed:
                logger.warning(f"[RESET] Car reset after TRACK COMPLETION - new attempt starting")
            elif timeout:
                logger.warning(f"[RESET] Car respawned after TIMEOUT - {self.timeout_seconds}s with no progress")
            else:
                logger.info(f"[RESET] Car reset - episode {self.episode} starting")
            
            return obs, info
        except Exception as e:
            logger.error(f"[ERROR] Failed to reset car: {e}")
            logger.warning("Attempting to continue with current state...")
            return None, None
    
    def extract_checkpoint_info(self, info, log_this_step=True):
        """Extract checkpoint and completion info from TMRL info dict"""
        checkpoint_reached = False
        track_completed = False
        checkpoint_value = 0
        
        # Check various possible info keys from TMRL/OpenPlanet
        if isinstance(info, dict):
            # Only log keys if not empty AND first time in episode
            if len(info) > 0 and log_this_step:
                available_keys = list(info.keys())
                print(f"[INFO KEYS] {available_keys}")
                logger.info(f"[INFO] Available keys in info dict: {available_keys}")
                
                # Print full info dict content for debugging (JSON format)
                try:
                    info_str = json.dumps({k: str(v)[:50] for k, v in info.items()}, indent=2)
                    print(f"[INFO DICT]\n{info_str}")
                    logger.info(f"[INFO] Full info dict:\n{info_str}")
                except Exception as e:
                    logger.info(f"[INFO] Could not serialize info dict: {e}")
            
            # Check for ANY True boolean value that might indicate completion/checkpoint
            for key, value in info.items():
                if isinstance(value, bool) and value:
                    print(f"[FOUND TRUE] '{key}' = True")
                    logger.warning(f"[FOUND TRUE] '{key}' = True")
                    
                    # Check if it's likely a checkpoint key
                    if any(x in key.lower() for x in ['check', 'cp', 'waypoint', 'passage']):
                        checkpoint_reached = True
                        logger.warning(f"[CHECKPOINT DETECTED] Key '{key}' matches checkpoint pattern")
                    
                    # Check if it's likely a completion key
                    if any(x in key.lower() for x in ['finish', 'complete', 'done', 'end', 'race', 'goal']):
                        track_completed = True
                        logger.warning(f"[COMPLETION DETECTED] Key '{key}' matches completion pattern")
            
            # Explicit key checks (original method as fallback)
            if 'checkpoint' in info:
                checkpoint_reached = bool(info.get('checkpoint', False)) or checkpoint_reached
                logger.info(f"[CHECKPOINT] 'checkpoint' key found: {info.get('checkpoint')}")
            if 'checkpoints_reached' in info:
                val = bool(info.get('checkpoints_reached', False))
                checkpoint_reached = val or checkpoint_reached
                logger.info(f"[CHECKPOINT] 'checkpoints_reached' key found: {val}")
            if 'is_checkpoint' in info:
                val = bool(info.get('is_checkpoint', False))
                checkpoint_reached = val or checkpoint_reached
                logger.info(f"[CHECKPOINT] 'is_checkpoint' key found: {val}")
            if 'checkpoint_count' in info:
                checkpoint_value = int(info.get('checkpoint_count', 0))
                if checkpoint_value > 0:
                    checkpoint_reached = True
                    logger.info(f"[CHECKPOINT] 'checkpoint_count' = {checkpoint_value}")
            
            # Track completion - multiple possible keys
            if 'track_completed' in info:
                val = bool(info.get('track_completed', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'track_completed' key found: TRUE")
            if 'finished' in info:
                val = bool(info.get('finished', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'finished' key found: TRUE")
            if 'done' in info:
                val = bool(info.get('done', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'done' key found: TRUE")
            if 'goal_reached' in info:
                val = bool(info.get('goal_reached', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'goal_reached' key found: TRUE")
            if 'race_finished' in info:
                val = bool(info.get('race_finished', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'race_finished' key found: TRUE")
            
            # Additional keys that might indicate completion
            if 'end_of_track' in info:
                val = bool(info.get('end_of_track', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'end_of_track' key found: TRUE")
            if 'track_complete' in info:
                val = bool(info.get('track_complete', False))
                track_completed = val or track_completed
                if val:
                    logger.warning(f"[TRACK_COMPLETE] 'track_complete' key found: TRUE")
        
        return checkpoint_reached, track_completed, checkpoint_value
    
    def collect_experience(self):
        """Collect experience trajectory from TMRL environment"""
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        obs, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        # Reset episode tracking
        self.episode_start_time = time.time()
        self.episode_checkpoint_reached = False
        self.episode_track_completion = False
        self.episode_max_reward = 0
        
        for step in range(self.n_steps):

            log_info = (step == 0)
            checkpoint_reached, track_completed, checkpoint_val = self.extract_checkpoint_info(info, log_this_step=log_info)
            

            if checkpoint_reached:
                self.episode_checkpoint_reached = True
                logger.warning(f"[INFO DICT] Checkpoint detected at step {step}")
            if track_completed:
                self.episode_track_completion = True
                logger.warning(f"[INFO DICT] Track completion detected at step {step}")
            

            if self.check_episode_timeout():

                obs, info = self.reset_car_on_track(track_completed=False, timeout=True)
                if obs is None:

                    step_output = self.create_empty_trajectory_tensors(step)
                    states, actions, rewards, values, log_probs, dones = step_output
                    return states, actions, rewards, values, log_probs, dones
                
                self.episode_start_time = time.time()
                self.episode_checkpoint_reached = False
                self.episode_track_completion = False
                self.episode += 1
                step_output = self.create_empty_trajectory_tensors(step)
                states, actions, rewards, values, log_probs, dones = step_output

                return states, actions, rewards, values, log_probs, dones

            if isinstance(obs, tuple):

                state_components = []
                
                for i, component in enumerate(obs):
                    if isinstance(component, np.ndarray):
                        if component.ndim == 0:
                        
                            state_components.append(float(component))
                        elif component.ndim == 1:
                        
                            state_components.extend(component.flatten())
                        elif component.ndim == 2:
                            
                            if component.shape[0] <= 5: 
                                state_components.extend(component.flatten())
                           
                        else:
                 
                            pass
                    else:
                        state_components.append(float(component))
                
                state_input = np.array(state_components, dtype=np.float32)
            else:
                state_input = obs

            if len(state_input) < 19:
                state_input = np.pad(state_input, (0, 19 - len(state_input)), mode='constant')
            else:
                # Take only first 19 dimensions
                state_input = state_input[:19]
            
            state_tensor = torch.from_numpy(state_input).float().to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, dist = self.policy_net.get_action(state_tensor, deterministic=False)
                value = self.value_net(state_tensor)
                log_prob = dist.log_prob(action).sum(dim=-1)
            

            action_np = action.cpu().numpy().flatten()
            
         
            steering = action_np[0]  
            accel = action_np[1]     
            
         
            gas = max(0, accel)      
            brake = max(0, -accel)   
            

            tmrl_action = np.array([gas, brake, steering], dtype=np.float32)
            
  
            if step == 0:
                logger.info(f"[ACTION MAPPING] steering={steering:.3f} accel={accel:.3f} -> gas={gas:.3f} brake={brake:.3f} steer={steering:.3f}")
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(tmrl_action)

            original_reward = reward
            

            accel = action_np[1]  
            

            if accel > 0.1: 
                forward_bonus = accel * 0.5 
                reward += forward_bonus
            elif accel < -0.1:  # Significant backward acceleration
                backward_penalty = abs(accel) * 1.5  # -0.15 to -1.5 SEVERE penalty for backward
                reward -= backward_penalty
                logger.info(f"[BACKWARD] Severe penalty applied: {backward_penalty:.3f} (reward before: {original_reward:.2f}, after: {reward:.2f})")
            

            if gas > 0.2: 
                gas_bonus = gas * 0.3 
                reward += gas_bonus
            

            if original_reward > 50:  # Large reward from TMRL
                logger.warning(f"[LARGE REWARD] {original_reward:.2f} - Might indicate checkpoint/completion!")
                if original_reward > 100:
                    track_completed = True
                    self.episode_track_completion = True
                    logger.warning(f"[REWARD-DETECT] Large reward {original_reward:.2f} detected - Setting COMPLETION flag!")
                elif original_reward > 50:
                    checkpoint_reached = True
                    self.episode_checkpoint_reached = True
                    logger.warning(f"[REWARD-DETECT] Medium reward {original_reward:.2f} detected - Setting CHECKPOINT flag!")
            
            # Force done if checkpoint or track completed
            done = terminated or truncated or self.episode_checkpoint_reached or self.episode_track_completion
            
            # Log immediately when track is completed
            if self.episode_track_completion and not done:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"[TRACK COMPLETION DETECTED] Episode {self.episode}\n"
                    f"Current Episode Reward: {episode_reward:.2f}\n"
                    f"Step: {step}/{self.n_steps}\n"
                    f"Elapsed Time: {time.time() - self.episode_start_time:.2f}s\n"
                    f"{'='*80}\n"
                )
                done = True  # Force episode end
            
            # Store experience
            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            dones.append(done)
            
            episode_reward += reward
            episode_length += 1
            self.global_step += 1
            self.total_reward += reward
            self.episode_max_reward = max(self.episode_max_reward, reward)
            
            # Real-time terminal display
            elapsed_ep = time.time() - self.episode_start_time
            status_str = "[COMPLETED]" if self.episode_track_completion else ""
            status_str = "[CHECKPOINT]" if (self.episode_checkpoint_reached and not status_str) else status_str
            status_str = f"({elapsed_ep:.1f}s)" if not status_str else status_str
            
            # Print progress every 50 steps
            if step % 50 == 0:
                sys.stdout.write(f"\r[Ep {self.episode}] Step {step}/{self.n_steps} | "
                               f"Reward: +{reward:+.2f} → {episode_reward:+.2f} | "
                               f"Max: {self.episode_max_reward:.2f} | Total: {self.total_reward:.0f} | "
                               f"{status_str}")
                sys.stdout.flush()
            
            if done:
                # Episode finished - log checkpoint/completion info
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.episode += 1
                
                completion_str = "[TRACK COMPLETED]" if self.episode_track_completion else ""
                checkpoint_str = "[CHECKPOINT REACHED]" if self.episode_checkpoint_reached else ""
                if completion_str and checkpoint_str:
                    episode_status = f"{completion_str} + {checkpoint_str}"
                elif completion_str or checkpoint_str:
                    episode_status = completion_str or checkpoint_str
                else:
                    episode_status = "[NO PROGRESS]"
                
                # Log episode summary
                print()  # New line after progress
                logger.warning(
                    f"\n{'='*80}\n"
                    f"[EPISODE {self.episode - 1} SUMMARY]\n"
                    f"Status: {episode_status}\n"
                    f"Reward: {episode_reward:.2f} | "
                    f"Length: {episode_length} steps | "
                    f"Elapsed: {elapsed_ep:.2f}s\n"
                    f"Global Step: {self.global_step} | "
                    f"Total Accumulated Reward: {self.total_reward:.0f}\n"
                    f"{'='*80}\n"
                )
                
                if self.episode % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards)
                    avg_length = np.mean(self.episode_lengths)
                    logger.info(
                        f"[10-EPISODE AVERAGE] "
                        f"Avg Reward: {avg_reward:.2f} | "
                        f"Avg Length: {avg_length:.0f} steps"
                    )
                
                # Reset for next episode using appropriate method
                was_track_completed = self.episode_track_completion
                obs, info = self.reset_car_on_track(track_completed=was_track_completed, timeout=False)
                
                if obs is None:
                    # If reset failed, get fresh environment
                    logger.warning("[RESET] Using fallback env.reset() after reset_car_on_track failure")
                    obs, info = self.env.reset()
                
                episode_reward = 0
                episode_length = 0
                
                # Reset episode tracking flags for new episode
                self.episode_start_time = time.time()
                self.episode_checkpoint_reached = False
                self.episode_track_completion = False
                self.episode_max_reward = 0
        
        return states, actions, rewards, values, log_probs, dones
    
    def compute_returns_and_advantages(self, rewards, values, dones):
        """Compute returns and advantages using GAE"""
        returns = []
        advantages = []
        
        next_value = 0
        gae = 0
        
        for t in reversed(range(len(rewards))):
            value = values[t].item() if isinstance(values[t], torch.Tensor) else values[t]
            done = dones[t]
            
            if t == len(rewards) - 1:
                next_value_t = 0 if done else next_value
            else:
                next_value_t = values[t + 1].item() if isinstance(values[t + 1], torch.Tensor) else values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - done) - value
            gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + value)
        
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_epoch(self, states, actions, old_log_probs, returns, advantages):
        """Train one epoch"""
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        n_samples = len(states)
        indices = np.random.permutation(n_samples)
        
        for epoch in range(self.n_epochs):
            for batch_idx in range(0, n_samples, self.batch_size):
                batch_indices = indices[batch_idx:batch_idx + self.batch_size]
                
                batch_states = torch.stack([states[i] for i in batch_indices])
                batch_actions = torch.stack([actions[i] for i in batch_indices])
                batch_old_log_probs = torch.stack([old_log_probs[i] for i in batch_indices])
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Policy loss
                new_log_probs, entropy = self.policy_net.evaluate(batch_states, batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * entropy
                
                # Value loss
                predicted_values = self.value_net(batch_states).squeeze(-1)
                value_loss = torch.nn.functional.mse_loss(predicted_values, batch_returns)
                
                # Total loss
                total_loss = policy_loss + self.value_loss_coeff * value_loss
                
                # Backward pass
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy_net.parameters()) + list(self.value_net.parameters()),
                    self.max_grad_norm
                )
                
                self.optimizer_policy.step()
                self.optimizer_value.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        return total_policy_loss / num_updates, total_value_loss / num_updates, total_entropy / num_updates
    
    def train(self, num_iterations=1000):
        """Main training loop"""
        logger.info("=" * 100)
        logger.info("Starting TMRL-based Trackmania RL Training with Checkpoint Tracking")
        logger.info("=" * 100)
        logger.info(f"Total iterations: {num_iterations}")
        logger.info(f"Steps per iteration: {self.n_steps}")
        logger.info(f"Total steps: {num_iterations * self.n_steps}")
        logger.info(f"Model reset timeout: {self.timeout_seconds} seconds")
        logger.info(f"Checkpoints saved to: {CHECKPOINT_DIR}")
        logger.info(f"Models saved to: {MODELS_DIR}")
        logger.info(f"Logs saved to: {LOGS_DIR}")
        logger.info("=" * 100)
        
        start_time = time.time()
        best_reward = -float('inf')
        
        try:
            for iteration in range(num_iterations):
                iter_start = time.time()
                
                # Collect experience
                states, actions, rewards, values, log_probs, dones = self.collect_experience()
                
                # Compute returns and advantages
                returns, advantages = self.compute_returns_and_advantages(rewards, values, dones)
                
                # Train
                policy_loss, value_loss, entropy = self.train_epoch(states, actions, log_probs, returns, advantages)
                
                iter_time = time.time() - iter_start
                
                # SAVE MODEL AFTER EVERY EPISODE regardless of completion
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                model_path = MODELS_DIR / f'trackmania_policy_ep{self.episode}_{timestamp}.pt'
                torch.save(self.policy_net.state_dict(), model_path)
                logger.debug(f"[OK] Model saved: {model_path}")
                
                # Display training data EVERY iteration
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
                max_reward = max(self.episode_rewards) if self.episode_rewards else 0
                min_reward = min(self.episode_rewards) if self.episode_rewards else 0
                
                # Update best reward
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    is_best = " [NEW BEST!]"
                else:
                    is_best = ""
                
                # Display training metrics every iteration
                logger.info(
                    f"\n{'='*120}\n"
                    f"[ITERATION {iteration + 1:4d}/{num_iterations}] "
                    f"Episode: {self.episode:5d} | "
                    f"Steps/Iter: {len(rewards):4d}\n"
                    f"\n"
                    f"REWARDS:\n"
                    f"  Avg Episode: {avg_reward:10.2f}{is_best} | "
                    f"Max: {max_reward:10.2f} | "
                    f"Min: {min_reward:10.2f}\n"
                    f"  Total Accumulated: {self.total_reward:12.0f} | "
                    f"Global Steps: {self.global_step:8d}\n"
                    f"\n"
                    f"LOSSES:\n"
                    f"  Policy Loss: {policy_loss:10.6f} | "
                    f"Value Loss: {value_loss:10.6f} | "
                    f"Entropy: {entropy:10.6f}\n"
                    f"\n"
                    f"TIMING:\n"
                    f"  Iteration: {iter_time:8.2f}s | "
                    f"Total Elapsed: {elapsed:10.0f}s ({elapsed/3600:6.2f}h) | "
                    f"Avg/Iter: {elapsed/(iteration+1):8.2f}s\n"
                    f"\n"
                    f"EPISODE STATE:\n"
                    f"  Checkpoint Reached: {'YES' if self.episode_checkpoint_reached else 'NO'} | "
                    f"Track Completed: {'YES' if self.episode_track_completion else 'NO'}\n"
                    f"{'='*120}\n"
                )
                
                # Save checkpoint every 10 iterations
                if (iteration + 1) % 10 == 0:
                    checkpoint_path = self.checkpoint_mgr.save(
                        self.policy_net,
                        self.value_net,
                        self.episode,
                        avg_reward,
                        self.optimizer_policy,
                        self.optimizer_value
                    )
                    logger.info(f"[OK] Checkpoint saved (Iteration {iteration+1}): {checkpoint_path}")
            
            logger.info("=" * 100)
            logger.info(f"Training completed! Best Average Reward: {best_reward:.2f}")
            logger.info(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
            logger.info(f"Checkpoints available at: {CHECKPOINT_DIR}")
            logger.info("=" * 100)
            
        except KeyboardInterrupt:
            logger.warning("\n" + "="*100)
            logger.warning("TRAINING INTERRUPTED BY USER (Ctrl+C)")
            logger.warning("="*100)
            self.save_final_model()
            logger.warning("Final model saved successfully")
            logger.warning("="*100 + "\n")
            raise
        except Exception as e:
            logger.error(f"\n[FAIL] Training error: {e}")
            logger.warning("Attempting to save model before exiting...")
            try:
                self.save_final_model()
                logger.info("Final model saved successfully")
            except Exception as save_err:
                logger.error(f"Could not save model: {save_err}")
            raise
    
    def save_final_model(self):
        """Save model on interruption or error"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = MODELS_DIR / f'trackmania_policy_final_{timestamp}.pt'
        torch.save(self.policy_net.state_dict(), model_path)
        logger.info(f"[OK] Final model saved: {model_path}")
        
        # Also save checkpoint
        checkpoint_path = self.checkpoint_mgr.save(
            self.policy_net,
            self.value_net,
            self.episode,
            np.mean(self.episode_rewards) if self.episode_rewards else 0,
            self.optimizer_policy,
            self.optimizer_value
        )
        logger.info(f"[OK] Final checkpoint saved: {checkpoint_path}")


def main():
    """Main entry point"""
    print("=" * 100)
    print("TRACKMANIA RL - TRAINING LAUNCHER")
    print("=" * 100)
    print()
    print("[OK] Python cache directory set to: F:\\aidata\\pycache")
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
        return
    
    print()
    print("=" * 100)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 100)
    print()
    
    # Initialize training environment (ViGEmBus check, resume state, etc.)
    if not initialize_training_environment():
        logger.error("\n[FAIL] Training environment initialization failed")
        logger.error("Please fix the errors above and try again")
        return
    
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
    
    # Import TMRL here to avoid issues if not installed
    try:
        from tmrl import get_environment
    except ImportError:
        logger.error("TMRL not installed. Install with: pip install tmrl")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Get TMRL environment (LIDAR-based)
    logger.info("Loading TMRL environment...")
    logger.info("(This may take a moment on first run or after driver restart)")
    
    try:
        env = get_environment()
        logger.info("[OK] TMRL environment loaded successfully")
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"[FAIL] Failed to load TMRL environment: {e}")
        
        # Check if it's a ViGEmBus/driver error
        if "vigem" in error_msg or "could not connect" in error_msg or "device" in error_msg:
            logger.error("\n" + "="*60)
            logger.error("ViGEmBus Driver Issue Detected")
            logger.error("="*60)
            logger.error("\nSOLUTION:")
            logger.error("1. Make sure Trackmania 2020 is running (in a race, not menu)")
            logger.error("2. If ViGEmBus was recently installed: RESTART YOUR PC")
            logger.error("3. Then run training again")
            logger.error("\nFor more help: See VIGEM_TROUBLESHOOTING.md")
        else:
            logger.error("\nMake sure:")
            logger.error("  1. OpenPlanet is installed (https://openplanet.nl/)")
            logger.error("  2. Trackmania 2020 is installed and running")
            logger.error("  3. You have run TMRL setup (tmrl-setup)")
            logger.error("  4. A race is running (not menu or replay)")
        return
    
    # Get environment observation/action spaces
    obs, info = env.reset()
    logger.info(f"Observation type: {type(obs)}")
    if isinstance(obs, tuple):
        logger.info(f"Observation components: {[o.shape if hasattr(o, 'shape') else len(o) for o in obs]}")
    else:
        logger.info(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
    
    # Create networks (state_dim = 19 for speed + 18 LIDAR or similar)
    state_dim = 19
    action_dim = 2  # steering, acceleration
    
    policy_net = PolicyNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    value_net = ValueNetwork(state_dim=state_dim, hidden_dim=256)
    
    logger.info(f"Policy Network: {state_dim} inputs -> {action_dim} outputs")
    logger.info(f"Value Network: {state_dim} inputs -> 1 output")
    
    # Create trainer
    trainer = PPOTrainerTMRL(
        env=env,
        policy_net=policy_net,
        value_net=value_net,
        device=device,
        lr_policy=3e-4,
        lr_value=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coeff=0.01,
        value_loss_coeff=0.5,
        max_grad_norm=0.5,
        batch_size=64,
        n_epochs=3,
        n_steps=2048,
        timeout_seconds=40,  # Reset model after 40 seconds of inactivity
    )
    
    # Train with proper exception handling
    try:
        trainer.train(num_iterations=1000)
    except KeyboardInterrupt:
        logger.warning("\n[INFO] Training interrupted by user (Ctrl+C)")
        logger.info("Model has been saved")
    except Exception as e:
        logger.error(f"[FAIL] Training error: {e}")
        logger.info("Attempting final model save...")
        try:
            trainer.save_final_model()
        except Exception as save_err:
            logger.error(f"Could not save model: {save_err}")
        raise
    finally:
        # Close environment
        try:
            env.close()
            logger.info("Environment closed")
        except Exception as e:
            logger.warning(f"Could not close environment: {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining stopped by user (Ctrl+C)")
        print("Checkpoints have been saved to: F:\\aidata\\trackmania_checkpoints\\")
        print("Logs saved to: F:\\aidata\\trackmania_logs\\")
        print("Models saved to: F:\\aidata\\trackmania_models\\")
        print("\nTo resume training later, run: python train_tmrl.py")
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
