"""
Demo Player - MVP Training Simulator with Live Trackmania Control
Mimics the real training process with fake models
Loads saved demonstrations and plays them in Trackmania while simulating realistic training output
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from collections import deque
import logging
from datetime import datetime
import random
import models
import train_tmrl

# Try to import keyboard controller for output
try:
    from pynput.keyboard import Controller, Key
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

# Setup logging
log_dir = Path('F:\\aidata\\trackmania_logs')
log_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = log_dir / f'demo_player_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import TMRL for live control (fallback)
try:
    from tmrl import get_environment
    TMRL_AVAILABLE = True
except ImportError:
    TMRL_AVAILABLE = False
    logger.warning("TMRL not available - running in keyboard mode only")


class DemoPlayer:
    """
    Plays back recorded demonstrations as fake training iterations
    Simultaneously sends real control commands to Trackmania
    Simulates the real training output with realistic metrics
    """
    
    def __init__(self, use_live_trackmania=True):
        """Initialize player"""
        self.demo_dir = Path('F:\\aidata\\trackmania_demos')
        self.models_dir = Path('F:\\aidata\\trackmania_models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.available_demos = []
        self.current_demo = None
        
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.global_step = 0
        self.episode = 0
        self.total_reward = 0.0
        self.total_loss = 0.0
        
        # Keyboard controller for output
        self.keyboard_controller = None
        self.use_keyboard = False
        if KEYBOARD_AVAILABLE:
            try:
                self.keyboard_controller = Controller()
                self.use_keyboard = True
                logger.info("[OK] Keyboard controller initialized for input play")
            except Exception as e:
                logger.warning(f"Could not initialize keyboard controller: {e}")
        
        # Trackmania environment (TMRL fallback)
        self.env = None
        self.use_live_trackmania = False
        
        if use_live_trackmania and TMRL_AVAILABLE and not self.use_keyboard:
            try:
                logger.info("Initializing TMRL environment for live control...")
                self.env = get_environment()
                self.env.reset()
                logger.info("[OK] TMRL environment ready for live control")
                self.use_live_trackmania = True
            except Exception as e:
                logger.warning(f"Could not initialize TMRL environment: {e}")
        
        logger.info("DemoPlayer initialized")
        logger.info(f"Demo directory: {self.demo_dir}")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Input method: {'Keyboard Controller' if self.use_keyboard else 'TMRL' if self.use_live_trackmania else 'DISABLED'}")
    
    def send_action_to_trackmania(self, steering, acceleration):
        """
        Send control action to Trackmania via keyboard or TMRL (legacy frame-based method)
        
        Args:
            steering: Value [-1, 1] where -1=left, 1=right
            acceleration: Value [-1, 1] where -1=brake, 1=gas
            
        Returns:
            bool: True if successful
        """
        # Try keyboard controller first (primary method)
        if self.use_keyboard and self.keyboard_controller:
            try:
                # Send keyboard inputs based on steering and acceleration
                # W = gas, S = brake, A = left, D = right
                
                # Handle steering
                if steering > 0.5:  # Right
                    self.keyboard_controller.press('d')
                    time.sleep(0.02)
                    self.keyboard_controller.release('d')
                elif steering < -0.5:  # Left
                    self.keyboard_controller.press('a')
                    time.sleep(0.02)
                    self.keyboard_controller.release('a')
                
                # Handle acceleration
                if acceleration > 0.5:  # Gas
                    self.keyboard_controller.press('w')
                    time.sleep(0.02)
                    self.keyboard_controller.release('w')
                elif acceleration < -0.5:  # Brake
                    self.keyboard_controller.press('s')
                    time.sleep(0.02)
                    self.keyboard_controller.release('s')
                
                return True
            except Exception as e:
                logger.debug(f"Keyboard controller error: {e}")
                return False
        
        # Fallback to TMRL if keyboard not available
        if not self.env:
            return False
        
        try:
            # Convert our action format to TMRL format
            gas = max(0, acceleration)        # [0, 1]: 0=no gas, 1=full gas
            brake = max(0, -acceleration)     # [0, 1]: 0=no brake, 1=full brake
            
            # TMRL expects: [gas, brake, steer]
            tmrl_action = np.array([gas, brake, steering], dtype=np.float32)
            
            # Send to environment
            try:
                obs, reward, terminated, truncated, info = self.env.step(tmrl_action)
            except Exception as e:
                # If episode is terminated/truncated, reset and try again
                if "terminated" in str(e).lower() or "truncated" in str(e).lower():
                    logger.debug(f"Episode ended, resetting: {e}")
                    try:
                        self.env.reset()
                        obs, reward, terminated, truncated, info = self.env.step(tmrl_action)
                    except Exception as reset_error:
                        logger.warning(f"Could not recover from episode end: {reset_error}")
                        return False
                else:
                    raise
            
            return True
        except Exception as e:
            logger.debug(f"Error sending action to Trackmania: {e}")
            return False
    
    def replay_demo_in_trackmania(self, demo_frames, playback_speed=1.0, show_progress=True, show_frame_data=False):

        if not self.use_keyboard and not self.use_live_trackmania:
            logger.info("Skipping Trackmania play (not available)")
            return {'frames_sent': 0, 'success': False}
        
        # Get key events from current demo
        key_events = self.current_demo.get('key_events', [])
        
        if not key_events:
            logger.info("No key events in demo, falling back to frame-based play")
            demo_frames = self.current_demo.get('frames', [])
            if not demo_frames:
                return {'frames_sent': 0, 'total_frames': 0, 'success': False}
        
        logger.info(f"Starting Trackmania play: {len(key_events)} key events at {playback_speed}x speed")
        
        keys_sent = 0
        last_line_length = 0
        
        try:
            # For keyboard controller, reset is immediate
            if self.use_keyboard:
                print("  [OK] Keyboard controller ready!")
            else:
                # Reset environment before starting play (TMRL)
                try:
                    obs, info = self.env.reset()
                    logger.info("Environment reset before play")
                except Exception as e:
                    logger.warning(f"Could not reset environment: {e}")
                    return {'frames_sent': 0, 'total_frames': len(key_events), 'success': False}
            
            print("  [OK] Starting input stream!\n")
            logger.info("play starting - sending key events")
            
            # Record when playback actually starts
            playback_start_time = time.time()
            
            # Process each key event
            for event_idx, event in enumerate(key_events):
                event_key = event.get('key')
                event_type = event.get('event')
                event_time = event.get('time', 0.0)
                
                # Calculate when to send this event based on playback speed
                target_time = playback_start_time + (event_time / playback_speed)
                current_time = time.time()
                
                # Wait until the exact time this event should be sent (with high precision)
                if target_time > current_time:
                    sleep_duration = target_time - current_time
                    # High precision sleep
                    if sleep_duration > 0.001:
                        time.sleep(sleep_duration - 0.001)  # Sleep a bit less
                    # Busy-wait for the final millisecond for precision
                    while time.time() < target_time:
                        pass
                
                # Send the key event
                if self.use_keyboard and self.keyboard_controller:
                    try:
                        if event_type == 'press':
                            self.keyboard_controller.press(event_key.lower())
                            logger.debug(f"Key pressed: {event_key} at {event_time:.3f}s")
                        elif event_type == 'release':
                            self.keyboard_controller.release(event_key.lower())
                            logger.debug(f"Key released: {event_key} at {event_time:.3f}s")
                        
                        keys_sent += 1
                    except Exception as e:
                        logger.debug(f"Error sending key event: {e}")
                
                # Display event data if requested
                if show_frame_data and event_idx % max(1, len(key_events) // 20) == 0:
                    try:
                        progress = (event_idx / len(key_events)) * 100
                        
                        # Build display line
                        display_line = (
                            f"  {progress:5.1f}% | "
                            f"Event {event_idx:4d}/{len(key_events)} | "
                            f"Time: {event_time:7.3f}s | "
                            f"{event_type.upper()}: {event_key}"
                        )
                        
                        # Clear previous line and print new one
                        sys.stdout.write("\r" + display_line.ljust(last_line_length))
                        sys.stdout.flush()
                        last_line_length = len(display_line)
                    except Exception as e:
                        logger.debug(f"Display error: {e}")
            
            # Final newline to move past the progress display
            print("\n")
            if show_progress:
                print(f"  [OK] Trackmania play complete: {keys_sent} key events sent")
            
            logger.info(f"Trackmania play complete: {keys_sent} key events sent successfully")
            
            return {
                'frames_sent': keys_sent,
                'total_frames': len(key_events),
                'success': keys_sent > 0
            }
        
        except KeyboardInterrupt:
            logger.warning("Trackmania play interrupted by user")
            print("\n  [INTERRUPTED] play stopped by user")
            return {'frames_sent': keys_sent, 'total_frames': len(key_events), 'success': False}
        except Exception as e:
            logger.error(f"Error during play: {e}")
            import traceback
            traceback.print_exc()
            return {'frames_sent': keys_sent, 'total_frames': len(key_events), 'success': False}
        finally:
            # Reset environment after play (TMRL only)
            try:
                if self.env:
                    self.env.reset()
            except Exception as e:
                logger.debug(f"Could not reset after play: {e}")
    
    
    def find_demos(self):
        """Find all available demonstrations"""
        if not self.demo_dir.exists():
            logger.warning(f"Demo directory does not exist: {self.demo_dir}")
            return []
        
        demos = list(self.demo_dir.glob('*.json'))
        self.available_demos = demos
        
        logger.info(f"Found {len(demos)} demonstrations")
        for demo in demos:
            logger.info(f"  - {demo.name}")
        
        return demos
    
    def load_demo(self, demo_path):
        """Load a demonstration file"""
        try:
            with open(demo_path, 'r') as f:
                self.current_demo = json.load(f)
            
            logger.info(f"Loaded demo: {demo_path.name}")
            logger.info(f"  Frames: {len(self.current_demo.get('frames', []))}")
            logger.info(f"  Duration: {self.current_demo.get('duration', 0):.2f}s")
            logger.info(f"  Total Reward: {self.current_demo.get('total_reward', 0):.2f}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load demo: {e}")
            return False
    
    def generate_fake_model(self, demo_name):
        """Generate a fake model from demo"""
        model_name = f"demo_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        model_path = self.models_dir / model_name
        
        model_data = {
            'source': 'human_demo',
            'demo': demo_name,
            'created': datetime.now().isoformat(),
            'architecture': {
                'type': 'PolicyNetwork (from demo)',
                'input_dim': 19,
                'hidden_dim': 256,
                'output_dim': 2
            },
            'metadata': {}
        }
        
        if self.current_demo:
            model_data['metadata'] = {
                'frames_from_demo': len(self.current_demo.get('frames', [])),
                'demo_reward': self.current_demo.get('total_reward', 0),
            }
        
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        logger.info(f"Generated fake model: {model_name}")
        return model_path
    
    def simulate_training_iteration(self, demo_frames):
        """
        Simulate a single training iteration
        Uses frames from demo to generate realistic output
        """
        # Simulate collecting experience
        states_collected = len(demo_frames)
        
        # Simulate computing returns/advantages
        time.sleep(0.1)  # Fake computation time
        
        # Simulate training epoch
        num_batches = states_collected // 64
        
        # Fake losses with realistic variation
        policy_loss = random.uniform(0.08, 0.25) * random.choice([1, 1, 1, 0.5, 2])
        value_loss = random.uniform(0.15, 0.45) * random.choice([1, 0.8, 1.2])
        entropy = random.uniform(0.01, 0.08)
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'states_collected': states_collected,
            'batches': num_batches
        }
    
    def simulate_episode(self, demo_frames):
        """Simulate an episode using demo frames"""
        episode_reward = 0.0
        checkpoint_reached = random.choice([True] * 3 + [False] * 2)  # 60% checkpoint rate
        track_completed = random.choice([True] * 1 + [False] * 10)    # 10% completion rate
        
        for frame in demo_frames:
            episode_reward += frame.get('reward', 0)
        
        # Add some randomness
        episode_reward *= random.uniform(0.8, 1.3)
        episode_length = len(demo_frames)
        
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.global_step += episode_length
        self.total_reward += episode_reward
        self.episode += 1
        
        return {
            'episode': self.episode - 1,
            'reward': episode_reward,
            'length': episode_length,
            'checkpoint_reached': checkpoint_reached,
            'track_completed': track_completed
        }
    
    def play_demo(self, num_iterations=3, playback_speed=1.0):
        """
        Play demonstration by playing exact recorded inputs to Trackmania
        
        Args:
            num_iterations: Number of times to play the demo
            playback_speed: Speed multiplier (1.0 = real-time, 2.0 = 2x faster)
        """
        if not self.current_demo:
            logger.error("No demo loaded")
            return
        
        demo_frames = self.current_demo.get('frames', [])
        if not demo_frames:
            logger.error("Demo has no frames")
            return
        
        demo_name = self.current_demo.get('timestamp', 'unknown')
        demo_duration = self.current_demo.get('duration', 0)
        
        print("\n" + "="*120)
        print("TRACKMANIA - EXACT INPUT play")
        print("="*120)
        print(f"Demo: {demo_name}")
        print(f"Frames to play: {len(demo_frames)} (Duration: {demo_duration:.1f}s)")
        print(f"Playback Speed: {playback_speed}x")
        print(f"plays: {num_iterations}")
        if self.use_keyboard:
            print(f"[OK] Input Method: KEYBOARD CONTROLLER")
        elif self.use_live_trackmania:
            print(f"[OK] Input Method: TMRL")
        else:
            print(f"[NO] Input Method: DISABLED")
        print("="*120 + "\n")
        
        logger.info(f"Starting exact input play: {num_iterations}x, {len(demo_frames)} frames per play")
        
        total_frames_sent = 0
        total_time = 0
        
        try:
            for replay_num in range(num_iterations):
                replay_start = time.time()
                
                print(f"\n[play {replay_num + 1}/{num_iterations}]")
                print("-" * 120)
                print(f"Frame-by-frame input display (showing exact recorded values):")
                print()
                
                if self.use_keyboard or self.use_live_trackmania:
                    replay_stats = self.replay_demo_in_trackmania(demo_frames, playback_speed=playback_speed, show_frame_data=True)
                    frames_sent = replay_stats.get('frames_sent', 0)
                    total_frames = replay_stats.get('total_frames', 0)
                    total_frames_sent += frames_sent
                    
                    # Show result
                    print()
                    print(f"Frames sent to car: {frames_sent}/{total_frames}")
                    print(f"Status: {'SUCCESS - Car should have moved!' if frames_sent > 0 else 'FAILED'}")
                else:
                    print("[SKIPPED] Input method not available (keyboard controller or TMRL not initialized)")
                    print()
                    # Still show the frame data even without input method
                    for i, frame in enumerate(demo_frames):
                        steering = frame.get('steering', 0.0)
                        acceleration = frame.get('acceleration', 0.0)
                        steer_bar = self._make_bar(steering)
                        accel_bar = self._make_bar(acceleration)
                        
                        if i % max(1, len(demo_frames) // 10) == 0:  # Show every 10%
                            print(
                                f"  Frame {i:4d}/{len(demo_frames)} | "
                                f"Steer: {steer_bar} {steering:+.1f} | "
                                f"Accel: {accel_bar} {acceleration:+.1f}"
                            )
                
                replay_time = time.time() - replay_start
                total_time += replay_time
                actual_duration = demo_duration / playback_speed
                
                print()
                print(f"play Duration: {replay_time:.1f}s (expected: {actual_duration:.1f}s)")
                print("-" * 120)
            
            # Final summary
            print("\n" + "="*120)
            print("play COMPLETE")
            print("="*120)
            print(f"Total plays: {num_iterations}")
            print(f"Total Frames Sent: {total_frames_sent}")
            print(f"Total Time: {total_time:.1f}s")
            if self.use_keyboard or self.use_live_trackmania:
                print(f"[OK]")
            print("="*120 + "\n")
            
            logger.info(f"play complete: {num_iterations}x, {total_frames_sent} total frames sent")
        
        except KeyboardInterrupt:
            print("\n\n[OK] play stopped by user")
            logger.warning("play stopped by user (Ctrl+C)")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            logger.error(f"Playback error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Cleanup
            if self.env:
                try:
                    self.env.close()
                    logger.info("TMRL environment closed")
                except Exception as e:
                    logger.warning(f"Could not close environment: {e}")
    
    @staticmethod
    def _make_bar(value, width=30):
        """Create a visual bar"""
        position = int((value + 1) / 2 * width)
        position = max(0, min(width - 1, position))
        bar = ['░'] * width
        bar[position] = '█'
        return ''.join(bar)


def main():
    """Main entry point"""
    print("="*100)
    print("TRACKMANIA RL - MVP DEMO PLAYER (WITH LIVE TRACKMANIA CONTROL)")
    print("="*100)
    print()
    print("This tool plays recorded human demonstrations in Trackmania")
    print("while simulating the training process in real-time!")
    print()
    print("The car will move in-game based on your recorded actions")
    print("while the terminal displays training metrics and progress.")
    print()
    
    # Check if Trackmania can be controlled
    if KEYBOARD_AVAILABLE:
        print("[OK] Keyboard controller available - will be used for input play")
    elif TMRL_AVAILABLE:
        print("[OK] TMRL is available - Live Trackmania control will be enabled")
    else:
        print("[NO] No input method available - Running in simulation-only mode")
        print("    (Install pynput for keyboard control or TMRL for TMRL mode)")
    
    print()
    
    player = DemoPlayer(use_live_trackmania=True)
    
    # Find available demos
    demos = player.find_demos()
    
    if not demos:
        print("[WARNING] No demonstrations found!")
        print(f"Please record a demonstration first using: python record_human_demo.py")
        print(f"Demos should be saved to: {player.demo_dir}")
        return
    
    # Show available demos
    print("\nAvailable Demonstrations:")
    for i, demo in enumerate(demos, 1):
        print(f"  [{i}] {demo.name}")
    
    # Select demo
    print()
    try:
        choice = 'r'
        
        if choice == 'r':
            selected_demo = random.choice(demos)
            print(f"[OK] Random selection: {selected_demo.name}")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                selected_demo = demos[idx]
            else:
                print("[ERROR] Invalid selection")
                return
    except Exception as e:
        print(f"[ERROR] {e}")
        return
    
    # Load and play
    if player.load_demo(selected_demo):
        print()
        print("IMPORTANT:")
        print("  1. Make sure Trackmania 2020 is running")
        print("  2. Start an ACTUAL RACE (not play, not menu)")
        print("  3. Keep Trackmania window visible in background")
        print()
        
        try:
            iterations = 1
            iterations = int(iterations) if iterations else 1
        except ValueError:
            iterations = 1
        
        print()
        player.play_demo(num_iterations=iterations, playback_speed=1.0)
    else:
        print("[ERROR] Failed to load demo")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[OK] Demo player stopped")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
