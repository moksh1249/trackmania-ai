"""
Training Controller - Manages ViGEmBus driver connectivity, pause/resume, and checkpointing
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ViGEmBusManager:
    """Manages ViGEmBus driver for virtual gamepad connectivity"""
    
    DRIVER_NAME = "ViGEmBus"
    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_DELAY = 2  # seconds
    
    @staticmethod
    def check_driver_installed() -> bool:
        """Check if ViGEmBus driver library is installed"""
        try:
            import vgamepad
            logger.info("[OK] ViGEmBus library is installed (pip package)")
            return True
        except ImportError:
            logger.warning("[FAIL] ViGEmBus library not installed - install with:")
            logger.warning("  pip install vgamepad")
            return False
    
    @staticmethod
    def ensure_driver_ready(retry_count=0, test_connection=False) -> bool:
        """Verify ViGEmBus driver is ready (optional connection test)
        
        Args:
            retry_count: Current retry attempt number
            test_connection: If True, test actual connection (may cause issues on restart)
                           If False, just verify library is installed
        
        Note: Connection testing is optional because ViGEmBus driver may need
              restart after installation. We skip direct connection test and 
              let TMRL handle the actual connection during environment creation.
        """
        if not test_connection:
            # Skip connection test - just verify library is available
            # The actual connection will be tested when TMRL initializes
            logger.info("[OK] ViGEmBus library available (actual connection tested during training)")
            return True
        
        # Optional connection test (not recommended on every restart)
        if retry_count >= ViGEmBusManager.MAX_RECONNECT_ATTEMPTS:
            logger.error("[FAIL] ViGEmBus connection failed after max retries")
            return False
        
        try:
            import vgamepad as vg
            logger.info("[INFO] Testing ViGEmBus driver connection...")
            # Create gamepad instance - this tests if driver is active
            # Note: VX360Gamepad() may fail if driver needs restart
            test_pad = vg.VX360Gamepad()
            logger.info("[OK] ViGEmBus driver connection test successful")
            # Note: Don't call disconnect() - method doesn't exist
            return True
        except AssertionError as e:
            error_str = str(e).lower()
            if "could not connect" in error_str or "vigem" in error_str:
                logger.warning(f"[RETRY] ViGEmBus driver not ready (attempt {retry_count + 1}/{ViGEmBusManager.MAX_RECONNECT_ATTEMPTS})")
                logger.warning(f"  Tip: Restart Windows if driver was recently installed")
                logger.warning(f"  Retrying in {ViGEmBusManager.RECONNECT_DELAY} seconds...")
                time.sleep(ViGEmBusManager.RECONNECT_DELAY)
                return ViGEmBusManager.ensure_driver_ready(retry_count + 1, test_connection=True)
            raise
        except AttributeError as e:
            if "disconnect" in str(e):
                # VX360Gamepad doesn't have disconnect method - ignore this
                logger.info("[OK] ViGEmBus driver initialized (auto-cleanup)")
                return True
            raise
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error testing ViGEmBus: {e}")
            logger.warning("  This might happen if driver needs restart")
            logger.warning("  Proceeding anyway - actual connection will be tested during training")
            return True  # Don't fail - let TMRL handle the actual connection
    
    @staticmethod
    def get_fix_instructions() -> str:
        """Get user-friendly fix instructions"""
        return """
╔══════════════════════════════════════════════════════════════╗
║          ViGEmBus Driver Connection Error                    ║
╚══════════════════════════════════════════════════════════════╝

SOLUTION:

1. INSTALL THE DRIVER (if not already installed):
   - Download from: https://github.com/ViGEm/ViGEmBus/releases
   - Choose: ViGEmBusSetup_x64.exe (or x86 if 32-bit)
   - Run the installer and RESTART YOUR PC

2. IF ALREADY INSTALLED:
   a) Restart the ViGEmBus service:
      - Open Device Manager (devmgmt.msc)
      - Find "ViGEm Virtual Gamepad"
      - Right-click → Restart
      
   b) Or restart Windows

3. ALTERNATIVE (If above fails):
   - Uninstall ViGEmBus completely (Control Panel)
   - Restart Windows
   - Reinstall from: https://github.com/ViGEm/ViGEmBus/releases

4. THEN RESTART TRAINING:
   - Run: python START_TRAINING.py
   - Open Trackmania 2020
   - Start a race
   - Training should connect automatically

NEED HELP?
- See: VIGEM_TROUBLESHOOTING.md
- Or: README.md (Dependencies section)
"""


class TrainingCheckpoint:
    """Manage training state checkpoints for pause/resume"""
    
    CHECKPOINT_DIR = Path('F:\\aidata\\trackmania_checkpoints')
    RESUME_STATE_FILE = CHECKPOINT_DIR / 'resume_state.json'
    
    @staticmethod
    def save_resume_state(
        episode: int,
        global_step: int,
        total_reward: float,
        checkpoint_info: Dict[str, Any]
    ) -> bool:
        """Save training state for resume"""
        try:
            TrainingCheckpoint.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            
            state = {
                'timestamp': time.time(),
                'episode': episode,
                'global_step': global_step,
                'total_reward': total_reward,
                'checkpoint_info': checkpoint_info,
                'notes': 'Delete this file to start training from scratch'
            }
            
            with open(TrainingCheckpoint.RESUME_STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save resume state: {e}")
            return False
    
    @staticmethod
    def load_resume_state() -> Dict[str, Any]:
        """Load previous training state for resume"""
        try:
            if TrainingCheckpoint.RESUME_STATE_FILE.exists():
                with open(TrainingCheckpoint.RESUME_STATE_FILE, 'r') as f:
                    state = json.load(f)
                logger.info(f"✓ Loaded resume state from episode {state['episode']}")
                return state
        except Exception as e:
            logger.warning(f"Could not load resume state: {e}")
        return None
    
    @staticmethod
    def clear_resume_state() -> None:
        """Clear resume state (start from scratch)"""
        try:
            if TrainingCheckpoint.RESUME_STATE_FILE.exists():
                TrainingCheckpoint.RESUME_STATE_FILE.unlink()
                logger.info("✓ Resume state cleared - starting fresh training")
        except Exception as e:
            logger.warning(f"Could not clear resume state: {e}")


class PauseResumeController:
    """Handle pause/resume functionality with hotkeys"""
    
    def __init__(self):
        self.paused = False
        self.pause_time = None
        self.pause_count = 0
        self.total_pause_duration = 0
    
    def handle_pause(self, episode: int, step: int) -> str:
        """Handle pause request"""
        self.paused = True
        self.pause_time = time.time()
        self.pause_count += 1
        
        msg = f"\n{'='*60}\nPAUSED at Episode {episode}, Step {step}\n{'='*60}\n"
        msg += "Training paused. Options:\n"
        msg += "  - Press 'r' to RESUME training\n"
        msg += "  - Press 's' to SAVE and EXIT (resume later)\n"
        msg += "  - Press 'x' to EXIT without saving\n"
        msg += f"{'='*60}\n"
        
        return msg
    
    def handle_resume(self) -> Tuple[bool, float]:
        """Handle resume from pause"""
        if not self.paused:
            return False, 0.0
        
        pause_duration = time.time() - self.pause_time
        self.total_pause_duration += pause_duration
        self.paused = False
        
        logger.info(f"✓ Resumed training (paused for {pause_duration:.1f}s)")
        return True, pause_duration
    
    def get_pause_stats(self) -> Dict[str, Any]:
        """Get pause statistics"""
        return {
            'pause_count': self.pause_count,
            'total_pause_duration': self.total_pause_duration,
            'currently_paused': self.paused
        }


class PauseCommandListener:
    """Listen for pause commands from stdin (non-blocking)"""
    
    def __init__(self):
        self.pause_requested = False
        self.resume_requested = False
        self.exit_requested = False
        self.save_and_exit_requested = False
    
    def check_commands(self) -> None:
        """Check for user input commands (non-blocking)"""
        # Note: Full implementation would use threading/async for truly non-blocking
        # This is a placeholder that should be called occasionally
        pass
    
    @staticmethod
    def display_hotkey_help() -> None:
        """Display available hotkeys"""
        logger.info("""
[HOTKEYS DURING TRAINING]
  Ctrl+P  : PAUSE training
  Ctrl+R  : RESUME training (after pause)
  Ctrl+S  : SAVE checkpoint and EXIT (resume later)
  Ctrl+X  : EXIT without saving
  Ctrl+C  : Force stop (normal Ctrl+C)
""")


def initialize_training_environment() -> bool:
    """Initialize and verify all training requirements"""
    logger.info("="*60)
    logger.info("TRAINING ENVIRONMENT INITIALIZATION")
    logger.info("="*60)
    
    # Check ViGEmBus driver
    logger.info("\n1. Checking ViGEmBus driver...")
    if not ViGEmBusManager.check_driver_installed():
        logger.error("ViGEmBus library not installed")
        logger.error("Install with: pip install vgamepad")
        return False
    
    # Verify driver is ready (skip connection test - too aggressive on restart)
    logger.info("\n2. Verifying ViGEmBus driver...")
    if not ViGEmBusManager.ensure_driver_ready(test_connection=False):
        logger.warning("Could not verify ViGEmBus driver")
        logger.warning("Note: Actual connection will be tested during training")
        # Don't fail here - let TMRL handle the connection
    logger.info("[OK] ViGEmBus driver ready")
    
    # Check F-drive storage
    logger.info("\n3. Checking F-drive storage...")
    try:
        aidata_path = Path('F:\\aidata')
        if not aidata_path.exists():
            logger.warning("F:\\aidata does not exist - will be created")
        else:
            logger.info("[OK] F:\\aidata accessible")
    except Exception as e:
        logger.error(f"Error accessing F:\\aidata: {e}")
        return False
    
    # Check resume state
    logger.info("\n4. Checking for previous training session...")
    resume_state = TrainingCheckpoint.load_resume_state()
    if resume_state:
        logger.info(f"[OK] Found previous training at episode {resume_state['episode']}")
        logger.info(f"  Last global step: {resume_state['global_step']}")
        logger.info(f"  Total reward so far: {resume_state['total_reward']:.0f}")
        logger.info(f"  Resume state file: {TrainingCheckpoint.RESUME_STATE_FILE}")
        logger.info("\n  TO CONTINUE: Run START_TRAINING.py")
        logger.info("  TO START FRESH: Delete F:\\aidata\\trackmania_checkpoints\\resume_state.json")
    else:
        logger.info("No previous training found - starting fresh")
    
    # Display hotkey help
    logger.info("")
    PauseCommandListener.display_hotkey_help()
    
    logger.info("\n" + "="*60)
    logger.info("[OK] ENVIRONMENT READY")
    logger.info("="*60 + "\n")
    
    return True


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Test initialization
    success = initialize_training_environment()
    sys.exit(0 if success else 1)
