# src/lpcvc_retrieval/logger.py
"""
Optional WandB Logger Wrapper.

How to remove:
1. Delete this file
2. In train.py: remove `from .logger import TrainLogger` line
3. In train.py: remove all `logger.xxx()` calls
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class TrainLogger:
    """
    Unified logger that supports both print() and WandB.
    WandB is optional - if use_wandb=False, only print() is used.
    
    Usage:
        logger = TrainLogger(use_wandb=True, project="my-project")
        logger.log({"loss": 0.5, "lr": 1e-4})
        logger.finish()
    """
    
    def __init__(
        self,
        use_wandb: bool = False,
        project: str = "lpcvc-clip-lite",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.use_wandb = use_wandb
        self._wandb = None
        
        if use_wandb:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(
                    project=project,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                print(f"[WandB] Initialized: project={project}, run={wandb.run.name}")
            except ImportError:
                print("[WandB] wandb not installed. pip install wandb")
                self.use_wandb = False
            except Exception as e:
                print(f"[WandB] Failed to initialize: {e}")
                self.use_wandb = False
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB (if enabled). Print is handled separately in train.py."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Log epoch-level metrics."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.log({"epoch": epoch, **metrics})
    
    def finish(self) -> None:
        """Finish the WandB run."""
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()
            print("[WandB] Run finished.")
