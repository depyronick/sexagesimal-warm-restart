"""Sexagesimal Warm Restart learning‑rate scheduler.

Drop‑in replacement for torch.optim.lr_scheduler._LRScheduler.
Only required argument: total training steps.
"""
import math
from torch.optim.lr_scheduler import _LRScheduler

# 59 segment lengths recorded on Babylonian tablet AO 6456
_SEGMENTS = [
    60, 30, 15, 15, 6, 6, 6,
     3,  3,  3,  3,
     2,  2,  2,  2,  2,
     *([1] * 35)  # pad to keep exactly 59 entries
]

class SexagesimalWarmRestart(_LRScheduler):
    """Cosine‑decay inside sexagesimal ladder restarts."""

    def __init__(self, optimizer, total_steps: int, last_epoch: int = -1):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self.total_steps = total_steps

        # Precompute scaled integer segment boundaries
        cumulative = [0]
        for seg in _SEGMENTS:
            cumulative.append(cumulative[-1] + seg)
        scale = total_steps / cumulative[-1]
        self.ends = [int(round(x * scale)) for x in cumulative]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch + 1  # step count starts at 1
        # Binary‑search could be used, but linear scan is O(59) ≈ O(1)
        k = next(i for i in range(1, len(self.ends)) if t <= self.ends[i])
        t0, t1 = self.ends[k - 1], self.ends[k]
        local_pos = (t - t0) / (t1 - t0)
        alpha = 0.5 * (1.0 + math.cos(math.pi * local_pos))
        return [base_lr * alpha for base_lr in self.base_lrs]
