"""DEV AC early stopping, independent from highest-score best checkpoint selection."""
from dataclasses import dataclass, asdict
import math


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0  # F1 on [0, 1], not percentage points
    reference: float | None = None
    bad_epochs: int = 0
    stopped: bool = False
    min_epochs: int = 0
    epochs_seen: int = 0

    def __post_init__(self):
        if self.patience < 1 or not math.isfinite(self.min_delta) or self.min_delta < 0 or self.min_epochs < 0 or self.epochs_seen < 0:
            raise ValueError('Require positive patience and finite nonnegative min_delta')

    def update(self, score, active=True):
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('DEV AC must be finite and in [0, 1]')
        if not active or self.stopped:
            return self.stopped
        self.epochs_seen += 1
        if self.reference is None or score > self.reference + self.min_delta:
            self.reference = score
            self.bad_epochs = 0
        elif self.epochs_seen > self.min_epochs:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return self.stopped

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        restored = EarlyStopping(**state)
        if (restored.patience, restored.min_delta, restored.min_epochs) != (self.patience, self.min_delta, self.min_epochs):
            raise ValueError('Early stopping configuration changed')
        self.reference, self.bad_epochs, self.stopped = restored.reference, restored.bad_epochs, restored.stopped
        self.epochs_seen = restored.epochs_seen
