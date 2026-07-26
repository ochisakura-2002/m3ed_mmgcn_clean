"""Shared validation-only early-stopping state for training workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Result of observing one validation metric."""

    improved: bool
    should_stop: bool
    counter: int
    best_epoch: int
    best_metric: float


class EarlyStoppingController:
    """Track a maximized validation metric with an optional stop floor.

    ``min_epochs`` only gates the stop decision. Best-metric updates remain
    active from epoch one, so a best checkpoint may be saved before the floor.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_epochs: int = 0,
        min_delta: float = 0.0,
        best_metric: float = float("-inf"),
        best_epoch: int = -1,
        counter: int = 0,
    ) -> None:
        if int(patience) < 0:
            raise ValueError("patience must be non-negative")
        if int(min_epochs) < 0:
            raise ValueError("min_epochs must be non-negative")
        if not math.isfinite(float(min_delta)) or float(min_delta) < 0:
            raise ValueError("min_delta must be finite and non-negative")
        if int(counter) < 0:
            raise ValueError("counter must be non-negative")
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.min_delta = float(min_delta)
        self.best_metric = float(best_metric)
        self.best_epoch = int(best_epoch)
        self.counter = int(counter)

    def update(self, epoch: int, metric: float) -> EarlyStoppingDecision:
        epoch = int(epoch)
        metric = float(metric)
        if epoch <= 0:
            raise ValueError("epoch must be positive")
        if not math.isfinite(metric):
            raise ValueError("early-stopping metric must be finite")

        improved = metric > self.best_metric + self.min_delta
        if improved:
            self.best_metric = metric
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        should_stop = (
            self.patience > 0
            and epoch >= self.min_epochs
            and self.counter >= self.patience
        )
        return EarlyStoppingDecision(
            improved=improved,
            should_stop=should_stop,
            counter=self.counter,
            best_epoch=self.best_epoch,
            best_metric=self.best_metric,
        )


__all__ = [
    "EarlyStoppingController",
    "EarlyStoppingDecision",
]
