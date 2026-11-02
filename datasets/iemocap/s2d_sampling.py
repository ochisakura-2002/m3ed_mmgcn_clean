"""Model-independent, deterministic sampling over the complete utterance clip."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class Uniform16Sampler:
    num_frames: int = 16

    def __post_init__(self):
        if self.num_frames != 16:
            raise ValueError("S2D-V0 freezes num_frames=16")

    def __call__(self, frame_count: int) -> tuple[int, ...]:
        if isinstance(frame_count, bool) or not isinstance(frame_count, Integral) or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        # Integer floor of linspace(0, N-1, 16), including both endpoints.
        # Repeated positions are intentional for clips shorter than 16 frames.
        return tuple(i * (int(frame_count) - 1) // 15 for i in range(16))
