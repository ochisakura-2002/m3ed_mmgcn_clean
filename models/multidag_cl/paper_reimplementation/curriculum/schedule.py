"""Pure one-bucket-per-global-epoch visibility schedule."""

from __future__ import annotations

from ..config import CurriculumScheduleProfile


class CurriculumSchedule:
    """Map a one-based global epoch to a safely saturated bucket prefix."""

    def __init__(
        self,
        profile: CurriculumScheduleProfile,
        *,
        enabled: bool = True,
    ) -> None:
        try:
            self.profile = (
                profile
                if isinstance(profile, CurriculumScheduleProfile)
                else CurriculumScheduleProfile(profile)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown curriculum schedule profile: {profile!r}") from error
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be bool")
        self.enabled = enabled

    def visible_bucket_count(
        self,
        global_epoch: int,
        actual_bucket_count: int,
    ) -> int:
        if isinstance(global_epoch, bool) or not isinstance(global_epoch, int):
            raise TypeError("global_epoch must be int")
        if isinstance(actual_bucket_count, bool) or not isinstance(actual_bucket_count, int):
            raise TypeError("actual_bucket_count must be int")
        if global_epoch < 1:
            raise ValueError("global_epoch is one-based and must be positive")
        if actual_bucket_count < 1:
            raise ValueError("actual_bucket_count must be positive")
        if not self.enabled:
            return actual_bucket_count
        return min(global_epoch, actual_bucket_count)


__all__ = ["CurriculumSchedule"]
