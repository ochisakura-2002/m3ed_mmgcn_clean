"""Shared, strict helpers for IEMOCAP session-based splits."""

from __future__ import annotations

import re
from typing import Any


IEMOCAP_SESSION_IDS = ("Ses01", "Ses02", "Ses03", "Ses04", "Ses05")
IEMOCAP_TRAIN_SESSION_IDS = IEMOCAP_SESSION_IDS[:4]
IEMOCAP_TEST_SESSION_ID = IEMOCAP_SESSION_IDS[4]
_SESSION_PATTERN = re.compile(r"^(Ses0[1-5])[FM]_", flags=re.IGNORECASE)


def parse_iemocap_session_id(dialogue_id: Any) -> str:
    """Return ``Ses01`` ... ``Ses05`` from a canonical dialogue ID.

    The parser is intentionally anchored and raises on unexpected identifiers;
    session-holdout construction must never guess a split assignment.
    """

    text = str(dialogue_id).strip()
    match = _SESSION_PATTERN.match(text)
    if match is None:
        raise ValueError(
            "Unable to parse IEMOCAP session from dialogue_id="
            f"{text!r}; expected an ID beginning with 'Ses0[1-5]F_' or "
            "'Ses0[1-5]M_'."
        )
    return match.group(1).capitalize()
