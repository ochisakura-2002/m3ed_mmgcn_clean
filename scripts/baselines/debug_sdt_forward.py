"""Fake-forward smoke test for the project SDT baseline port."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baselines.sdt import SDTBaseline  # noqa: E402


def _fake_batch(
    batch_size: int,
    seq_len: int,
    text_dim: int,
    audio_dim: int,
    visual_dim: int,
    num_classes: int,
    num_speakers: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    text_features = torch.randn(batch_size, seq_len, text_dim, device=device)
    audio_features = torch.randn(batch_size, seq_len, audio_dim, device=device)
    visual_features = torch.randn(batch_size, seq_len, visual_dim, device=device)

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    if seq_len > 1:
        attention_mask[-1, -1] = False

    labels = torch.randint(0, num_classes, (batch_size, seq_len), device=device)
    labels = labels.masked_fill(~attention_mask, 0)

    speaker_ids_int = torch.randint(
        0,
        num_speakers,
        (batch_size, seq_len),
        device=device,
    )
    speaker_ids_int = speaker_ids_int.masked_fill(~attention_mask, 0)

    return (
        text_features,
        audio_features,
        visual_features,
        attention_mask,
        speaker_ids_int,
        labels,
    )


def _run_case(
    name: str,
    seq_len: int,
    text_dim: int,
    audio_dim: int,
    visual_dim: int,
    num_classes: int,
    num_speakers: int,
) -> None:
    batch_size = 2
    device = torch.device("cpu")
    batch = _fake_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        num_classes=num_classes,
        num_speakers=num_speakers,
        device=device,
    )
    (
        text_features,
        audio_features,
        visual_features,
        attention_mask,
        speaker_ids_int,
        labels,
    ) = batch

    model = SDTBaseline(
        text_dim=text_dim,
        audio_dim=audio_dim,
        visual_dim=visual_dim,
        hidden_dim=128,
        num_classes=num_classes,
        num_speakers=num_speakers,
        num_heads=4,
        dropout=0.1,
        temperature=1.0,
        use_self_distillation=True,
    ).to(device)

    output = model(
        text_features=text_features,
        audio_features=audio_features,
        visual_features=visual_features,
        attention_mask=attention_mask,
        speaker_ids_int=speaker_ids_int,
        labels=labels,
    )
    logits = output["logits"]
    expected_shape = (batch_size, seq_len, num_classes)
    if tuple(logits.shape) != expected_shape:
        raise AssertionError(
            f"{name} logits shape mismatch: expected {expected_shape}, "
            f"got {tuple(logits.shape)}"
        )

    loss = output["loss"]
    if loss is None:
        raise AssertionError(f"{name} loss should not be None when labels are provided")
    loss_finite = bool(torch.isfinite(loss).item())
    if not loss_finite:
        raise AssertionError(f"{name} loss is not finite: {loss.item()}")

    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    backward_ok = bool(gradients) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    if not backward_ok:
        raise AssertionError(f"{name} backward produced missing or non-finite gradients")

    print(f"[{name} fake] logits shape: {logits.shape}")
    print(f"[{name} fake] loss finite: {loss_finite}")
    print(f"[{name} fake] backward ok: {backward_ok}")


def main() -> None:
    torch.manual_seed(7)
    _run_case(
        name="M3ED",
        seq_len=4,
        text_dim=768,
        audio_dim=1024,
        visual_dim=342,
        num_classes=7,
        num_speakers=2,
    )
    _run_case(
        name="IEMOCAP",
        seq_len=5,
        text_dim=100,
        audio_dim=1582,
        visual_dim=342,
        num_classes=6,
        num_speakers=2,
    )


if __name__ == "__main__":
    main()
