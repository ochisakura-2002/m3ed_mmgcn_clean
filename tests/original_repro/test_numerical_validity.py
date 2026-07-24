from __future__ import annotations

import copy
import math
from pathlib import Path
from uuid import uuid4

import pytest
import torch
from torch.nn import functional as F

from models.registry.paper_aligned import build_original_repro_model
from models.gsmcc.project_variant.full_context.model import (
    angular_similarity,
    build_sliding_multimodal_graph,
)
from scripts.baselines.original_merc_runtime import (
    NUMERIC_STATUS_GRADIENT,
    NUMERIC_STATUS_PARAMETER,
    NumericValidationError,
    build_dataloader,
    build_optimizer,
    checkpoint_numeric_summary,
    forward_batch,
    load_checkpoint,
    load_yaml_config,
    move_batch,
    normalized_training_config,
    resolve_path,
    seed_everything,
    validate_model_output_finite,
    validate_named_tensors_finite,
    validate_runtime_config,
    verify_feature_sha256,
)
import scripts.baselines.train_original_merc_baseline as train_module
from scripts.diagnostics.models.gsmcc.diagnose_gsmcc_numerics import diagnose_mode


def test_closed_interval_acos_reproduces_nonfinite_boundary_gradient() -> None:
    vector = torch.tensor([[1.0, 0.0]], requires_grad=True)
    cosine = F.cosine_similarity(vector, vector, dim=-1).clamp(-1.0, 1.0)
    old_similarity = 1.0 - torch.acos(cosine) / math.pi
    assert torch.isfinite(old_similarity).all()
    old_similarity.sum().backward()
    assert not torch.isfinite(vector.grad).all()


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 2.0], [1.0, 2.0], None),
        ([1.0, 2.0], [1.0 + 1e-6, 2.0 - 1e-6], None),
        ([1.0, 0.0], [-1.0, 0.0], None),
        ([1.0, 0.0], [0.0, 1.0], 0.5),
        ([0.0, 0.0], [0.0, 0.0], 0.5),
    ],
)
def test_angular_similarity_forward_and_backward_are_finite(
    left: list[float], right: list[float], expected: float | None
) -> None:
    left_tensor = torch.tensor([left], dtype=torch.float32, requires_grad=True)
    right_tensor = torch.tensor([right], dtype=torch.float32, requires_grad=True)
    similarity = angular_similarity(left_tensor, right_tensor, eps=1e-7)
    assert torch.isfinite(similarity).all()
    assert bool(((similarity >= 0) & (similarity <= 1)).all())
    if expected is not None:
        torch.testing.assert_close(similarity, torch.tensor([expected]), atol=1e-6, rtol=0)
    similarity.sum().backward()
    assert torch.isfinite(left_tensor.grad).all()
    assert torch.isfinite(right_tensor.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_angular_similarity_cuda_float32_boundary_gradient_is_finite() -> None:
    vector = torch.randn(8, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    similarity = angular_similarity(vector, vector)
    similarity.sum().backward()
    assert torch.isfinite(similarity).all()
    assert torch.isfinite(vector.grad).all()


def test_graph_self_loops_are_fixed_and_normalization_gradients_are_finite() -> None:
    torch.manual_seed(9)
    modalities = tuple(torch.randn(1, 3, 4, requires_grad=True) for _ in range(3))
    mask = torch.tensor([[1, 1, 0]])
    adjacency, node_mask, diagnostics = build_sliding_multimodal_graph(
        modalities,
        mask,
        window=1,
        angular_eps=1e-7,
        return_diagnostics=True,
    )
    valid_diagonal = diagnostics["raw_adjacency"].diagonal(dim1=1, dim2=2)[node_mask.bool()]
    torch.testing.assert_close(valid_diagonal, torch.ones_like(valid_diagonal))
    assert torch.isfinite(diagnostics["degree"][node_mask.bool()]).all()
    assert bool((diagnostics["degree"][node_mask.bool()] > 0).all())
    assert torch.isfinite(diagnostics["inv_sqrt_degree"]).all()
    assert torch.isfinite(adjacency).all()
    adjacency.sum().backward()
    assert all(torch.isfinite(value.grad).all() for value in modalities)


def _gsmcc_config(use_contrastive_loss: bool) -> dict:
    return {
        "model": {
            "name": "project_paper_oriented_gsmcc",
            "text_feature_dim": 8,
            "audio_feature_dim": 6,
            "visual_feature_dim": 5,
            "num_classes": 3,
            "hidden_dim": 8,
            "num_speakers": 2,
            "graph_layers": 1,
            "window": 1,
            "dropout": 0.0,
            "angular_similarity_eps": 1e-7,
            "use_contrastive_loss": use_contrastive_loss,
        },
        "optimizer": {"name": "Adam", "learning_rate": 1e-3, "weight_decay": 0.0},
    }


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    return {
        "text_features": torch.randn(2, 4, 8),
        "audio_features": torch.randn(2, 4, 6),
        "visual_features": torch.randn(2, 4, 5),
        "attention_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]),
        "lengths": torch.tensor([4, 3]),
        "speaker_ids_int": torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]]),
        "labels": torch.tensor([[0, 1, 2, 1], [2, 0, 1, -100]]),
    }


@pytest.mark.parametrize("contrastive_enabled", [True, False])
def test_gsmcc_optimizer_step_is_finite_with_contrastive_on_and_off(
    contrastive_enabled: bool,
) -> None:
    config = _gsmcc_config(contrastive_enabled)
    model = build_original_repro_model(config)
    optimizer = build_optimizer(model, config)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(**_batch())
        assert torch.isfinite(output["logits"]).all()
        assert torch.isfinite(output["loss"]).all()
        output["loss"].backward()
        assert all(
            torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def _finite_output(loss: torch.Tensor) -> dict:
    return {
        "logits": torch.zeros(1, 1, 2),
        "classification_loss": loss,
        "aux_losses": {},
        "loss": loss,
        "features": {},
        "diagnostics": {},
    }


def test_nan_loss_fails_immediately_with_complete_context() -> None:
    output = _finite_output(torch.tensor(float("nan")))
    with pytest.raises(NumericValidationError) as captured:
        validate_model_output_finite(
            output,
            model_name="project_paper_oriented_gsmcc",
            epoch=3,
            batch_index=2,
            learning_rate=1e-5,
            amp_enabled=False,
        )
    message = str(captured.value)
    for field in (
        "model_name",
        "epoch",
        "batch_index",
        "stage",
        "tensor_or_parameter",
        "classification_loss",
        "auxiliary_losses",
        "total_loss",
        "learning_rate",
        "amp_enabled",
    ):
        assert f"{field}=" in message


@pytest.mark.parametrize(
    ("status", "stage"),
    [
        (NUMERIC_STATUS_GRADIENT, "gradients_after_backward"),
        (NUMERIC_STATUS_PARAMETER, "parameters_after_optimizer_step"),
    ],
)
def test_nonfinite_gradient_or_parameter_fails_immediately(status: str, stage: str) -> None:
    output = _finite_output(torch.tensor(1.0))
    with pytest.raises(NumericValidationError) as captured:
        validate_named_tensors_finite(
            [("bad", torch.tensor(float("nan")))],
            numeric_status=status,
            stage=stage,
            model_name="project_paper_oriented_gsmcc",
            epoch=1,
            batch_index=1,
            output=output,
            learning_rate=1e-5,
            amp_enabled=False,
        )
    assert captured.value.numeric_status == status
    assert captured.value.stage == stage


class _FiniteForwardNaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


class _GateFixtureModel(torch.nn.Module):
    def __init__(self, failure: str) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.failure = failure

    def forward(self, text_features, **_kwargs):
        logits = self.weight * torch.ones(
            (*text_features.shape[:2], 2), dtype=text_features.dtype
        )
        if self.failure == "loss":
            loss = self.weight * torch.tensor(float("nan"))
        elif self.failure == "gradient":
            loss = _FiniteForwardNaNBackward.apply(self.weight)
        else:
            loss = self.weight.square()
        return {
            "logits": logits,
            "classification_loss": loss,
            "aux_losses": {},
            "loss": loss,
            "features": {},
            "diagnostics": {},
        }


class _NaNAfterStepOptimizer(torch.optim.SGD):
    def step(self, closure=None):
        result = super().step(closure)
        with torch.no_grad():
            for group in self.param_groups:
                for parameter in group["params"]:
                    parameter.fill_(float("nan"))
        return result


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        ("loss", "NONFINITE_LOSS"),
        ("gradient", "NONFINITE_GRADIENT"),
        ("parameter", "NONFINITE_PARAMETER"),
    ],
)
def test_common_training_loop_fails_fast_at_first_nonfinite_stage(
    failure: str,
    expected_status: str,
) -> None:
    model = _GateFixtureModel(failure)
    optimizer = (
        _NaNAfterStepOptimizer(model.parameters(), lr=1e-3)
        if failure == "parameter"
        else torch.optim.SGD(model.parameters(), lr=1e-3)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    raw_batch = {
        "text_features": torch.ones(1, 1, 1),
        "audio_features": torch.ones(1, 1, 1),
        "visual_features": torch.ones(1, 1, 1),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
        "lengths": torch.ones(1, dtype=torch.long),
        "speaker_ids_int": torch.zeros(1, 1, dtype=torch.long),
        "labels": torch.zeros(1, 1, dtype=torch.long),
    }
    with pytest.raises(NumericValidationError) as captured:
        train_module.train_one_epoch(
            model,
            [raw_batch],
            optimizer,
            scaler,
            torch.device("cpu"),
            1.0,
            False,
            1,
            model_name="fixture",
            epoch=1,
        )
    assert captured.value.numeric_status == expected_status
    assert captured.value.batch_index == 1


def test_nonfinite_complex_checkpoint_is_rejected_on_reload() -> None:
    checkpoint = {
        "model_state_dict": {
            "bad": torch.tensor(complex(float("nan"), 0.0), dtype=torch.complex64)
        },
        "optimizer_state_dict": {},
        "config": {},
        "model_key": "project_paper_oriented_gsmcc",
        "epoch": 1,
        "best_val_weighted_f1": 0.1,
        "feature_sha256": "unused",
        "test_split_used_for_selection": False,
    }
    numeric = checkpoint_numeric_summary(checkpoint)
    assert numeric["checkpoint_numeric_validation"] == "failed"
    assert numeric["checkpoint_nonfinite_tensor_count"] == 1
    test_root = Path("tmp") / f"pytest_nonfinite_checkpoint_{uuid4().hex}"
    test_root.mkdir(parents=True)
    path = test_root / "nonfinite.pt"
    torch.save(checkpoint, path)
    with pytest.raises(NumericValidationError) as captured:
        load_checkpoint(path, torch.device("cpu"))
    assert captured.value.numeric_status == "NONFINITE_CHECKPOINT"


def test_nonfinite_checkpoint_is_rejected_before_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad_payload = {
        "model_state_dict": {"weight": torch.tensor(float("nan"))},
        "optimizer_state_dict": {},
    }
    monkeypatch.setattr(train_module, "checkpoint_payload", lambda **_kwargs: bad_payload)
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    test_root = Path("tmp") / f"pytest_checkpoint_save_gate_{uuid4().hex}"
    path = test_root / "bad.pt"
    with pytest.raises(NumericValidationError):
        train_module.save_checkpoint(
            path,
            config={"model": {"name": "project_paper_oriented_gsmcc"}, "training": {}},
            optimizer=optimizer,
            scaler=torch.amp.GradScaler("cuda", enabled=False),
            epoch=1,
        )
    assert not path.exists()


def test_diagnostic_records_required_stages_for_two_batches_and_both_modes() -> None:
    config = {
        **_gsmcc_config(True),
        "dataset": {"name": "SYNTHETIC", "num_classes": 3, "label_list": ["a", "b", "c"]},
        "synthetic": {
            "split_sizes": {"train": 4, "val": 2, "test": 2},
            "sequence_length": 3,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "seed": 3,
            "select_best_by": "val_weighted_f1",
            "grad_clip": 1.0,
            "num_workers": 0,
            "amp": False,
        },
        "scheduler": {"name": "none"},
        "system": {"seed": 3, "device": "cpu"},
        "output": {"root": "tmp/unused_diagnostic_output"},
    }
    config = normalized_training_config(config)
    loader = build_dataloader(config, "train", shuffle=False, batch_size=2)
    tensor_rows: list[dict] = []
    gradient_rows: list[dict] = []
    parameter_before: list[dict] = []
    parameter_after: list[dict] = []
    losses: dict = {}
    adjacency: dict = {}
    for mode in ("on", "off"):
        completed = diagnose_mode(
            config,
            loader,
            torch.device("cpu"),
            mode,
            2,
            tensor_rows,
            gradient_rows,
            parameter_before,
            parameter_after,
            losses,
            adjacency,
        )
        assert completed == 2
    names = {row["name"] for row in tensor_rows}
    assert {
        "raw_text_features",
        "text_input",
        "speaker_embedding",
        "text_encoder",
        "temporal_cosine_similarity_nonself",
        "temporal_angular_similarity_nonself",
        "cross_modal_cosine_similarity",
        "cross_modal_angular_similarity",
        "self_loop_weights",
        "raw_adjacency",
        "degree",
        "inv_sqrt_degree",
        "adjacency",
        "low_layer_0.fft_output_real",
        "low_layer_0.fft_output_imag",
        "low_layer_0.complex_gain_real",
        "low_layer_0.inverse_fft",
        "low_frequency_output",
        "high_frequency_output",
        "fused_representation",
        "classification_logits",
        "classification_loss",
        "total_loss",
    }.issubset(names)
    assert all(row["is_finite"] for row in tensor_rows)
    assert all(row["is_finite"] for row in gradient_rows)
    assert all(row["is_finite"] for row in parameter_before)
    assert all(row["is_finite"] for row in parameter_after)
    assert set(losses) == {"on.batch_1", "on.batch_2", "off.batch_1", "off.batch_2"}
    assert all(value["normalized_adjacency_finite"] for value in adjacency.values())


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/gsmcc/project_variant/iemocap/full_context/legacy_mmgcn_features/screening.yaml",
        "configs/gsmcc/project_variant/iemocap/full_context/clean_roberta_features/gsmcc_clean.yaml",
    ],
)
def test_real_gsmcc_two_train_batches_are_finite_when_features_exist(
    config_path: str,
) -> None:
    config = normalized_training_config(load_yaml_config(resolve_path(config_path)))
    feature_path = resolve_path(str(config["dataset"]["feature_pkl_path"]))
    if not feature_path.is_file():
        pytest.skip(f"real feature PKL is unavailable: {feature_path}")
    validate_runtime_config(config)
    verify_feature_sha256(config)
    config["system"]["device"] = "cpu"
    loader = build_dataloader(config, "train", shuffle=False, batch_size=2)
    for contrastive_enabled in (True, False):
        mode_config = copy.deepcopy(config)
        mode_config["model"]["use_contrastive_loss"] = contrastive_enabled
        seed_everything(int(mode_config["system"]["seed"]))
        model = build_original_repro_model(mode_config)
        optimizer = build_optimizer(model, mode_config)
        completed = 0
        for raw_batch in loader:
            completed += 1
            optimizer.zero_grad(set_to_none=True)
            output = forward_batch(model, move_batch(raw_batch, torch.device("cpu")))
            assert torch.isfinite(output["logits"]).all()
            assert torch.isfinite(output["loss"]).all()
            output["loss"].backward()
            assert all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
            if completed == 2:
                break
        assert completed == 2
