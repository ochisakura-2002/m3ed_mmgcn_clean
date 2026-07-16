from __future__ import annotations

import math
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.iemocap.official_feature_dataset import iemocap_dialogue_collate_fn
from datasets.iemocap.original_repro_split import (
    build_official_train_split,
    build_outer_session_split,
)
from models.baselines.original_repro.dialoguegcn import (
    DialogueGCNRelationalGraphNetwork,
    build_dialoguegcn_graph,
    dialoguegcn_relation_id,
)
from scripts.baselines.original_merc_runtime import curriculum_train_loader
from models.baselines.original_repro.gsmcc.model import (
    build_sliding_multimodal_graph,
    cross_frequency_contrastive_loss,
)
from models.baselines.original_repro.multidag_cl import (
    build_get_adj_v1,
    curriculum_baby_step_indices,
    dialogue_difficulty_from_sequences,
)


def _split_metadata():
    ids = [f"Ses{session:02d}{speaker}_impro{index:02d}" for session in range(1, 6) for speaker in "FM" for index in range(2)]
    labels = {dialogue_id: [0, 1, 1, 2] for dialogue_id in ids}
    speakers = {dialogue_id: ["F", "M", "F", "M"] for dialogue_id in ids}
    return ids, labels, speakers


def test_all_outer_session_folds_are_disjoint_deterministic_and_complete() -> None:
    ids, labels, speakers = _split_metadata()
    for session in range(1, 6):
        first = build_outer_session_split(
            ids, labels, speakers, f"Ses{session:02d}", inner_val_ratio=0.2, split_seed=13
        )
        second = build_outer_session_split(
            ids, labels, speakers, f"Ses{session:02d}", inner_val_ratio=0.2, split_seed=13
        )
        assert first == second
        train = set(first.train_dialogue_ids)
        val = set(first.inner_val_dialogue_ids)
        test = set(first.test_dialogue_ids)
        assert not train & val and not train & test and not val & test
        assert train | val | test == set(ids)
        assert first.test_split_used_for_selection is False
        assert all(dialogue_id.startswith(f"Ses{session:02d}") for dialogue_id in test)


def test_official_split_keeps_testvid_untouched_and_validation_inside_trainvid() -> None:
    ids, labels, speakers = _split_metadata()
    train_vid = [value for value in ids if not value.startswith("Ses05")]
    test_vid = [value for value in ids if value.startswith("Ses05")]
    split = build_official_train_split(
        train_vid, test_vid, labels, speakers, inner_val_ratio=0.2, split_seed=42
    )
    assert list(split.test_dialogue_ids) == test_vid
    assert set(split.train_dialogue_ids) | set(split.inner_val_dialogue_ids) == set(train_vid)
    assert not set(split.inner_val_dialogue_ids) & set(test_vid)
    assert split.test_split_used_for_selection is False


def test_multidag_predecessor_scan_and_curriculum_equation() -> None:
    speakers = torch.tensor([[0, 1, 1, 0, 1]])
    mask = torch.ones_like(speakers)
    adjacency = build_get_adj_v1(speakers, mask, window_past=1)
    # Target 4 scans back through target 3 and stops at same-speaker target 2.
    assert adjacency[0, 4].tolist() == [False, False, True, True, False]
    difficulty = dialogue_difficulty_from_sequences([0, 1, 2, 0], [0, 1, 0, 1])
    assert math.isclose(difficulty, (2 + 2) / (4 + 2))
    visible = curriculum_baby_step_indices([0.8, 0.1, 0.4, 0.2], epoch=1, bucket_count=2)
    assert visible == [1, 3]


def test_dialoguegcn_relation_space_and_temporal_windows() -> None:
    mask = torch.tensor([[1, 1, 1]])
    speakers = torch.tensor([[0, 1, 0]])
    adjacency, relation_ids, mapping = build_dialoguegcn_graph(mask, speakers, 2, 1, 1)
    assert len(mapping) == 8
    assert adjacency[0, 1].tolist() == [True, True, True]
    assert relation_ids[0, 0, 1].item() == dialoguegcn_relation_id(0, 1, 0, 1, 2)
    assert relation_ids[0, 2, 1].item() == dialoguegcn_relation_id(0, 1, 2, 1, 2)


def test_dialoguegcn_two_layer_formula_matches_hand_computed_fixture() -> None:
    network = DialogueGCNRelationalGraphNetwork(
        input_dim=2,
        graph_hidden_dim=2,
        num_relations=2,
        dropout=0.0,
    ).double()
    with torch.no_grad():
        network.relation_weights.zero_()
        network.relation_weights[0].copy_(torch.eye(2, dtype=torch.float64))
        network.relation_weights[1].copy_(2.0 * torch.eye(2, dtype=torch.float64))
        network.root.weight.copy_(3.0 * torch.eye(2, dtype=torch.float64))
        network.second_neighbor.weight.copy_(torch.eye(2, dtype=torch.float64))
        network.second_root.weight.copy_(2.0 * torch.eye(2, dtype=torch.float64))

    nodes = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=torch.float64
    )
    adjacency = torch.ones(1, 3, 3, dtype=torch.bool)
    relation_ids = torch.tensor([[[0, 0, 0], [0, 0, 1], [1, 1, 0]]])
    attention = torch.tensor(
        [[[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.4]]],
        dtype=torch.float64,
    )
    output = network(
        nodes,
        adjacency,
        relation_ids,
        attention,
        torch.ones(1, 3, dtype=torch.long),
    )
    expected = torch.tensor(
        [[[20.1, 26.4], [26.3, 34.1], [24.8, 31.9]]], dtype=torch.float64
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=1e-12)


class _FixedDialogueDataset(Dataset):
    split = "train"

    def __init__(self) -> None:
        self.dialogues = [
            ("Ses01F_impro01", [0, 0, 0], [0, 0, 0]),
            ("Ses02M_impro01", [0, 1, 0, 1], [0, 1, 0, 1]),
            ("Ses03F_impro01", [2, 2, 1, 1], [0, 1, 0, 1]),
            ("Ses04M_impro01", [0, 1, 2, 0, 1], [0, 0, 0, 0, 0]),
        ]

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, index: int) -> dict:
        dialogue_id, labels, speakers = self.dialogues[index]
        length = len(labels)
        return {
            "dialogue_id": dialogue_id,
            "utterance_ids": [f"{dialogue_id}_{i}" for i in range(length)],
            "sentences": ["fixed real-label fixture"] * length,
            "text_features": torch.zeros(length, 2),
            "audio_features": torch.zeros(length, 2),
            "visual_features": torch.zeros(length, 2),
            "labels": torch.tensor(labels),
            "speaker_ids_int": torch.tensor(speakers),
            "length": length,
        }


def test_multidag_curriculum_real_dialogue_fixture_expands_and_excludes_test() -> None:
    dataset = _FixedDialogueDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=iemocap_dialogue_collate_fn)
    enabled = SimpleNamespace(use_curriculum_learning=True, curriculum_bucket_count=3)
    epoch_one = curriculum_train_loader(loader, enabled, epoch=1, seed=42)
    epoch_two = curriculum_train_loader(loader, enabled, epoch=2, seed=42)
    assert 0 < len(epoch_one.dataset) < len(epoch_two.dataset) <= len(dataset)

    disabled = SimpleNamespace(use_curriculum_learning=False, curriculum_bucket_count=3)
    assert curriculum_train_loader(loader, disabled, epoch=1, seed=42) is loader

    dataset.split = "test"
    with torch.no_grad():
        try:
            curriculum_train_loader(loader, enabled, epoch=1, seed=42)
        except ValueError as error:
            assert "train split" in str(error)
        else:
            raise AssertionError("test split entered curriculum ordering")


def test_gsmcc_graph_frequency_branches_and_contrastive_negatives() -> None:
    torch.manual_seed(3)
    modalities = tuple(torch.randn(1, 3, 4) for _ in range(3))
    mask = torch.tensor([[1, 1, 0]])
    adjacency, node_mask = build_sliding_multimodal_graph(modalities, mask, window=1)
    assert adjacency.shape == (1, 9, 9)
    assert torch.allclose(adjacency, adjacency.transpose(1, 2), atol=1e-6)
    assert node_mask.sum().item() == 6
    low = torch.randn(1, 9, 4, requires_grad=True)
    high = torch.randn(1, 9, 4, requires_grad=True)
    loss = cross_frequency_contrastive_loss(low, high, node_mask, temperature=0.2)
    colder = cross_frequency_contrastive_loss(low, high, node_mask, temperature=0.1)
    assert torch.isfinite(loss) and loss.item() > 0
    assert not torch.allclose(loss, colder)
    loss.backward()
    assert low.grad is not None and high.grad is not None
