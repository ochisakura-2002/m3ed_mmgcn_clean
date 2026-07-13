from __future__ import annotations

import pickle
import random
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from datasets.iemocap import (
    IEMOCAPOfficialFeatureDataset,
    IEMOCAP_TRAIN_SESSION_IDS,
    parse_iemocap_session_id,
)


TRAIN_IDS = [
    "Ses03F_impro01",
    "Ses01F_impro01",
    "Ses04M_impro01",
    "Ses02M_impro01",
    "Ses01M_impro02",
    "Ses03M_impro02",
    "Ses02F_impro02",
    "Ses04F_impro02",
]
TEST_IDS = ["Ses05F_impro01", "Ses05M_impro02"]


def _feature_rows(base: float, width: int) -> List[List[float]]:
    return [
        [base + float(column) for column in range(width)],
        [base + 10.0 + float(column) for column in range(width)],
    ]


def _write_synthetic_feature_pkl(path: Path) -> None:
    dialogue_ids = TRAIN_IDS + TEST_IDS
    video_ids: Dict[str, List[str]] = {}
    video_speakers: Dict[str, List[str]] = {}
    video_labels: Dict[str, List[int]] = {}
    video_text: Dict[str, List[List[float]]] = {}
    video_audio: Dict[str, List[List[float]]] = {}
    video_visual: Dict[str, List[List[float]]] = {}
    video_sentence: Dict[str, List[str]] = {}

    for index, dialogue_id in enumerate(dialogue_ids):
        video_ids[dialogue_id] = [
            f"{dialogue_id}_F000",
            f"{dialogue_id}_M001",
        ]
        video_speakers[dialogue_id] = ["F", "M"]
        video_labels[dialogue_id] = [index % 6, (index + 1) % 6]
        video_text[dialogue_id] = _feature_rows(float(index), width=4)
        video_audio[dialogue_id] = _feature_rows(float(index + 20), width=3)
        video_visual[dialogue_id] = _feature_rows(float(index + 40), width=2)
        video_sentence[dialogue_id] = ["first", "second"]

    payload = [
        video_ids,
        video_speakers,
        video_labels,
        video_text,
        video_audio,
        video_visual,
        video_sentence,
        list(TRAIN_IDS),
        list(TEST_IDS),
    ]
    with path.open("wb") as file:
        pickle.dump(payload, file)


def _sessions(dialogue_ids: Iterable[str]) -> Set[str]:
    return {parse_iemocap_session_id(dialogue_id) for dialogue_id in dialogue_ids}


def _qualified_speakers(
    dataset: IEMOCAPOfficialFeatureDataset,
    dialogue_ids: Iterable[str],
) -> Set[str]:
    speakers: Set[str] = set()
    for dialogue_id in dialogue_ids:
        session_id = parse_iemocap_session_id(dialogue_id)
        speakers.update(
            f"{session_id}{str(speaker).upper()}"
            for speaker in dataset.videoSpeakers[dialogue_id]
        )
    return speakers


def _assert_pairwise_disjoint(values: Sequence[Set[str]]) -> None:
    for first, second in combinations(values, 2):
        if not first.isdisjoint(second):
            raise AssertionError(f"Expected disjoint sets, got overlap={first & second}")


class TestIEMOCAPSessionHoldout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temp_dir = tempfile.TemporaryDirectory()
        cls.synthetic_feature_pkl = Path(cls._temp_dir.name) / "IEMOCAP_features.pkl"
        _write_synthetic_feature_pkl(cls.synthetic_feature_pkl)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp_dir.cleanup()

    def test_session_holdout_is_ordered_and_disjoint(self) -> None:
        for val_session_id in IEMOCAP_TRAIN_SESSION_IDS:
            with self.subTest(val_session_id=val_session_id):
                dataset = IEMOCAPOfficialFeatureDataset(
                    feature_pkl_path=self.synthetic_feature_pkl,
                    split="train",
                    valid_ratio=0.99,
                    val_split_strategy="session_holdout",
                    val_session_id=val_session_id,
                    seed=999,
                )
                split_ids = dataset.get_split_ids()

                expected_val = [
                    dialogue_id
                    for dialogue_id in TRAIN_IDS
                    if parse_iemocap_session_id(dialogue_id) == val_session_id
                ]
                expected_train = [
                    dialogue_id
                    for dialogue_id in TRAIN_IDS
                    if parse_iemocap_session_id(dialogue_id) != val_session_id
                ]

                self.assertEqual(split_ids["train"], expected_train)
                self.assertEqual(split_ids["val"], expected_val)
                self.assertEqual(split_ids["test"], TEST_IDS)
                self.assertEqual(dataset.keys, expected_train)
                self.assertEqual(
                    set(split_ids["train"]) | set(split_ids["val"]),
                    set(TRAIN_IDS),
                )

                dialogue_sets = [
                    set(split_ids[name]) for name in ("train", "val", "test")
                ]
                session_sets = [
                    _sessions(split_ids[name]) for name in ("train", "val", "test")
                ]
                speaker_sets = [
                    _qualified_speakers(dataset, split_ids[name])
                    for name in ("train", "val", "test")
                ]
                _assert_pairwise_disjoint(dialogue_sets)
                _assert_pairwise_disjoint(session_sets)
                _assert_pairwise_disjoint(speaker_sets)

                self.assertEqual(session_sets[1], {val_session_id})
                self.assertEqual(session_sets[2], {"Ses05"})

    def test_session_holdout_test_dataset_keeps_official_test_order(self) -> None:
        dataset = IEMOCAPOfficialFeatureDataset(
            feature_pkl_path=self.synthetic_feature_pkl,
            split="test",
            val_split_strategy="session_holdout",
            val_session_id="Ses02",
        )

        self.assertEqual(dataset.keys, TEST_IDS)
        self.assertEqual(dataset.get_split_ids()["test"], TEST_IDS)

    def test_official_prefix_behavior_is_unchanged(self) -> None:
        dataset = IEMOCAPOfficialFeatureDataset(
            feature_pkl_path=self.synthetic_feature_pkl,
            split="train",
            valid_ratio=0.25,
            val_split_strategy="official_prefix",
            val_session_id="Ses04",
            seed=17,
        )

        split_ids = dataset.get_split_ids()
        self.assertEqual(split_ids["val"], TRAIN_IDS[:2])
        self.assertEqual(split_ids["train"], TRAIN_IDS[2:])
        self.assertEqual(split_ids["test"], TEST_IDS)

    def test_random_behavior_is_unchanged(self) -> None:
        seed = 17
        shuffled = list(TRAIN_IDS)
        random.Random(seed).shuffle(shuffled)

        dataset = IEMOCAPOfficialFeatureDataset(
            feature_pkl_path=self.synthetic_feature_pkl,
            split="val",
            valid_ratio=0.25,
            val_split_strategy="random",
            val_session_id="Ses04",
            seed=seed,
        )

        split_ids = dataset.get_split_ids()
        self.assertEqual(split_ids["val"], shuffled[:2])
        self.assertEqual(split_ids["train"], shuffled[2:])
        self.assertEqual(split_ids["test"], TEST_IDS)
        self.assertEqual(dataset.keys, shuffled[:2])


if __name__ == "__main__":
    unittest.main()
