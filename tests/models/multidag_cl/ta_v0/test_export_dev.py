import csv
import json

import numpy as np
import pytest
import torch

from scripts.models.multidag_cl.ta_v0.export_dev import PREDICTION_COLUMNS, export_dev
from scripts.models.multidag_cl.ta_v0.train import MODEL_INPUTS, synthetic_runner


@pytest.fixture(autouse=True, scope="module")
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture():
    runner = synthetic_runner()
    runner.batch_size = 1
    batches = list(runner.loader("dev"))
    expected = {}
    with torch.no_grad():
        runner.model.eval()
        for batch in batches:
            output = runner.model(**{k: batch[k] for k in MODEL_INPUTS})
            for b, ids in enumerate(batch["utterance_ids"]):
                for t, uid in enumerate(ids):
                    expected[uid] = (int(batch["labels"][b, t]),
                                     output.logits_ta[b, t].numpy().copy(),
                                     output.prob_ta[b, t].numpy().copy(),
                                     output.h_ta[b, t].numpy().copy())
    runner.model.train()
    return runner, batches, expected


def test_export_preserves_id_label_logits_prob_and_h_ta_under_shuffled_batches(tmp_path):
    runner, batches, expected = fixture()
    directory = tmp_path / "export"
    result = export_dev(
        runner, reversed(batches), directory, expected_ids=list(reversed(expected)),
        expected_rows=4, checkpoint_identity="synthetic-checkpoint", config_identity="synthetic-config",
    )
    assert runner.model.training
    assert result["rows"] == result["unique_utterance_ids"] == 4
    assert result["padding_rows"] == result["duplicate_ids"] == 0
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert tuple(reader.fieldnames) == PREDICTION_COLUMNS
        predictions = list(reader)
    with np.load(directory / "h_ta.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"utterance_ids", "h_ta", "labels", "split"}
        ids = arrays["utterance_ids"].tolist()
        assert ids == sorted(expected)
        assert ids == [row["utterance_id"] for row in predictions]
        assert arrays["h_ta"].shape == (4, 1500)
        assert arrays["split"].tolist() == ["dev"] * 4
        for index, row in enumerate(predictions):
            label, logits, probs, h_ta = expected[row["utterance_id"]]
            assert arrays["labels"][index] == int(row["true_label"]) == label
            assert int(row["predicted_label"]) == int(logits.argmax())
            np.testing.assert_allclose([float(row[f"logit_{i}"]) for i in range(6)], logits)
            np.testing.assert_allclose([float(row[f"prob_{i}"]) for i in range(6)], probs)
            np.testing.assert_allclose(arrays["h_ta"][index], h_ta)
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["representation_name"] == "ta_v0_contextual_preclassifier"
    assert metadata["dimension"] == 1500
    assert metadata["label_vocab"] == ["hap", "sad", "neu", "ang", "exc", "fru"]
    assert metadata["checkpoint_identity"] == "synthetic-checkpoint"
    assert metadata["config_identity"] == "synthetic-config"


@pytest.mark.parametrize("case", ["duplicate", "missing", "test", "wrong_ids", "wrong_rows"])
def test_export_fails_before_writing_on_identity_split_or_count_error(tmp_path, case):
    runner, batches, expected = fixture()
    ids, count = list(expected), 4
    if case == "duplicate":
        batches.append(batches[0])
    elif case == "missing":
        batches.pop()
    elif case == "test":
        batches[0]["split"] = "test"
    elif case == "wrong_ids":
        ids[0] = "unknown"
    elif case == "wrong_rows":
        count = 980
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError):
        export_dev(runner, batches, directory, expected_ids=ids, expected_rows=count,
                   checkpoint_identity="synthetic", config_identity="synthetic")
    assert not directory.exists()


def test_export_excludes_padding_and_refuses_existing_directory(tmp_path):
    runner = synthetic_runner()
    batches = list(runner.loader("dev"))
    assert batches[0]["attention_mask"].shape == (2, 3)
    assert int(batches[0]["attention_mask"].sum()) == 4
    ids = [uid for batch in batches for dialogue in batch["utterance_ids"] for uid in dialogue]
    directory = tmp_path / "padded"
    result = export_dev(runner, batches, directory, expected_ids=ids, expected_rows=4,
                        checkpoint_identity="synthetic", config_identity="synthetic")
    assert result["rows"] == 4 and result["padding_rows"] == 0
    original = (directory / "predictions.csv").read_bytes()
    with pytest.raises(FileExistsError):
        export_dev(runner, batches, directory, expected_ids=ids, expected_rows=4,
                   checkpoint_identity="other-checkpoint", config_identity="synthetic")
    assert (directory / "predictions.csv").read_bytes() == original
