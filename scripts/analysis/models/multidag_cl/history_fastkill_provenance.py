"""Read the one authenticated recovery artifact without reranking its samples."""

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from zipfile import ZipFile

import numpy as np

from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file

from .failure_audit_inference import HISTORY
from .failure_audit_metrics import GROUPS, metadata
from .pairwise_preflight_inputs import CLASS_ORDER, F0_IDENTITY, H_GROUPS, SEED, validate_hold
from .run_history_top25_recovery import ANCHORS, GROUP, HISTORY_CSV, H_CSV, MANIFEST, REPORT, SHIFT_CSV, UH_CSV


RECOVERY_BATCH = "history_top25_seed100_20261124_212530"
RECOVERY_SHA256 = "519740deb419683d978c1eadca5879855d8818e9fb3c4bdc34e628e91a229d3e"
RECOVERY_OUTPUTS = (HISTORY_CSV, H_CSV, UH_CSV, SHIFT_CSV, REPORT)


@dataclass(frozen=True)
class RecoveryMembership:
    manifest: dict
    manifest_sha256: str
    population: tuple
    predictions: tuple
    groups: dict

    def verify_source(self, source, projection, dev_population):
        evidence = self.manifest
        if (self.population != source.population or evidence["f0_checkpoint"] != source.identity
                or evidence["f0_run_id"] != source.config["experiment_id"]
                or evidence["f0_dev_metrics"] != source.manifest["best_dev_metrics"]
                or evidence["archived_dev_logits_sha256"] != json_identity(source.logits.tolist())
                or evidence["train_dev_projection"] != projection
                or evidence["dev_population"] != dev_population
                or self.predictions != tuple(source.logits.argmax(axis=1))):
            raise ValueError("recovery artifact does not match the authenticated F0 and Train/Dev projection")


def read_payload(directory, archive):
    if sha256_file(archive) != RECOVERY_SHA256:
        raise ValueError("recovery archive SHA-256 differs from the frozen batch")
    names = (*RECOVERY_OUTPUTS, MANIFEST)
    with ZipFile(archive) as zipped:
        if sorted(zipped.namelist()) != sorted(names):
            raise ValueError("recovery ZIP member list differs from the frozen schema")
        files = {name: zipped.read(name) for name in names}
    if sha256_file(archive) != RECOVERY_SHA256 or any(
            (directory / name).read_bytes() != value for name, value in files.items()):
        raise ValueError("recovery directory files differ from the frozen archive")
    return files


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError("invalid recovery membership boolean")
    return value == "True"


def read_recovery(directory, archive):
    files = read_payload(directory, archive)
    manifest = json.loads(files[MANIFEST])
    validate_hold(manifest)
    validate_hold(manifest["train_dev_projection"])
    expected = {**ANCHORS, "schema": "ct_multidag_history_top25_recovery_v1", "experiment_group": GROUP,
                "batch_id": RECOVERY_BATCH, "split": "dev", "seed": SEED, "f0_checkpoint": F0_IDENTITY,
                "history_conditions": list(HISTORY), "FORMAL_TRAIN_RUNS": 0, "OPTIMIZER_STEPS": 0,
                "WORKSTREAM_CHECK": "PASS", "PROVENANCE_RECOVERY": "PASS", "F0_IDENTITY": "PASS",
                "H_TOP25_PROVENANCE": "PASS", "H_SUBGROUP_PROVENANCE": "PASS", "TEST_TOUCHED": "NO"}
    if (any(manifest.get(key) != value for key, value in expected.items())
            or manifest["outputs_sha256"] != {name: hashlib.sha256(files[name]).hexdigest()
                                              for name in RECOVERY_OUTPUTS}):
        raise ValueError("recovery manifest anchors or output digests differ")

    h, uh, shift, history = (csv_rows(files[name]) for name in (H_CSV, UH_CSV, SHIFT_CSV, HISTORY_CSV))
    indexes = list(range(ANCHORS["DEV_POPULATION"]))
    if any([int(row["dev_index"]) for row in rows] != indexes for rows in (uh, shift)):
        raise ValueError("recovery Dev index is incomplete, duplicated or reordered")
    population = tuple((row["utterance_id"], row["dialogue_id"], CLASS_ORDER.index(row["gold"])) for row in uh)
    if (len({uid for uid, _, _ in population}) != len(indexes)
            or json_identity(population) != manifest["dev_gold_population_sha256"]):
        raise ValueError("recovery Dev gold population identity differs")
    dev_meta = metadata(population)
    fields = ("utterance_id", "dialogue_id", "target_index", "speaker", "true_label", "shift_group")

    def matches_meta(row, index):
        return (int(row["seed"]) == SEED and row["gold"] == dev_meta[index]["true_label"]
                and all(row[key] == str(dev_meta[index][key]) for key in fields))

    if any(not matches_meta(row, index) for table in (uh, shift) for index, row in enumerate(table)):
        raise ValueError("recovery SHIFT or U/H keys, labels or metadata differ")
    predictions = tuple(CLASS_ORDER.index(row["full_prediction"]) for row in uh)
    wrong = np.asarray([prediction != row[2] for row, prediction in zip(population, predictions)])
    u_member = np.asarray([boolean(row["is_u_top25"]) for row in uh])
    h_member = np.asarray([boolean(row["is_h_top25"]) for row in uh])
    h_indexes = [int(row["dev_index"]) for row in h]
    if (len(h_indexes) != len(set(h_indexes)) or set(h_indexes) != set(np.flatnonzero(h_member))
            or [int(row["rank"]) for row in h] != list(range(1, ANCHORS["H_TOP25_SUPPORT"] + 1))
            or any(not matches_meta(row, index) or not boolean(row["is_h_top25"])
                   or row["full_prediction"] != uh[index]["full_prediction"] for index, row in zip(h_indexes, h))):
        raise ValueError("frozen H-top25 rank/keys differ from the U/H membership")

    partitions = dict(zip(H_GROUPS, (u_member & ~h_member, h_member & ~u_member,
                                    u_member & h_member, ~(u_member | h_member))))
    subgroup = np.asarray([row["subgroup"] for row in uh])
    shifts = np.asarray([row["shift_group"] for row in shift])
    if (any(not np.array_equal(mask, subgroup == name) for name, mask in partitions.items())
            or Counter(subgroup) != manifest["uh_subgroup_counts"]
            or Counter(shifts) != {name: ANCHORS[name] for name in GROUPS}
            or (int(wrong.sum()), int(h_member.sum()), int((h_member & wrong).sum()),
                int(u_member.sum()), int((u_member & wrong).sum())) != (
                    ANCHORS["F0_TOTAL_ERRORS"], ANCHORS["H_TOP25_SUPPORT"], ANCHORS["H_TOP25_F0_ERRORS"],
                    ANCHORS["U_TOP25_SUPPORT"], ANCHORS["U_TOP25_F0_ERRORS"])):
        raise ValueError("recovery U/H partition, SHIFT support or F0 error anchors differ")

    history_keys = set()
    for row in history:
        index = int(row["dev_index"])
        condition = row["condition"]
        if (index not in range(len(indexes)) or condition not in HISTORY or row["row_type"] != "sample"
                or not matches_meta(row, index) or row["prediction"] not in CLASS_ORDER
                or (condition == "FULL_HISTORY" and row["prediction"] != uh[index]["full_prediction"])):
            raise ValueError("recovered history trace disagrees with the frozen Dev population or F0")
        history_keys.add((index, condition))
    if len(history_keys) != ANCHORS["HISTORY_SAMPLE_ROWS"] or len(history) != len(history_keys):
        raise ValueError("recovered history trace lacks complete unique Dev/condition pairs")
    groups = {"H_TOP25": h_member, **partitions, **{name: shifts == name for name in GROUPS}}
    return RecoveryMembership(manifest, hashlib.sha256(files[MANIFEST]).hexdigest(), population, predictions, groups)
