"""Frozen 24/56 experiment design; no metric-dependent decisions."""

from copy import deepcopy
import csv
import hashlib
from pathlib import Path

import yaml

from scripts.runtime.multidag_cl_ta_v0.formal import json_identity
from scripts.runtime.multidag_cl_ta_v0.training import PROTOCOL

ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = Path("configs/multidag_cl/tav_full_audit_v1")
SOURCE = ("configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
          "paper_data_reproduction/causal_text_encoder_diagnostic/modality_ablation/"
          "cl7_unilstm_tav.yaml")
MODALITIES = ("TAV", "TA", "TV", "AV", "T", "A", "V")
CORES = ("original", "coop", "track", "dcn")
CURRICULA = ("CL_OFF", "ORIGINAL_CL7")
CONTEXTS = {"past_all_causal": None, "w0": 0, "w1": 1, "w3": 3, "w5": 5}
SCHEMA = "multidag_tav_full_audit_v1"
LINEAGE = "tav_architecture_t1024_a1582_v342_unilstm_100_each_v1"
TRAIN_MATRIX = "TAV_FULL_AUDIT_MATRIX.csv"
EVAL_MATRIX = "TAV_MISSING_MODALITY_EVAL_MATRIX.csv"


def experiment_id(core, modalities, curriculum, context):
    cl = "cl7" if curriculum == "ORIGINAL_CL7" else "cloff"
    return f"tav_{core}_{cl}_{modalities.lower()}_{context}_s100"


def source_sha256():
    # Git checks out CRLF on Windows and LF on the asset host. Hash the
    # UTF-8 source after universal-newline decoding to keep identity portable.
    return hashlib.sha256((ROOT / SOURCE).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def make_config(core="original", modalities="TAV", curriculum="CL_OFF", context="past_all_causal"):
    if core not in CORES or modalities not in MODALITIES or curriculum not in CURRICULA or context not in CONTEXTS:
        raise ValueError("unknown frozen audit axis")
    if core != "original" and (modalities != "TAV" or context != "past_all_causal"):
        raise ValueError("variants are full TAV/all-past only")
    if context != "past_all_causal" and (modalities != "TAV" or curriculum != "CL_OFF"):
        raise ValueError("context audit is Original TAV CL_OFF only")
    training = deepcopy(PROTOCOL)
    training["curriculum"] = curriculum == "ORIGINAL_CL7"
    cl = "cl7" if training["curriculum"] else "cloff"
    return {
        "schema": SCHEMA, "experiment_id": experiment_id(core, modalities, curriculum, context),
        "model_name": "multidag_tav_t1024", "architecture_lineage": LINEAGE,
        "frontend_source": {"config_path": SOURCE, "sha256": source_sha256(), "hash_policy": "UTF8_LF",
                            "encoder": "PaperFormulaModalityEncoder", "causal_text": True},
        "graph_core": {"enabled": core != "original", "variant": core},
        "active_modalities": modalities, "context_mode": context,
        "modality_mask": "deterministic_zero_after_encoder",
        "curriculum": {"variant": curriculum, "bucket_count": 7,
                       "partition": "balanced_stable_contiguous", "schedule": "official_one_bucket_per_epoch",
                       "difficulty": "within_speaker_emotion_shifts_plus_speakers_over_utterances_plus_speakers",
                       "code_lineage": "scripts/runtime/multidag_cl_paper_reimplementation/curriculum.py",
                       "source_config": SOURCE},
        "dataset": {"schema": "iemocap_asym_tav_t1024_v1", "asset_path": None,
                    "model_manifest_path": None, "provenance_manifest_path": None,
                    "feature_dimensions": {"text": 1024, "audio": 1582, "visual": 342}},
        "training": training,
        "runtime": {"splits": ["train", "dev"], "test_policy": "HOLD"},
        "checkpoint": {"selector": ["dev_weighted_f1_desc", "dev_loss_asc", "epoch_asc"],
                       "test_split_used_for_selection": False, "test_evaluation_count": 0},
        "output": {"root": "outputs", "experiment_date": None,
                   "experiment_group": f"tav_audit_{core}_{cl}_{modalities.lower()}_{context}"},
    }


def validate_config(raw):
    try:
        expected = make_config(raw["graph_core"]["variant"], raw["active_modalities"],
                               raw["curriculum"]["variant"], raw["context_mode"])
    except (KeyError, TypeError) as error:
        raise ValueError("missing audit config identity") from error
    if json_identity(raw) != json_identity(expected):
        raise ValueError("frozen TAV audit protocol/config identity mismatch")
    return deepcopy(raw)


def design():
    axes = []
    for curriculum in CURRICULA:
        axes.append((1, "baseline", "modality_x_cl", "original", "TAV", curriculum, "past_all_causal"))
    for phase, curriculum in ((2, "CL_OFF"), (3, "ORIGINAL_CL7")):
        for core in CORES[1:]:
            axes.append((phase, "graph_core", "graphcore_x_cl", core, "TAV", curriculum, "past_all_causal"))
    for modalities in MODALITIES[1:]:
        for curriculum in CURRICULA:
            axes.append((4, "modality_train", "modality_x_cl", "original", modalities, curriculum, "past_all_causal"))
    for context in ("w0", "w1", "w3", "w5"):
        axes.append((5, "context", "context", "original", "TAV", "CL_OFF", context))
    result = []
    for order, (phase, folder, family, core, modalities, curriculum, context) in enumerate(axes, 1):
        raw = make_config(core, modalities, curriculum, context)
        result.append({"experiment_id": raw["experiment_id"], "model_name": raw["model_name"],
                       "graph_core": core, "modalities": modalities, "curriculum": curriculum,
                       "context_mode": context, "seed": "100", "epochs": "30",
                       "config_path": (CONFIG_ROOT / folder / (raw["experiment_id"] + ".yaml")).as_posix(),
                       "output_group": raw["output"]["experiment_group"],
                       "primary_or_secondary": "primary" if curriculum == "CL_OFF" else "secondary",
                       "comparison_family": family, "status": "planned", "phase": str(phase),
                       "execution_order": str(order), "architecture_lineage": LINEAGE,
                       "audit_type": "TRAIN_TIME_MODALITY_ABLATION" if folder == "modality_train" else "FORMAL_TRAINING",
                       "config_sha256": json_identity(raw)})
    return result


def evaluation_design():
    return [{"evaluation_id": f"{row['experiment_id']}__dev_{mask.lower()}",
             "source_experiment_id": row["experiment_id"], "graph_core": row["graph_core"],
             "curriculum": row["curriculum"], "trained_modalities": "TAV", "active_modalities": mask,
             "split": "dev", "audit_type": "TEST_TIME_MODALITY_MASK", "phase": "6", "status": "planned"}
            for row in design()[:8] for mask in MODALITIES]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_matrices(directory=ROOT / CONFIG_ROOT):
    directory = Path(directory)
    training, evaluation = read_csv(directory / TRAIN_MATRIX), read_csv(directory / EVAL_MATRIX)
    if training != design() or evaluation != evaluation_design():
        raise ValueError("matrix must exactly match frozen 24/56 design and phase order")
    for row in training:
        raw = validate_config(yaml.safe_load((ROOT / row["config_path"]).read_text(encoding="utf-8")))
        if json_identity(raw) != row["config_sha256"]:
            raise ValueError("matrix config SHA mismatch")
    return training, evaluation
