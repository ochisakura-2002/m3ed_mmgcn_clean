"""Authenticate frozen Protocol B membership and adapt only formal Train dialogues."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import json

import yaml
from torch.utils.data import Dataset

from models.multidag_cl.ta_v0 import LABEL_VOCAB
from scripts.analysis.models.multidag_cl.pairwise_preflight_inputs import (
    F0_IDENTITY, HOLD, PROJECTION_MANIFEST, SPLITS, TRAIN_DEV_COUNTS, load_projected_train_dev, validate_hold,
)
from scripts.analysis.models.multidag_cl.post_ph.internal_protocol import metadata
from scripts.runtime.multidag_cl_fusion.spec import make_final_config
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity


SESSIONS = ("Ses01", "Ses02", "Ses03", "Ses04")
FIRST_FOLD = "Ses03"
SEED = 100
ENDPOINT = 30
PARAMETERS = 1_638_306
GROUP = "g0_group_isolated_ct_multidag"
SCHEMA = "ct_multidag_g0_protocol_b_v1"
CONFIG_DIR = Path("configs/multidag_cl/tav_full_audit_v1/group_isolated_protocol_b")
PROJECTION_FIELDS = ("schema", "splits", "split_counts", "source_data_identity",
                     "sha256", "row_order_identity_sha256", *HOLD)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fold_config(session):
    if session not in SESSIONS:
        raise ValueError("expected a Protocol B session from Ses01 through Ses04")
    return {
        "schema": SCHEMA,
        "held_out_session": session,
        "seed": SEED,
        "checkpoint_epoch": ENDPOINT,
        "calibration": None,
        "run_id": f"ct_multidag_g0_{session.lower()}_s{SEED}",
        "output": {"root": "outputs", "experiment_group": GROUP},
    }


def load_config(path):
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or json_identity(config) != json_identity(fold_config(config.get("held_out_session"))):
        raise ValueError("G0 fold config differs from the frozen protocol")
    return config


def read_frozen(root):
    frozen = read_json(root / CONFIG_DIR / "fold_identities.json")
    raw = make_final_config("F0", SEED)
    architecture = {key: raw[key] for key in ("architecture_lineage", "frontend_source", "graph_core", "fusion")}
    if (frozen["schema"] != SCHEMA or frozen["formal_f0"]["seed100_checkpoint_identity"] != F0_IDENTITY
            or json_identity(raw) != frozen["formal_f0"]["config_sha256"][str(SEED)]
            or json_identity(architecture) != frozen["formal_f0"]["architecture_sha256"]
            or raw["training"]["max_epochs"] != ENDPOINT or tuple(LABEL_VOCAB) != tuple(frozen["label_order"])):
        raise ValueError("HEAD F0 config, architecture, or vocabulary differs from the audited recipe")
    if set(frozen["folds"]) != set(SESSIONS):
        raise ValueError("expected four frozen Protocol B identities")
    full = None
    held_dialogues = set()
    for session in SESSIONS:
        fold = frozen["folds"][session]
        fit, held = fold["fit"], fold["held_out"]
        assert_isolation(fit, held, session)
        population = set(fit["dialogue_ids"]) | set(held["dialogue_ids"])
        if full is None:
            full = population
        if (population != full or held_dialogues & set(held["dialogue_ids"])
                or len(population) != 100 or fit["utterances"] + held["utterances"] != 4830):
            raise ValueError("Protocol B folds fail exact coverage of the formal Train universe")
        held_dialogues.update(held["dialogue_ids"])
    if held_dialogues != full:
        raise ValueError("each formal Train dialogue must be held out exactly once")
    return frozen, raw


def assert_isolation(fit, held, session):
    expected_sessions = {"fit": set(SESSIONS) - {session}, "held_out": {session}}
    for role, view in (("fit", fit), ("held_out", held)):
        sessions = expected_sessions[role]
        ids = view["dialogue_ids"]
        if (set(view["sessions"]) != sessions or set(view["actors"]) != {s + g for s in sessions for g in "FM"}
                or set(did[:5] for did in ids) != sessions or len(ids) != len(set(ids))
                or view["dialogues"] != len(ids) or len(view["class_support"]) != len(LABEL_VOCAB)
                or any(n <= 0 for n in view["class_support"]) or sum(view["class_support"]) != view["utterances"]
                or sum(view["dialogue_type_support"].values()) != view["utterances"]):
            raise ValueError(f"{role} is inconsistent with the frozen session/actor assignment")
    if any(set(fit[key]) & set(held[key]) for key in ("dialogue_ids", "sessions", "actors")):
        raise ValueError("dialogue, session or actor leakage")


def population(dataset):
    """Read the authenticated physical dialogue and canonical turn order without creating a batch."""
    result = []
    for asset_id in dataset.dialogue_ids:
        for _, _, uid in dataset.ordered[asset_id]:
            row = dataset.rows[uid]
            result.append((uid, row["dialogue_id"], int(row["benchmark_label_id"])))
    return result


def describe(rows):
    turns_by_dialogue = {}
    sessions, actors = set(), set()
    counts, types = Counter(), Counter()
    seen = set()
    for uid, did, gold in rows:
        if uid in seen or gold not in range(len(LABEL_VOCAB)):
            raise ValueError("duplicate utterance or invalid label in population tape")
        seen.add(uid)
        meta = metadata(uid, did)
        sessions.add(meta["session"])
        actors.add(meta["speaker_id"])
        counts[gold] += 1
        types[meta["dialogue_type"]] += 1
        turns_by_dialogue.setdefault(did, []).append(uid)
    return {
        "dialogue_ids": list(turns_by_dialogue),
        "dialogues": len(turns_by_dialogue),
        "utterances": len(rows),
        "sessions": sorted(sessions),
        "actors": sorted(actors),
        "class_support": [counts[i] for i in range(len(LABEL_VOCAB))],
        "dialogue_type_support": dict(types),
        "order_sha256": json_identity(list(turns_by_dialogue.items())),
        "labeled_tape_sha256": json_identity(rows),
    }


class FoldView(Dataset):
    """Only formal Train dialogue items. Runtime split tags are local to G0."""

    def __init__(self, items, split):
        self.items = tuple(items)
        self.split = split

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return {**self.items[index], "split": self.split}


def load_fold(config, frozen, projection_dir, *, fit_only=False):
    # Bind to the checksums recorded by Ubuntu J/K before the existing
    # projection loader can open the certified pickle.
    certificate = read_json(projection_dir / PROJECTION_MANIFEST)
    formal = frozen["formal_f0"]
    expected_certificate = {
        **frozen["certified_projection"], **HOLD, "splits": list(SPLITS),
        "split_counts": TRAIN_DEV_COUNTS, "source_data_identity": formal["data_identity"],
    }
    validate_hold(certificate)
    if expected_certificate != {key: certificate.get(key) for key in PROJECTION_FIELDS}:
        raise ValueError("projection must match the existing authenticated POST-PH J/K record")
    source = SimpleNamespace(manifest={"data_identity": formal["data_identity"]})
    datasets, projection = load_projected_train_dev(projection_dir, source)
    if projection != expected_certificate:
        raise ValueError("loaded projection differs from the POST-PH J/K identity")
    train_rows = population(datasets["train"])
    dev_rows = population(datasets["dev"])
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        summary = describe(rows)
        tape = {key: summary[key] for key in ("dialogues", "utterances")}
        tape["sha256"] = summary["order_sha256"]
        if (tape != formal["population"][split]
                or summary["labeled_tape_sha256"] != formal["labeled_tape_sha256"][split]):
            raise ValueError(f"certified {split} order, IDs or gold differ from formal F0")
    for key in (0, 1):
        if {row[key] for row in train_rows} & {row[key] for row in dev_rows}:
            raise ValueError("formal Dev overlaps the formal Train input")

    session = config["held_out_session"]
    expected = frozen["folds"][session]
    entries = {}
    for key in ("fit", "held_out"):
        selected = set(expected[key]["dialogue_ids"])
        entries[key] = [row for row in train_rows if row[1] in selected]
        if describe(entries[key]) != expected[key]:
            raise ValueError(f"{key} population differs from the frozen Protocol B tape")
    if sum(map(len, entries.values())) != len(train_rows):
        raise ValueError("incomplete fold membership")
    owner = {did: key for key in entries for did in expected[key]["dialogue_ids"]}
    splits = {"fit": "train"} if fit_only else {"fit": "train", "held_out": "dev"}
    items = {role: [] for role in splits}
    train = datasets["train"]
    for index, asset_id in enumerate(train.dialogue_ids):
        uid = train.ordered[asset_id][0][2]
        role = owner[train.rows[uid]["dialogue_id"]]
        if role in items:
            items[role].append(train[index])
    views = {role: FoldView(items[role], split) for role, split in splits.items()}
    identity = {
        "schema": SCHEMA, "held_out_session": session, "fold": expected,
        "projection": projection, "formal_f0_population": formal["population"],
        "source_protocol_b_sha256": frozen["source_protocol_b_sha256"], **HOLD,
    }
    return views, entries, identity
