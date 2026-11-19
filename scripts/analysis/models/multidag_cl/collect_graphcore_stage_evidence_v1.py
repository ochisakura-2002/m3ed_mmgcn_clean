"""Collect the complete GraphCore stage from explicit, validated evidence sources."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from scripts.analysis.models.multidag_cl.summarize_tav_full_audit_v1 import (
    collect as collect_audit, metric_columns, read_json, result_tables, write_csv,
)
from scripts.analysis.models.multidag_cl.summarize_tav_multiseed_confirm_v1 import collect as collect_confirm, tables
from scripts.runtime.multidag_cl_graph_core_variants.formal import PARENT_CONFIG, config_path
from scripts.runtime.multidag_cl_ta_v0.formal import json_identity, sha256_file
from scripts.runtime.multidag_cl_tav_full_audit_v1.confirm import (
    CONFIRM_GROUP, MATRIX_PATH, checked_completion, checked_epochs, historical_jobs, test_hold,
)
from scripts.runtime.multidag_cl_tav_full_audit_v1.spec import (
    CONFIG_ROOT, EVAL_MATRIX, SOURCE, TRAIN_MATRIX, load_matrices, validate_config,
)
from utils.output_paths import resolve_output_paths, validate_experiment_date

BASELINE, SCREEN, AUDIT, CONFIRM, PRED, DATA, CONFIG, DOCS, INVENTORY = (
    "01_baseline", "02_true_ta_graphcore", "03_tav_full_audit", "04_multiseed_confirm",
    "05_predictions_and_perclass", "06_data_provenance", "07_configs_and_matrices", "08_research_docs", "09_inventory",
)
TA_RUNS = {"t768": "ta_v0", "t1024": "ta_v0_t1024_ab1_diagnostic", "coop": "multidag_coop_t1024_v0",
           "track": "multidag_track_t1024_v0", "dcn": "multidag_dcn_t1024_v0"}
RUN_ARTIFACTS = {"manifests/run_manifest.json": "run_manifest", "reports/validation_best_metrics.json": "best_dev_metrics",
                 "logs/epoch_metrics.tsv": "epoch_log", "resolved_config.yaml": "resolved_config"}
DEV_ARTIFACTS = (
    "reports/validation_best_dev_evidence.json", "predictions/validation_best_predictions.tsv",
    "predictions/validation_best_predictions.csv", "predictions/validation_best_predictions.json",
    "reports/validation_best_predictions.csv", "reports/validation_best_predictions.json",
    "reports/validation_best_per_class_metrics.tsv", "reports/validation_best_per_class_metrics.csv",
    "reports/validation_best_per_class_metrics.json", "reports/validation_best_confusion_matrix.tsv",
    "reports/validation_best_confusion_matrix.csv", "reports/validation_best_confusion_matrix.json",
)
RESEARCH_DOCS = (
    "docs/research/dag_core_variants/GRAPHCORE_ARCHITECTURE_DECISION.md",
    "docs/research/dag_core_variants/IMPLEMENTATION_REPORT.md",
    "docs/research/dag_core_variants/PAPER_TO_MULTIDAG_MAPPING.md",
    "docs/research/tav_full_audit_v1/TAV_FULL_AUDIT_SOURCE_AUDIT.md",
    "docs/research/tav_full_audit_v1/TAV_FULL_AUDIT_IMPLEMENTATION_REPORT.md",
    "docs/research/tav_full_audit_v1/AUDIO_FINGERPRINT_BRIDGE_IMPLEMENTATION_REPORT.md",
    "docs/reproduction/TA_V0_Q2_0F_FORMAL_RUNTIME_REPORT.md",
)
TEXT_EXTENSIONS = {".json", ".csv", ".tsv", ".yaml", ".md", ".txt"}
MAX_LIGHTWEIGHT_BYTES = 8 * 1024 * 1024


def identity(stage, family, *, model="", modalities="", cl="", seed=""):
    return {"stage": stage, "experiment_family": family, "model": model, "modalities": modalities, "CL": cl, "seed": seed}


class Evidence:
    def __init__(self, root):
        self.root = root
        self.index = []
        self.dev_coverage = []

    def record(self, relative, metadata, kind):
        self.index.append({**metadata, "artifact_type": kind, "relative_path": relative,
                           "sha256": sha256_file(self.root / relative)})

    def copy(self, source, relative, metadata, kind):
        source = Path(source)
        if ("\n" in relative or "\r" in relative or "\\" in relative or source.is_symlink()
                or not source.is_file() or source.suffix.lower() not in TEXT_EXTENSIONS
                or source.stat().st_size > MAX_LIGHTWEIGHT_BYTES):
            raise ValueError(f"expected a lightweight regular evidence file: {source}")
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as original, target.open("xb") as destination:
            shutil.copyfileobj(original, destination)
        if sha256_file(target) != sha256_file(source):
            raise ValueError(f"source changed while collecting: {source}")
        self.record(relative, metadata, kind)

    def csv(self, relative, rows, metadata, kind):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        write_csv(target, rows)
        self.record(relative, metadata, kind)

    def text(self, relative, value, metadata, kind):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
        self.record(relative, metadata, kind)

    def run(self, source, stage, key, metadata, manifest):
        for relative, kind in RUN_ARTIFACTS.items():
            self.copy(source / relative, f"{stage}/runs/{key}/{relative}", metadata, kind)
        bucket = "manifests/curriculum_bucket_manifest.tsv"
        if (source / bucket).is_file():
            self.copy(source / bucket, f"{stage}/runs/{key}/{bucket}", metadata, "curriculum_manifest")
        found = []
        for relative in DEV_ARTIFACTS:
            path = source / relative
            if path.is_file():
                if relative.endswith("validation_best_dev_evidence.json"):
                    report = read_json(path)
                    test_hold(report)
                    if (report.get("split") != "dev" or report.get("optimizer_steps_performed") != 0
                            or report.get("checkpoint_sha256") != manifest["best_checkpoint_sha256"]
                            or report.get("metrics") != manifest["best_dev_metrics"]):
                        raise ValueError(f"Dev export identity mismatch: {path}")
                self.copy(path, f"{PRED}/{stage}/{key}/{relative}", metadata, "dev_predictions_and_per_class")
                found.append(relative)
        self.dev_coverage.append({**metadata, "experiment_id": key, "artifact_count": len(found),
                                  "artifacts": ",".join(found), "status": "available" if found else "unavailable"})


def archive_ta(evidence, inputs):
    runs = inputs["runs"]
    if set(runs) != set(TA_RUNS):
        raise ValueError("historical inputs must identify t768, t1024, coop, track and dcn runs")
    ta_rows = []
    for key, model in TA_RUNS.items():
        source = Path(runs[key])
        manifest = read_json(source / "manifests/run_manifest.json")
        checked_completion(source, manifest)
        config = yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))
        best = read_json(source / "reports/validation_best_metrics.json")
        checkpoint = manifest["best_checkpoint_sha256"]
        if (manifest["model_lineage"] != model or manifest.get("seed") != 100
                or config["model"]["text_dim"] != (768 if key == "t768" else 1024)
                or config["checkpoint"]["test_split_used_for_selection"] is not False
                or config["training"]["curriculum"] is not False or best.get("checkpoint_identity") != checkpoint
                or {name: value for name, value in best.items() if name != "checkpoint_identity"} != manifest["best_dev_metrics"]
                or sha256_file(source / "checkpoints/best_model.pt") != checkpoint):
            raise ValueError(f"historical TA run identity mismatch: {source}")
        checked_epochs(source, manifest["best_epoch"])
        stage = BASELINE if key in ("t768", "t1024") else SCREEN
        metadata = identity(stage, key, model=model, modalities="true_TA", cl="CL_OFF", seed=100)
        evidence.run(source, stage, key, metadata, manifest)
        if key == "t768":
            source_config = PARENT_CONFIG.replace("formal_train_dev_t1024_ab1.yaml", "formal_train_dev.yaml")
        else:
            source_config = PARENT_CONFIG if key == "t1024" else config_path(key)
        evidence.copy(ROOT / source_config, f"{CONFIG}/true_ta/{key}.yaml", metadata, "source_config")
        count = manifest.get("model_parameter_count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"missing TA parameter count: {source}")
        ta_rows.append({**metadata, "run_id": manifest["run_id"], **metric_columns(best), "best_epoch": manifest["best_epoch"],
                        "parameter_count": count, "checkpoint_sha256": checkpoint,
                        "asset_sha256": manifest["data_identity"]["sha256"]["asset"],
                        "evidence_run_dir": f"{stage}/runs/{key}"})
    evidence.csv(f"{BASELINE}/T768_T1024_BASELINE_RECONCILIATION.csv", ta_rows[:2], identity(BASELINE, "reconciliation"), "summary")
    evidence.csv(f"{SCREEN}/TRUE_TA_GRAPHCORE_RESULTS.csv", ta_rows[1:], identity(SCREEN, "graphcore_screening"), "summary")
    for number, summary in enumerate(inputs.get("t768_summaries", []), 1):
        path = Path(summary)
        evidence.copy(path, f"{BASELINE}/reconciliation/{number}_{path.name}",
                      identity(BASELINE, "t768", model=TA_RUNS["t768"], modalities="true_TA", cl="CL_OFF", seed=100),
                      "historical_summary")
    return ta_rows[1]["asset_sha256"]


def archive_audit(evidence, batch_path, batch, training, missing):
    if (set(batch["completed"] + batch["skipped"]) != {row["experiment_id"] for row in training}
            or len(batch["completed"] + batch["skipped"]) != 24
            or sorted(batch["evaluated"]) != sorted(row["evaluation_id"] for row in missing)
            or batch.get("matrix_identity") != json_identity({"training": batch["training_jobs"], "evaluation": batch["evaluation_jobs"]})):
        raise ValueError("Full Audit completion ledger is inconsistent")
    load_matrices()
    metadata = identity(AUDIT, "tav_full_audit")
    evidence.copy(batch_path, f"{AUDIT}/batch_manifest.json", metadata, "batch_manifest")
    for name in (TRAIN_MATRIX, EVAL_MATRIX):
        evidence.copy(ROOT / CONFIG_ROOT / name, f"{CONFIG}/full_audit/{name}", metadata, "matrix")
    by_id = {job["experiment_id"]: job for job in training}
    for row in training:
        source = Path(row["run_dir"])
        run_id = row["experiment_id"]
        manifest = read_json(source / "manifests/run_manifest.json")
        checked_completion(source, manifest)
        raw = validate_config(yaml.safe_load((source / "resolved_config.yaml").read_text(encoding="utf-8"))["audit_config"])
        if json_identity(raw) != row["config_sha256"] or row["status"] != "completed":
            raise ValueError(f"Full Audit resolved config/status mismatch: {run_id}")
        checked_epochs(source, manifest["best_epoch"])
        run = identity(AUDIT, row["comparison_family"], model=f"{row['model_name']}/{row['graph_core']}",
                       modalities=row["modalities"], cl=row["curriculum"], seed=100)
        evidence.run(source, AUDIT, run_id, run, manifest)
        evidence.copy(ROOT / row["config_path"], f"{CONFIG}/full_audit/{Path(row['config_path']).name}", run, "source_config")
    evaluation_root = batch_path.parents[3] / "analysis" / "batches" / batch["batch_id"] / "missing_modality"
    for row in missing:
        artifact = evaluation_root / f"{row['evaluation_id']}.json"
        report = read_json(artifact)
        source = by_id[row["source_experiment_id"]]
        if (row["status"] != "completed" or report["split"] != "dev" or report["metrics"]["valid_target_count"] != 980
                or Path(report["source_run_dir"]).resolve() != Path(source["run_dir"]).resolve()
                or row["best_epoch"] != source["best_epoch"] or row["parameter_count"] != source["parameter_count"]):
            raise ValueError(f"Full Audit Dev mask evidence mismatch: {artifact}")
        mask = identity(AUDIT, "dev_missing_modality", model=f"{source['model_name']}/{row['graph_core']}",
                        modalities=row["active_modalities"], cl=row["curriculum"], seed=100)
        evidence.copy(artifact, f"{AUDIT}/missing_modality/{artifact.name}", mask, "dev_mask_evaluation")
    results = result_tables(training, missing)
    if tuple(map(len, results.values())) != (24, 8, 20, 14, 5, 56):
        raise ValueError("Full Audit table row counts mismatch")
    for name, rows in results.items():
        evidence.csv(f"{AUDIT}/{name}", rows, metadata, "result_table")


def archive_confirm(evidence, batch_path, batch, seedwise):
    metadata = identity(CONFIRM, "tav_multiseed_confirm")
    for name, rows in tables(seedwise).items():
        evidence.csv(f"{CONFIRM}/{name}", rows, metadata, "result_table")
    evidence.copy(batch_path, f"{CONFIRM}/batch_manifest.json", metadata, "batch_manifest")
    evidence.copy(ROOT / MATRIX_PATH, f"{CONFIG}/{MATRIX_PATH.name}", metadata, "matrix")
    for job in batch["training_jobs"]:
        key = job["experiment_id"]
        source = Path(job["run_dir"])
        run = identity(CONFIRM, job["configuration"], model=f"{job['model']}/{job['graph_core']}",
                       modalities=job["active_modalities"], cl=job["curriculum"], seed=job["seed"])
        evidence.run(source, CONFIRM, key, run, read_json(source / "manifests/run_manifest.json"))


def archive_data(evidence, batch, files, ta_sha, *, subdir=""):
    model, provenance, bridge = (Path(files[key]) for key in ("model_manifest", "provenance_manifest", "bridge_audit"))
    recorded = batch["data_identity"]["sha256"]
    prov, audit = read_json(provenance), read_json(bridge)
    test_hold(prov)
    test_hold(audit)
    source_hashes = {key: source["sha256"] for key, source in prov["sources"].items()}
    required = {"canonical_asset", "historical_asset", "model_manifest", "order_manifest", "historical_split_manifest", "fingerprint_bridge"}
    if (sha256_file(model) != recorded["model_manifest"] or sha256_file(provenance) != recorded["provenance_manifest"]
            or prov["model_manifest_sha256"] != recorded["model_manifest"] or prov["asset_sha256"] != recorded["asset"]
            or prov["schema"] != batch["data_identity"]["schema"] or prov["status"] != "complete"
            or prov["row_order_identity_sha256"] != batch["data_identity"]["row_order_identity_sha256"]
            or audit["schema"] != "audio_fingerprint_bridge_v1" or audit["status"] != "complete"
            or audit["bridge_rows"] != 7433 or source_hashes.keys() != required
            or not all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in (*source_hashes.values(), recorded["asset"]))
            or source_hashes["canonical_asset"] != ta_sha or source_hashes["fingerprint_bridge"] != audit["bridge_sha256"]
            or any(audit["sources"][key]["sha256"] != source_hashes[key] for key in required - {"fingerprint_bridge"})):
        raise ValueError("TA / TAV / fingerprint bridge provenance identity mismatch")
    destination = f"{DATA}/{subdir}" if subdir else DATA
    metadata = identity(DATA, "canonical_ta_tav", modalities="TAV")
    evidence.copy(model, f"{destination}/model_input_manifest_tav_t1024_v1.csv", metadata, "model_manifest")
    evidence.copy(provenance, f"{destination}/tav_t1024_provenance_manifest_v1.json", metadata, "asset_provenance")
    evidence.copy(bridge, f"{destination}/audio_fingerprint_bridge_t1024_v1.audit.json", metadata, "audio_bridge_audit")
    evidence.text(f"{destination}/CANONICAL_ASSET_IDENTITIES.json",
                  json.dumps({"tav": recorded, "source_sha256": source_hashes}, indent=2, sort_keys=True) + "\n",
                  metadata, "asset_sha256_identities")


def inventory(evidence, batch):
    meta = identity(INVENTORY, "stage_evidence")
    evidence.csv(f"{PRED}/DEV_EVIDENCE_COVERAGE.csv", evidence.dev_coverage, meta, "coverage")
    unavailable = sum(row["status"] == "unavailable" for row in evidence.dev_coverage)
    evidence.text(f"{INVENTORY}/STAGE_SUMMARY.md", f"""# GraphCore TAV stage evidence

- Confirmatory Dev results cover 40 identities across 8 configurations and 5 seeds.
- This batch reused {batch['reused_seed100']} seed100 runs and executed {batch['new_runs']} new runs.
- Full Audit evidence covers 24 completed trainings and 56 completed Dev mask evaluations.
- Test loaders and evaluations were both zero. Checkpoint selection used Dev Weighted-F1, Dev loss and epoch in that order.
- TA-active uses the TAV architecture with a deterministic zero mask after the V encoder. The screening runs use true TA.
- Summary scores are proportions. The paired differences and 95% Student-t intervals use percentage points and five matched seeds.
- Dev prediction/per-class coverage is listed in `05_predictions_and_perclass/DEV_EVIDENCE_COVERAGE.csv`. {unavailable} run entries have no available lightweight artifact. The collector runs no inference.
- `FILELIST.txt` includes every file. `EVIDENCE_INDEX.csv` covers the evidence and file list. `SHA256SUMS.txt` covers the index and all other files except the checksum file itself.
""", meta, "stage_summary")
    entries = sorted([row["relative_path"] for row in evidence.index] +
                     [f"{INVENTORY}/{name}" for name in ("FILELIST.txt", "EVIDENCE_INDEX.csv", "SHA256SUMS.txt")])
    evidence.text(f"{INVENTORY}/FILELIST.txt", "\n".join(entries) + "\n", meta, "file_list")
    index_path = evidence.root / INVENTORY / "EVIDENCE_INDEX.csv"
    write_csv(index_path, sorted(evidence.index, key=lambda row: row["relative_path"]))
    hashes = {row["relative_path"]: row["sha256"] for row in evidence.index}
    hashes[f"{INVENTORY}/EVIDENCE_INDEX.csv"] = sha256_file(index_path)
    checksum_path = f"{INVENTORY}/SHA256SUMS.txt"
    (evidence.root / checksum_path).write_text("".join(f"{hashes[name]}  {name}\n" for name in sorted(hashes)), encoding="utf-8")
    hashes[checksum_path] = sha256_file(evidence.root / checksum_path)
    if entries != sorted(hashes):
        raise ValueError("evidence inventory coverage mismatch")
    return hashes


def verify_archive(archive, package_name, hashes):
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        expected = {f"{package_name}/{relative}": sha for relative, sha in hashes.items()}
        if (len(members) != len(expected) or {member.name for member in members} != expected.keys()
                or any(not member.isfile() for member in members)):
            raise ValueError("archive entries do not match the allowlisted inventory")
        for member in members:
            digest = hashlib.sha256()
            with bundle.extractfile(member) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected[member.name]:
                raise ValueError(f"archive extraction SHA256 mismatch: {member.name}")


def collect(*, confirm_batch_manifest, full_audit_batch_manifest, historical_inputs, model_manifest_path,
            provenance_manifest_path, bridge_audit_path, output_root="outputs", batch_id=None, package_date=None):
    confirm_path, audit_path = Path(confirm_batch_manifest), Path(full_audit_batch_manifest)
    confirm, seedwise = collect_confirm(confirm_path)
    reference = confirm["historical_batch_manifest"]
    if reference and reference["sha256"] != sha256_file(audit_path):
        raise ValueError("provided Full Audit batch differs from confirmation reuse provenance")
    historical_jobs(audit_path)
    audit, training, missing = collect_audit(audit_path)
    inputs = read_json(historical_inputs)
    package_date = validate_experiment_date(package_date or confirm["experiment_date"])
    name = f"GRAPHCORE_TAV_STAGE_FINAL_EVIDENCE_{package_date}"
    parent = resolve_output_paths(output_base=output_root, experiment_date=confirm["experiment_date"],
                                  experiment_group=CONFIRM_GROUP, batch_id=batch_id or confirm["batch_id"]).report_root
    parent.mkdir(parents=True, exist_ok=True)
    archive = parent / (name + ".tar.gz")
    lock = parent / (name + ".lock")
    with lock.open("x"):
        pass
    try:
        if archive.exists() or archive.is_symlink():
            raise FileExistsError(f"evidence package already exists: {archive}")
        workspace = parent.resolve() / (".e" + uuid4().hex[:8])
        workspace.mkdir()
        try:
            evidence = Evidence(workspace)
            ta_sha = archive_ta(evidence, inputs)
            archive_audit(evidence, audit_path, audit, training, missing)
            archive_confirm(evidence, confirm_path, confirm, seedwise)
            archive_data(evidence, confirm, {"model_manifest": model_manifest_path,
                                            "provenance_manifest": provenance_manifest_path,
                                            "bridge_audit": bridge_audit_path}, ta_sha)
            if audit["data_identity"] != confirm["data_identity"]:
                archive_data(evidence, audit, inputs["full_audit_data"], ta_sha, subdir="historical_full_audit")
            evidence.copy(ROOT / SOURCE, f"{CONFIG}/tav_frontend.yaml", identity(CONFIG, "tav_frontend", modalities="TAV"), "frontend_config")
            for path in RESEARCH_DOCS:
                evidence.copy(ROOT / path, f"{DOCS}/{Path(path).name}", identity(DOCS, "research_provenance"), "research_doc")
            hashes = inventory(evidence, confirm)
            temporary_archive = workspace / ".package.tar.gz"
            with tarfile.open(temporary_archive, "x:gz") as bundle:
                for relative in sorted(hashes):
                    bundle.add(workspace / relative, arcname=f"{name}/{relative}", recursive=False)
            verify_archive(temporary_archive, name, hashes)
            digest = sha256_file(temporary_archive)
            temporary_archive.rename(archive)
        finally:
            if workspace.resolve() != workspace or workspace.parent != parent.resolve():
                raise ValueError("evidence staging directory escaped its output root")
            shutil.rmtree(workspace)
    finally:
        lock.unlink()
    return {"MULTISEED_CONFIRM": "COMPLETE", "EXPECTED_IDENTITIES": 40, "COMPLETE_IDENTITIES": len(seedwise),
            "REUSED_SEED100": confirm["reused_seed100"], "NEW_RUNS": confirm["new_runs"], "TEST_EVALUATION_COUNT": 0,
            "EVIDENCE_PACKAGE": archive.as_posix(), "EVIDENCE_FILE_COUNT": len(hashes), "EVIDENCE_PACKAGE_SHA256": digest,
            "READY_FOR_GPT_FINAL_STAGE_ANALYSIS": "YES"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("confirm-batch-manifest", "full-audit-batch-manifest", "historical-inputs", "model-manifest-path",
                 "provenance-manifest-path", "bridge-audit-path"):
        parser.add_argument("--" + name, required=True, help=("JSON with runs {t768,t1024,coop,track,dcn}; optional "
                            "t768_summaries and full_audit_data" if name == "historical-inputs" else None))
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--batch-id")
    parser.add_argument("--package-date")
    print(json.dumps(collect(**vars(parser.parse_args(argv))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
