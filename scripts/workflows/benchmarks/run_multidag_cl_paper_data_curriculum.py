"""Run the four pending Paper Table 4 curriculum ablations serially."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Sequence


PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "AGENTS.md").is_file() and (parent / "scripts").is_dir()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.output_paths import (  # noqa: E402
    resolve_experiment_date,
    resolve_output_paths,
    validate_batch_id,
    validate_experiment_group,
)


EXPERIMENT_GROUP = "multidag_cl_paper_data_curriculum"
BATCH_PREFIX = "multidag_cl_paper_data_curriculum"
ENTRYPOINT = Path("scripts/models/multidag_cl/paper_reimplementation/train.py")
CONFIG_ROOT = Path(
    "configs/multidag_cl/paper_reimplementation/iemocap/full_context/"
    "paper_data_reproduction/curriculum_comparison"
)
RUN_PLAN = (
    ("CL4", CONFIG_ROOT / "cl4.yaml"),
    ("CL7", CONFIG_ROOT / "cl7.yaml"),
    ("CL10", CONFIG_ROOT / "cl10.yaml"),
    ("CL15", CONFIG_ROOT / "cl15.yaml"),
)
SUMMARY_FIELDS = ("experiment", "config", "exit_code", "status", "run_id")
RESULT_PREFIX = "MULTIDAG_STAGE_B3_RESULT="


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-date", default=None)
    parser.add_argument("--experiment-group", default=EXPERIMENT_GROUP)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-id", default=None)
    return parser.parse_args(argv)


def build_command(
    config_path: Path,
    *,
    output_root: Path,
    experiment_date: str,
    experiment_group: str,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        ENTRYPOINT.as_posix(),
        "--mode",
        "train",
        "--config",
        config_path.as_posix(),
        "--output-root",
        output_root.as_posix(),
        "--experiment-date",
        experiment_date,
        "--experiment-group",
        experiment_group,
        "--device",
        device,
    ]


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_tsv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in SUMMARY_FIELDS} for row in rows
        )
    temporary.replace(path)


def _extract_result(log_path: Path) -> dict[str, Any] | None:
    result: dict[str, Any] | None = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(RESULT_PREFIX):
            continue
        try:
            candidate = json.loads(line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            result = candidate
    return result


def _run_one(
    command: Sequence[str],
    log_path: Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]],
) -> tuple[int, dict[str, Any] | None]:
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = runner(
            list(command),
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode), _extract_result(log_path)


def launch_batch(
    args: argparse.Namespace,
    *,
    runner: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
    now_provider: Callable[[], datetime] = datetime.now,
) -> dict[str, Any]:
    experiment_date = resolve_experiment_date(cli_date=args.experiment_date)
    experiment_group = validate_experiment_group(args.experiment_group)
    launched_at = now_provider()
    batch_id = validate_batch_id(
        args.batch_id
        or f"{BATCH_PREFIX}_{launched_at.strftime('%Y%m%d_%H%M%S')}"
    )
    output_root = Path(args.output_root)
    layout = resolve_output_paths(
        output_base=output_root,
        experiment_date=experiment_date,
        experiment_group=experiment_group,
        batch_id=batch_id,
    )
    for directory in (
        layout.launcher_log_root,
        layout.manifest_root,
        layout.report_root,
    ):
        directory.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    manifest_path = layout.manifest_root / "run_manifest.tsv"
    for index, (experiment, config_path) in enumerate(RUN_PLAN, start=1):
        if not (PROJECT_ROOT / config_path).is_file():
            raise FileNotFoundError(f"curriculum config does not exist: {config_path}")
        command = build_command(
            config_path,
            output_root=output_root,
            experiment_date=experiment_date,
            experiment_group=experiment_group,
            device=args.device,
        )
        log_path = layout.launcher_log_root / f"{index:02d}_{experiment.lower()}.log"
        try:
            exit_code, result = _run_one(command, log_path, runner=runner)
        except OSError as error:
            with log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(f"launcher_error={error}\n")
            exit_code, result = 127, None
        result_status = "" if result is None else str(result.get("status", ""))
        status = "PASS" if exit_code == 0 and result_status == "PASS" else "FAILED"
        rows.append(
            {
                "experiment": experiment,
                "config": config_path.as_posix(),
                "exit_code": exit_code,
                "status": status,
                "run_id": "" if result is None else str(result.get("run_id", "")),
            }
        )
        _atomic_tsv(manifest_path, rows)

    summary = {
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "COMPLETED_WITH_FAILURES",
        "batch_id": batch_id,
        "experiment_date": experiment_date,
        "experiment_group": experiment_group,
        "started_at": launched_at.astimezone().isoformat(),
        "finished_at": now_provider().astimezone().isoformat(),
        "sequential": True,
        "continue_on_error": True,
        "experiments": rows,
    }
    _atomic_json(layout.report_root / "final_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = launch_batch(parse_args(argv))
    print("MULTIDAG_CURRICULUM_BATCH_RESULT=" + json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
