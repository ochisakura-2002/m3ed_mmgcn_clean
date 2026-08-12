"""Validate official MultiDAG-CL assets and import them into project artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.build.multidag_cl.convert_official_json_to_project_pkl import (
    convert_official_directory_to_project_pkl,
)
from scripts.data.build.multidag_cl.inspect_official_assets import (
    OfficialAssetsUnavailable,
    inspect_official_assets,
)
from scripts.data.build.multidag_cl.schema import (
    OFFICIAL_ASSET_MANIFEST_FILENAME,
    OFFICIAL_SPLIT_MANIFEST_FILENAME,
    PROJECT_PKL_FILENAME,
    SPLIT_ORDER,
    atomic_write_json,
    compute_sha256,
)


def import_official_assets(
    *,
    official_root: Path,
    output_dir: Path,
    fail_if_exists: bool = True,
) -> dict[str, Any]:
    """Run inspect -> validate -> structural conversion with no feature transform."""

    official_root = Path(official_root)
    output_dir = Path(output_dir)
    asset_manifest_path = output_dir / OFFICIAL_ASSET_MANIFEST_FILENAME
    split_manifest_path = output_dir / OFFICIAL_SPLIT_MANIFEST_FILENAME
    project_pkl_path = output_dir / PROJECT_PKL_FILENAME
    destinations = [asset_manifest_path, split_manifest_path, project_pkl_path]
    if fail_if_exists:
        existing = [path for path in destinations if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite imported artifacts: {existing}")

    # Inspection is performed before the output directory is created, so a
    # missing source cannot leave behind apparent imported artifacts.
    asset_manifest = inspect_official_assets(
        official_root=official_root,
        manifest_output=None,
    )
    conversion = convert_official_directory_to_project_pkl(
        official_dir=official_root,
        output_pkl=project_pkl_path,
        split_manifest_output=split_manifest_path,
        fail_if_exists=fail_if_exists,
    )
    asset_manifest["layer2"] = {
        "project_pkl_filename": project_pkl_path.name,
        "project_pkl_sha256": conversion["project_pkl_sha256"],
        "split_manifest_filename": split_manifest_path.name,
        "split_manifest_sha256": compute_sha256(split_manifest_path),
        "conversion": "STRUCTURAL_ONLY",
    }
    atomic_write_json(asset_manifest_path, asset_manifest)

    counts = {
        split: {
            "dialogues": asset_manifest["splits"][split]["dialogue_count"],
            "utterances": asset_manifest["splits"][split]["utterance_count"],
        }
        for split in SPLIT_ORDER
    }
    return {
        "status": "PASS",
        "official_root": str(official_root),
        "output_dir": str(output_dir),
        "official_asset_manifest": str(asset_manifest_path),
        "official_split_manifest": str(split_manifest_path),
        "project_pkl": str(project_pkl_path),
        "project_pkl_sha256": conversion["project_pkl_sha256"],
        "counts": counts,
        "dimensions": asset_manifest["dimensions"],
        "exact_split_preserved": True,
        "feature_value_transform": "NONE",
        "normalization": "NONE",
        "re_extraction": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = parse_args()
    try:
        result = import_official_assets(
            official_root=args.official_root,
            output_dir=args.output_dir,
            fail_if_exists=not args.allow_overwrite,
        )
    except OfficialAssetsUnavailable as error:
        result = {
            "status": "BLOCKED_OFFICIAL_ASSETS",
            "official_root": str(args.official_root),
            "output_dir": str(args.output_dir),
            "project_pkl_created": False,
            "error": str(error),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
