"""Read-only identity guard for the three long-lived MERC workstreams."""

from __future__ import annotations

import argparse
import ntpath
import posixpath
import subprocess
import sys
from pathlib import Path


EXPECTED_REMOTE = "https://github.com/ochisakura-2002/m3ed_mmgcn_clean.git"
# User-approved deployment identities, not model/data/output configuration paths.
WORKSTREAMS = {
    "multidag": {
        "branch": "feat/multidag-cl-paper-reimplementation",
        "windows": r"E:\MERC\m3ed_mmgcn_clean",
        "ubuntu": "/home/zhiyuan/research_lh/m3ed_mmgcn_clean",
    },
    "s2d": {
        "branch": "feat/s2d-iemocap-v0",
        "windows": r"E:\MERC\worktrees\s2d-iemocap-v0",
        "ubuntu": "/home/zhiyuan/research_lh/worktrees/s2d-iemocap-v0",
    },
    "asymmetric_merc": {
        "branch": "integration/asymmetric-merc-v0",
        "windows": r"E:\MERC\worktrees\asymmetric-merc-v0",
        "ubuntu": "/home/zhiyuan/research_lh/worktrees/asymmetric-merc-v0",
    },
}


class GuardParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print("WORKSTREAM_CHECK=FAIL", flush=True)
        super().error(message)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def normalize_path(value: str, platform: str) -> str:
    if platform == "windows":
        return ntpath.normcase(ntpath.normpath(value))
    return posixpath.normpath(value)


def check_workstream(name: str) -> list[str]:
    if sys.platform == "win32":
        platform = "windows"
    elif sys.platform.startswith("linux"):
        platform = "ubuntu"
    else:
        raise ValueError(f"Unsupported platform: {sys.platform}")

    expected = WORKSTREAMS[name]
    root = git("rev-parse", "--show-toplevel")
    branch = git("symbolic-ref", "--quiet", "--short", "HEAD")
    fetch_urls = git("remote", "get-url", "--all", "origin").splitlines()
    push_urls = git("remote", "get-url", "--push", "--all", "origin").splitlines()
    print(f"WORKSTREAM={name.upper()}")
    print(f"PATH={root}")
    print(f"BRANCH={branch}")
    print(f"REMOTE={','.join(fetch_urls)}")
    print(f"REMOTE_PUSH={','.join(push_urls)}")

    errors = []
    if fetch_urls != [EXPECTED_REMOTE] or push_urls != [EXPECTED_REMOTE]:
        errors.append(f"origin fetch/push must each be exactly {EXPECTED_REMOTE}")
    if branch != expected["branch"]:
        errors.append(f"Expected branch: {expected['branch']}")
    if normalize_path(root, platform) != normalize_path(expected[platform], platform):
        errors.append(f"Expected worktree path: {expected[platform]}")

    script_root = str(Path(__file__).resolve().parents[2])
    if normalize_path(script_root, platform) != normalize_path(root, platform):
        errors.append("Run the guard from the current worktree's own script")

    registered = False
    for record in git("worktree", "list", "--porcelain").split("\n\n"):
        fields = dict(line.split(" ", 1) for line in record.splitlines() if " " in line)
        if normalize_path(fields.get("worktree", ""), platform) == normalize_path(root, platform):
            registered = fields.get("branch") == f"refs/heads/{expected['branch']}"
            break
    if not registered:
        errors.append("Current root/branch is not the expected registered worktree")

    # Reject an independent clone placed at a linked worktree's expected path.
    common_dir = str(Path(git("rev-parse", "--git-common-dir")).resolve())
    path_module = ntpath if platform == "windows" else posixpath
    expected_common = path_module.join(WORKSTREAMS["multidag"][platform], ".git")
    if normalize_path(common_dir, platform) != normalize_path(expected_common, platform):
        errors.append(f"Expected shared Git directory: {expected_common}")
    return errors


def main() -> int:
    parser = GuardParser(description=__doc__)
    parser.add_argument("workstream", choices=tuple(WORKSTREAMS))
    args = parser.parse_args()
    try:
        errors = check_workstream(args.workstream)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        errors = [str(exc)]
    for error in errors:
        print(f"ERROR={error}")
    print(f"WORKSTREAM_CHECK={'FAIL' if errors else 'PASS'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
