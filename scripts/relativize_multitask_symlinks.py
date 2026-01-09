from __future__ import annotations

import argparse
import os
from pathlib import Path


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite absolute symlinks under data/multitask into portable relative "
            "symlinks pointing to files within this repo (typically data/raw/*)."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repo root (default: inferred from this script location).",
    )
    parser.add_argument(
        "--multitask-dir",
        type=Path,
        default=None,
        help="Override multitask dataset dir (default: <repo-root>/data/multitask).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print changes only.")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    multitask_dir = (args.multitask_dir or (repo_root / "data" / "multitask")).resolve()

    if not multitask_dir.exists():
        raise SystemExit(f"Not found: {multitask_dir}")

    total = 0
    changed = 0
    unchanged = 0
    skipped_outside = 0
    skipped_missing = 0

    for path in multitask_dir.rglob("*"):
        if not path.is_symlink():
            continue

        total += 1
        raw_target = os.readlink(path)

        if os.path.isabs(raw_target):
            abs_target = Path(raw_target)
        else:
            abs_target = (path.parent / raw_target)

        resolved_target = abs_target.resolve(strict=False)
        if not resolved_target.exists():
            skipped_missing += 1
            continue

        if not _is_within(resolved_target, repo_root):
            skipped_outside += 1
            continue

        rel_target = os.path.relpath(resolved_target, start=path.parent)
        rel_target = Path(rel_target).as_posix()

        if raw_target == rel_target:
            unchanged += 1
            continue

        if args.dry_run:
            print(f"{path}: {raw_target} -> {rel_target}")
            changed += 1
            continue

        path.unlink()
        os.symlink(rel_target, path)
        changed += 1

    print(
        "Symlink rewrite summary:\n"
        f"  repo_root: {repo_root}\n"
        f"  multitask_dir: {multitask_dir}\n"
        f"  total_symlinks: {total}\n"
        f"  changed: {changed}\n"
        f"  unchanged: {unchanged}\n"
        f"  skipped_missing_target: {skipped_missing}\n"
        f"  skipped_outside_repo: {skipped_outside}\n"
        f"  dry_run: {args.dry_run}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

