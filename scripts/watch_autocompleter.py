#!/usr/bin/env python3
"""Watch local files and restart autocompleter with debug-friendly defaults."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

WATCH_DIRS = (
    "autocompleter",
    "tests",
    "scripts",
)

WATCH_FILES = (
    ".env",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "AGENTS.md",
    "CLAUDE.md",
    "COMMANDS.md",
    "dump_ax_tree.py",
    "dump_ax_tree_json.py",
    "dump_pipeline.py",
)

SKIP_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
    "venv",
    "__pycache__",
    "dumps",
}

WATCH_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".txt",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
}


def iter_watch_paths(root: Path) -> list[Path]:
    paths: list[Path] = []

    for rel in WATCH_FILES:
        path = root / rel
        if path.exists():
            paths.append(path)

    for rel_dir in WATCH_DIRS:
        base = root / rel_dir
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [
                name for name in dirnames
                if name not in SKIP_DIR_NAMES and not name.startswith(".")
            ]
            current_dir = Path(dirpath)
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = current_dir / filename
                if path.suffix in WATCH_SUFFIXES:
                    paths.append(path)

    # Stable order helps produce deterministic change messages.
    return sorted(set(paths))


def build_snapshot(root: Path) -> dict[Path, tuple[int, int]]:
    snapshot: dict[Path, tuple[int, int]] = {}
    for path in iter_watch_paths(root):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        snapshot[path] = (stat.st_mtime_ns, stat.st_size)
    return snapshot


def detect_changes(
    old: dict[Path, tuple[int, int]],
    new: dict[Path, tuple[int, int]],
) -> list[Path]:
    changed: list[Path] = []
    all_paths = set(old) | set(new)
    for path in sorted(all_paths):
        if old.get(path) != new.get(path):
            changed.append(path)
    return changed


def format_paths(root: Path, paths: list[Path], limit: int = 8) -> str:
    rels = [str(path.relative_to(root)) for path in paths[:limit]]
    if len(paths) > limit:
        rels.append(f"... (+{len(paths) - limit} more)")
    return ", ".join(rels)


def terminate_process(proc: subprocess.Popen[bytes] | None, timeout_s: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.kill()
    proc.wait(timeout=timeout_s)


def start_process(
    python_bin: str,
    root: Path,
    log_file: Path,
    dump_dir: Path,
    extra_args: list[str],
) -> subprocess.Popen[bytes]:
    cmd = [
        python_bin,
        "-m",
        "autocompleter",
        "--log-file",
        str(log_file),
        "--log-level",
        "DEBUG",
        "--dump-dir",
        str(dump_dir),
        *extra_args,
    ]
    print(f"[watch] starting: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(cmd, cwd=root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Watch local files and restart autocompleter on changes.",
    )
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / "venv" / "bin" / "python"),
        help="Python interpreter to use. Defaults to repo venv python.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds. Default: 0.5",
    )
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=300,
        help="Debounce window before restart after a change. Default: 300",
    )
    parser.add_argument(
        "--log-file",
        default=str(REPO_ROOT / "dumps" / "watch" / "autocompleter.log"),
        help="Debug log file path for the child process.",
    )
    parser.add_argument(
        "--dump-dir",
        default=str(REPO_ROOT / "dumps" / "watch"),
        help="Dump directory for trigger snapshots/fixtures.",
    )
    parser.add_argument(
        "autocompleter_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to `python -m autocompleter`.",
    )
    args = parser.parse_args()

    python_bin = args.python
    log_file = Path(args.log_file).expanduser()
    dump_dir = Path(args.dump_dir).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if not Path(python_bin).exists():
        print(f"[watch] python not found: {python_bin}", file=sys.stderr)
        return 1

    extra_args = list(args.autocompleter_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    child: subprocess.Popen[bytes] | None = None
    snapshot = build_snapshot(REPO_ROOT)
    stop = False

    def handle_signal(signum, frame):  # type: ignore[unused-argument]
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        child = start_process(
            python_bin=python_bin,
            root=REPO_ROOT,
            log_file=log_file,
            dump_dir=dump_dir,
            extra_args=extra_args,
        )

        while not stop:
            time.sleep(args.poll_interval)

            if child is not None and child.poll() is not None:
                print(
                    f"[watch] child exited with code {child.returncode}; waiting for file changes to restart",
                    flush=True,
                )
                child = None

            new_snapshot = build_snapshot(REPO_ROOT)
            changed = detect_changes(snapshot, new_snapshot)
            if not changed:
                continue

            # Give editors/build tools a short window to finish writing.
            settle_s = max(args.settle_ms, 0) / 1000.0
            if settle_s:
                time.sleep(settle_s)
                new_snapshot = build_snapshot(REPO_ROOT)
                changed = detect_changes(snapshot, new_snapshot)
                if not changed:
                    continue

            print(
                f"[watch] change detected: {format_paths(REPO_ROOT, changed)}",
                flush=True,
            )
            snapshot = new_snapshot
            terminate_process(child)
            child = start_process(
                python_bin=python_bin,
                root=REPO_ROOT,
                log_file=log_file,
                dump_dir=dump_dir,
                extra_args=extra_args,
            )
    finally:
        terminate_process(child)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
