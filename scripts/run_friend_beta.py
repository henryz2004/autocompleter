#!/usr/bin/env python3
"""Preflight and launch the friend beta from the repo-local venv."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / "venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
ENV_PATH = REPO_ROOT / ".env"
ENV_EXAMPLE_PATH = REPO_ROOT / ".env.example"


def _print(msg: str) -> None:
    print(f"[friend-beta] {msg}")


def _shell_name() -> str:
    term_program = os.environ.get("TERM_PROGRAM", "").strip()
    mapping = {
        "Apple_Terminal": "Terminal",
        "iTerm.app": "iTerm",
        "WarpTerminal": "Warp",
        "vscode": "Visual Studio Code terminal",
    }
    return mapping.get(term_program, "your terminal app")


def _ensure_repo_python() -> bool:
    if VENV_PYTHON.exists():
        return True
    _print("Missing ./venv/bin/python. Run `make friend-beta-bootstrap` first.")
    return False


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _ensure_env_file() -> bool:
    if ENV_PATH.exists():
        return True
    shutil.copyfile(ENV_EXAMPLE_PATH, ENV_PATH)
    _print("Created ./.env from ./.env.example")
    _print("Fill in your beta proxy URL, install id, and install key, then rerun this command.")
    return False


def _validate_env() -> bool:
    env = _parse_env_file(ENV_PATH)
    proxy_enabled = env.get("AUTOCOMPLETER_PROXY_ENABLED", "").lower() in {"1", "true"}
    if not proxy_enabled:
        return True

    required = [
        "AUTOCOMPLETER_PROXY_BASE_URL",
        "AUTOCOMPLETER_PROXY_API_KEY",
        "AUTOCOMPLETER_TELEMETRY_URL",
        "AUTOCOMPLETER_INSTALL_ID",
    ]
    missing = [key for key in required if not env.get(key, "").strip()]
    if not missing:
        return True

    _print("Your ./.env is missing beta settings:")
    for key in missing:
        print(f"  - {key}")
    _print("Paste in the values from your beta invite and rerun this command.")
    return False


def _check_accessibility() -> bool:
    probe = subprocess.run(
        [
            str(VENV_PYTHON),
            "-c",
            (
                "from ApplicationServices import AXIsProcessTrusted; "
                "import sys; sys.exit(0 if AXIsProcessTrusted() else 1)"
            ),
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return probe.returncode == 0


def _print_accessibility_help() -> None:
    host_app = _shell_name()
    _print("Accessibility permission is required before Autocompleter can read or write text.")
    print("  1. Open System Settings > Privacy & Security > Accessibility")
    print(f"  2. Enable access for {host_app}")
    print("  3. Quit and reopen that terminal app if macOS does not pick it up immediately")
    print("  4. Rerun this command")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preflight and launch the friend beta from this repo.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Validate the local beta setup without launching the app.",
    )
    parser.add_argument(
        "--skip-accessibility-check",
        action="store_true",
        help="Launch even if the preflight accessibility probe fails.",
    )
    parser.add_argument(
        "autocompleter_args",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to `python -m autocompleter`.",
    )
    args = parser.parse_args()

    if not _ensure_repo_python():
        return 1
    if not _ensure_env_file():
        return 1
    if not _validate_env():
        return 1

    if args.bootstrap_only:
        _print("Bootstrap complete.")
        return 0

    if not args.skip_accessibility_check and not _check_accessibility():
        _print_accessibility_help()
        return 1

    extra_args = list(args.autocompleter_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd = [str(VENV_PYTHON), "-m", "autocompleter", *extra_args]
    _print(f"Launching: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
