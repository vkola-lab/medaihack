"""
Lightweight run logger.

Two outputs per run:

  1. Per-run folder:        logs/<run_name>/
                              ├── config.json        (args + git + host + argv)
                              ├── epoch_log.csv      (train only, one row per epoch)
                              └── metrics.json       (final metrics)

  2. Global registry:       logs/runs.jsonl
                              one JSON line per run, grep / pandas friendly.

No external dependencies; append-only I/O.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def make_run_dir(
    run_type: str,
    model:    str,
    base:     str = "logs",
    run_name: str | None = None,
) -> Path:
    """Create `logs/<run_name>/` and return it.

    run_name defaults to `<UTC-stamp>_<model>_<run_type>`.
    """
    if run_name is None:
        run_name = f"{_stamp()}_{model}_{run_type}"
    run_dir = Path(base) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: dict) -> None:
    payload = {
        "timestamp":  _utcnow(),
        "git_commit": _git_commit(),
        "host":       socket.gethostname(),
        "python":     sys.version.split()[0],
        "argv":       sys.argv,
        **config,
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, default=str))


def write_metrics(run_dir: Path, metrics: dict) -> None:
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))


def append_epoch(run_dir: Path, row: dict) -> None:
    """Append one row to epoch_log.csv; write header on first call."""
    path = run_dir / "epoch_log.csv"
    new  = not path.exists()
    keys = list(row.keys())
    with path.open("a") as f:
        if new:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(row[k]) for k in keys) + "\n")


def append_registry(entry: dict, registry_path: str = "logs/runs.jsonl") -> None:
    """Append one JSONL entry to the global registry."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts":   _utcnow(),
        "git":  _git_commit(),
        "host": socket.gethostname(),
        **entry,
    }
    with path.open("a") as f:
        f.write(json.dumps(payload, default=str) + "\n")
