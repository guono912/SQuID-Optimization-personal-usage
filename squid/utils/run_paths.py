"""Unified run-directory and output-path helpers for SQuID CLIs.

Every SQuID CLI that produces artifacts must use this module to resolve its
run/output directory. Rules (see runs/FILE_MANAGEMENT_GUIDELINES.md):

- generated files go into an explicit run_dir/output_dir, never the
  repository root and never os.getcwd();
- output paths must stay inside the requested run dir;
- manifest metadata is recorded next to the artifacts.

There is deliberately only one path helper in the repository: this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def resolve_run_dir(run_dir, run_name=None, base="runs"):
    """Resolve an explicit run directory.

    Parameters
    ----------
    run_dir : str | Path | None
        Explicit run directory (from CLI/config). Preferred.
    run_name : str | None
        Fallback name used to build ``<base>/<run_name>`` when run_dir is
        missing. Relative to the process cwd (a ``runs/`` tree under it).
    base : str
        Directory name used with run_name.

    Returns
    -------
    Path
        Absolute run directory. Raises ValueError when neither run_dir nor
        run_name is provided: ad-hoc runs must name their output directory.
    """
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    if run_name:
        return (Path(base) / run_name).expanduser().resolve()
    raise ValueError(
        "an explicit run directory is required: pass --run_dir / --output_dir "
        "(or --run_name for a <cwd>/runs/<name> fallback)"
    )


def ensure_run_dir(run_dir, run_name=None, base="runs"):
    """Resolve the run directory and create it (parents included)."""
    path = resolve_run_dir(run_dir, run_name, base)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_output_path(run_dir, filename):
    """Return ``run_dir / filename``, refusing escapes outside the run dir.

    Raises ValueError if a filename tries to traverse outside ``run_dir``.
    """
    run_dir = Path(run_dir)
    target = (run_dir / filename).resolve()
    root = run_dir.resolve()
    if target != root and root not in target.parents:
        raise ValueError(
            f"output path {filename!r} escapes run directory {run_dir}"
        )
    return target


def manifest_basic_fields(run_dir, command=None, extra=None):
    """Base manifest fields for a run directory.

    Returns a dict with run_dir, timestamp (UTC ISO), command line, python
    executable, and an empty ``artifacts`` list to extend. Include it in any
    manifest.json written into the run directory.
    """
    fields = {
        "run_dir": str(Path(run_dir).resolve()),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": command if command is not None else " ".join(sys.argv),
        "python": sys.executable,
        "artifacts": [],
    }
    if extra:
        fields.update(extra)
    return fields


def write_manifest(run_dir, command=None, extra=None, name="manifest.json"):
    """Write a minimal manifest.json into the run directory and return its path."""
    path = safe_output_path(run_dir, name)
    fields = manifest_basic_fields(run_dir, command=command, extra=extra)
    path.write_text(json.dumps(fields, indent=2) + "\n", encoding="utf-8")
    return path


def sha256_file(path, chunk=1 << 20):
    """Return the lowercase hex SHA256 of a file's contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def run_vmec_in_directory(run_dir, command, **kwargs):
    """Run a VMEC/SIMSOPT subprocess with cwd set to the run directory.

    Legacy VMEC/SIMSOPT relative outputs are relative to the process cwd, so
    any external solver invocation must execute inside ``run_dir``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        command, cwd=str(run_dir), check=True, **kwargs
    )


def require_env(name):
    """Exit with a clear message when a required python module is missing."""
    try:
        __import__(name)
    except ImportError:
        print(f"ERROR: module {name!r} is required for this command.", file=sys.stderr)
        raise SystemExit(1)
