# SPDX-License-Identifier: Apache-2.0

"""Driver resolution and detached launch for ``areal run``."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from areal.experimental.cli.state import (
    RunState,
    pid_alive,
    run_log_path,
    run_state_path,
)


def _raw_yaml(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Top-level of {config_path} must be a YAML mapping.")
    return data


def _peek_driver(config_path: Path) -> str | None:
    return _raw_yaml(config_path).get("driver")


def _override_value(overrides: list[str], key: str) -> str | None:
    for o in reversed(overrides):
        stripped = o.lstrip("+")
        if stripped.startswith(f"{key}="):
            return stripped.split("=", 1)[1]
    return None


def resolve_driver(config_path: Path, cli_driver: str | None) -> str:
    if cli_driver:
        return cli_driver
    yaml_driver = _peek_driver(config_path)
    if yaml_driver:
        return yaml_driver
    raise SystemExit(
        f"No driver specified.\n"
        f"  Either add a `driver:` field to {config_path}:\n"
        f"      driver: examples.math.gsm8k_rl:main\n"
        f"  Or pass --driver on the command line:\n"
        f"      areal run --config {config_path} --driver examples.math.gsm8k_rl:main"
    )


def resolve_name(config_path: Path, overrides: list[str]) -> str:
    raw = _raw_yaml(config_path)
    exp = _override_value(overrides, "experiment_name") or raw.get("experiment_name")
    trial = _override_value(overrides, "trial_name") or raw.get("trial_name")
    if not (exp and trial):
        raise SystemExit(
            f"experiment_name and trial_name are required.\n"
            f"  Add them to {config_path}, or pass\n"
            f"      experiment_name=<exp> trial_name=<trial>\n"
            f"  as Hydra overrides."
        )
    return f"{exp}/{trial}"


def _refuse_if_active(name: str) -> None:
    p = run_state_path(name)
    if not p.exists():
        return
    try:
        existing = RunState.load(name)
    except (FileNotFoundError, ValueError):
        return
    if pid_alive(existing.pid):
        raise SystemExit(
            f"Run {name!r} already active (pid={existing.pid}). "
            f"Use `areal stop {name}` first."
        )


def start_detached(
    *, name: str, driver_spec: str, config_path: Path, overrides: list[str]
) -> int:
    _refuse_if_active(name)

    log_file = run_log_path(name)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "areal.experimental.cli._exec",
        "--name", name,
        "--driver", driver_spec,
        "--config", str(config_path),
        "--",
        *overrides,
    ]

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    with open(log_file, "wb", buffering=0) as lf:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=lf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )

    state = RunState(
        name=name,
        driver=driver_spec,
        config_path=str(config_path),
        pid=proc.pid,
        started_at=time.time(),
        log_path=str(log_file),
        overrides=list(overrides),
        last_heartbeat=time.time(),
    )
    state.save()

    print(f"Started run {name!r} (pid {proc.pid}).")
    print(f"  log:   {log_file}")
    print(f"  state: {run_state_path(name)}")
    return 0
