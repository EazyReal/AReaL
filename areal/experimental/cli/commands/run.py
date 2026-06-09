# SPDX-License-Identifier: Apache-2.0

"""``areal run`` — attached (foreground) training driver launcher."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from areal.utils.logging import getLogger

logger = getLogger("TrainCli")


_DESCRIPTION = """\
Launch a training driver in the current process (foreground / attached).

The CLI process IS the driver process — it stays alive until the driver
exits. Exit code reflects the driver outcome (0 = success, non-zero =
failure). This is the right entry for K8s Jobs / Slurm sbatch scripts /
any orchestrator that uses process lifecycle to detect task status.

For "fire and forget" launches (login-node convenience), use `areal start`
instead, which spawns a detached background process.

Run name is derived from `experiment_name`/`trial_name` in the yaml
(Hydra overrides honored). The yaml MUST contain both, or the overrides
must supply them.

A background heartbeat thread updates RunState.last_heartbeat every 5 s
so other CLIs can detect hung drivers via `areal status` (later stages).

Examples:
  areal run --config experiments/grpo.yaml
  areal run --config experiments/grpo.yaml --driver examples.math.gsm8k_rl:main
  areal run --config experiments/grpo.yaml actor.lr=1e-5 trial_name=lr-sweep-3
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch a training driver in the foreground.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument(
        "--driver", default=None,
        help="Driver entry 'module.path:func' (overrides yaml `driver:`).",
    )
    p.add_argument(
        "overrides", nargs=argparse.REMAINDER,
        help="Hydra-style overrides forwarded to the driver.",
    )
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli._exec import run_with_wrapper
    from areal.experimental.cli.runner import (
        refuse_if_active,
        resolve_driver,
        resolve_name,
    )
    from areal.experimental.cli.state import RunState, run_state_path

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    overrides = args.overrides or []
    driver = resolve_driver(config_path, cli_driver=args.driver)
    name = resolve_name(config_path, overrides=overrides)
    refuse_if_active(name)

    state = RunState(
        name=name,
        driver=driver,
        config_path=str(config_path),
        pid=os.getpid(),
        started_at=time.time(),
        log_path="",
        overrides=list(overrides),
        last_heartbeat=time.time(),
    )
    state.save()

    logger.info(
        "Run %r starting (pid %d). State: %s",
        name, os.getpid(), run_state_path(name),
    )
    return run_with_wrapper(name, driver, config_path, overrides)
