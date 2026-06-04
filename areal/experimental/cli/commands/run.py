# SPDX-License-Identifier: Apache-2.0

"""``areal run`` — detached training driver launcher."""

from __future__ import annotations

import argparse
from pathlib import Path


_DESCRIPTION = """\
Launch a training driver in the background.

Resolve the driver entry from --driver or the yaml `driver:` field, then
spawn it as a detached subprocess. The CLI returns immediately after the
child has been launched; the run is tracked under ~/.areal/runs/ via
state + heartbeat (refreshed every 5 s by the wrapper process).

Run name is derived from `experiment_name`/`trial_name` in the yaml
(or matching Hydra overrides). The yaml MUST contain both, or the
overrides must supply them.

Examples:
  areal run --config experiments/grpo.yaml
  areal run --config experiments/grpo.yaml --driver examples.math.gsm8k_rl:main
  areal run --config experiments/grpo.yaml actor.lr=1e-5 trial_name=lr-sweep-3

Hydra overrides (key=value, +key=value, ~key) after the parsed flags are
forwarded verbatim to the driver.
"""


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch a training driver in the background.",
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
    from areal.experimental.cli.runner import (
        resolve_driver,
        resolve_name,
        start_detached,
    )

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    overrides = args.overrides or []
    driver = resolve_driver(config_path, cli_driver=args.driver)
    name = resolve_name(config_path, overrides=overrides)
    return start_detached(
        name=name,
        driver_spec=driver,
        config_path=config_path,
        overrides=overrides,
    )
