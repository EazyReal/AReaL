# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — Ollama-style inference service operator console.

This namespace owns its OWN service lifecycle (gateway + router +
optional model backends).  Services are standalone — they don't need
a training yaml, an experiment_name, or any cluster scheduler
integration.  A user starts a serving stack, registers external or
internal models, and chats / collects against them.

A separate training surface (``areal train ...``) will land in a
later PR; the two namespaces are intentionally decoupled — training
experiments wrap ``RolloutControllerV2`` in-process, while ``inf``
spawns gateway/router/data-proxy as detached subprocesses owned by
nothing but the kernel.

Phase 1 verbs:
  run     Launch gateway + router (detached).  Optional inline model
          registration (external via --api-url or internal via
          --backend / --model-path).
  stop    Tear down gateway + router (and any tracked model backends).
  status  Health for one service + its components and registered models.
  ps      List locally tracked services.
  logs    Tail gateway / router / model logs.

Future phases:
  register / deregister / models  — standalone model lifecycle verbs.
  chat / collect / interactive shell.

State lives under ``~/.areal/inf/``; see ``state.py`` for layout.
"""

from __future__ import annotations

import argparse


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "inf",
        help="Manage inference services and models.",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="verb", required=True, metavar="VERB")

    from areal.experimental.cli.commands.inf import (
        logs as cmd_logs,
        ps as cmd_ps,
        run as cmd_run,
        status as cmd_status,
        stop as cmd_stop,
    )

    cmd_run.add_parser(sub)
    cmd_stop.add_parser(sub)
    cmd_status.add_parser(sub)
    cmd_ps.add_parser(sub)
    cmd_logs.add_parser(sub)
