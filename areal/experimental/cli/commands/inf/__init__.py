# SPDX-License-Identifier: Apache-2.0

"""``areal inf`` — Ollama-style inference service operator console.

This namespace owns its OWN service lifecycle (gateway + router +
optional model backends).  It is intentionally NOT the same shape as
``areal run`` (which orchestrates a full RL training experiment):

  * ``areal inf`` services are standalone — they don't need a training
    yaml, an experiment_name, or any cluster scheduler integration.  A
    user starts a serving stack, registers external or internal models,
    and chats / collects against them.
  * ``areal run`` services are training experiments — they wrap the v2
    ``RolloutControllerV2`` and live and die with the experiment.

Phase 1 verbs:
  run     Launch gateway + router (detached).  Optional inline external
          model registration via --model + --api-url.
  stop    Tear down gateway + router (and any tracked model backends).
  status  Health for one service + its components and registered models.
  ps      List locally tracked services.
  logs    Tail gateway / router / model logs.

Phase 2 (separate verbs):
  register / deregister / models  — for external models, then internal.

Phase 3:
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
