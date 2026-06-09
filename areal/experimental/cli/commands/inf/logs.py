# SPDX-License-Identifier: Apache-2.0

"""``areal inf logs`` — tail gateway / router / model logs.

Logs live under ``~/.areal/inf/logs/<service>/<component>.log``.  Phase 1
ships ``gateway`` and ``router``; phase 3 will add ``<model-name>`` files
when internal models are spawned.
"""

from __future__ import annotations

import argparse
import os
import sys


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "logs",
        help="Tail gateway / router / model logs.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--service", default=None, help="Service name (defaults to current).")
    p.add_argument(
        "--component", default="gateway",
        help="One of `gateway`, `router`, or a model name.  Becomes "
             "`<component>.log` under the service log dir.",
    )
    p.add_argument("-f", "--follow", action="store_true", help="Stream new lines.")
    p.add_argument("-n", "--lines", type=int, default=200)
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.commands.inf.state import (
        resolve_service,
        service_logs_dir,
    )

    name = resolve_service(args.service)
    log_dir = service_logs_dir(name)
    log_file = log_dir / f"{args.component}.log"
    if not log_file.exists():
        print(
            f"No {args.component}.log at {log_file}.",
            file=sys.stderr,
        )
        return 1

    cmd = ["tail", f"-n{args.lines}"]
    if args.follow:
        cmd.append("-F")
    cmd.append(str(log_file))
    os.execvp(cmd[0], cmd)
