# SPDX-License-Identifier: Apache-2.0

"""Top-level entry point for the ``areal`` console-script.

This baseline only wires the ``inf`` namespace.  Other top-level
surfaces (``train``, ``agent``, ``weight-update``) will land in
follow-up PRs once we agree on the click-based registration shape;
keeping the surface to one namespace makes that refactor easy to
review.

Heavy imports (torch / sglang / vllm / fastapi / ...) must stay out
of this module and out of any ``add_parser`` function.  Each verb's
``_handle`` is the only place that may pull in those packages.
"""

from __future__ import annotations

import argparse
import sys

from areal.experimental.cli.commands import inf as cmd_inf
from areal.version import __version__

_DESCRIPTION = """\
AReaL operator CLI for the v2 microservice architecture.

Available namespaces:
  inf            Manage inference services and models

State lives under ~/.areal/<namespace>/.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="areal",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"areal {__version__}",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )
    cmd_inf.add_parser(subparsers)
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 2
    result = func(args)
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    sys.exit(cli())
