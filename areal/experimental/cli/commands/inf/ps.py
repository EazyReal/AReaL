# SPDX-License-Identifier: Apache-2.0

"""``areal inf ps`` — list locally tracked services."""

from __future__ import annotations

import argparse
import json
import sys
import time

from areal.utils.logging import getLogger

logger = getLogger("InfCli")


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "ps",
        help="List locally tracked services.",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--all", action="store_true", dest="show_all",
        help="Include stale/dead services.",
    )
    p.add_argument("--json", action="store_true", dest="as_json")
    p.set_defaults(func=_handle)


def _handle(args: argparse.Namespace) -> int:
    from areal.experimental.cli.commands.inf.state import (
        ServiceModels,
        ServiceState,
        gateway_alive,
        get_current_service,
        router_alive,
        services_dir,
    )

    current = get_current_service()
    entries: list[dict] = []
    now = time.time()

    for f in sorted(services_dir().glob("*.json")):
        # The file stem encodes the service name (with sanitize_name applied
        # only for special chars; for typical names it round-trips).
        try:
            name = f.stem
            state = ServiceState.load(name)
        except (FileNotFoundError, ValueError, TypeError, KeyError):
            continue

        alive = gateway_alive(state) or router_alive(state)
        if not alive and not args.show_all:
            continue

        sm = ServiceModels.load(name)
        entries.append(
            {
                "name": name,
                "current": name == current,
                "state": "running" if alive else "dead",
                "gateway": state.gateway_url,
                "router": state.router_url,
                "models": len(sm.models),
                "default_model": sm.default_model,
                "age_s": int(max(0, now - state.created_at)),
            }
        )

    if args.as_json:
        print(json.dumps(entries, indent=2))
        return 0

    if not entries:
        msg = "No services."
        if not args.show_all:
            msg += "  (Add --all to include dead ones.)"
        logger.info("%s", msg)
        return 0

    cols = ("CURRENT", "NAME", "STATE", "GATEWAY", "ROUTER", "MODELS", "AGE")
    rows = [
        (
            "*" if e["current"] else "",
            e["name"],
            e["state"],
            e["gateway"],
            e["router"],
            f"{e['models']}" + (f" (default={e['default_model']})" if e["default_model"] else ""),
            f"{e['age_s']}s",
        )
        for e in entries
    ]
    widths = [max(len(r[i]) for r in (cols, *rows)) for i in range(len(cols))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*cols))
    for r in rows:
        print(fmt.format(*r))
    return 0
