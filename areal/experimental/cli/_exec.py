# SPDX-License-Identifier: Apache-2.0

"""Driver execution wrapper.

``run_with_wrapper`` runs a driver in the current process with a heartbeat
thread, SIGTERM trap, and final-state write. Used by both:

  - ``areal run`` (attached / foreground): CLI process is the wrapper
  - ``areal start`` (detached / background): wrapper is a child process
    spawned via ``python -m areal.experimental.cli._exec`` (entry: ``main``)
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

from areal.experimental.cli.state import RunState


HEARTBEAT_INTERVAL_S = 5.0


def _heartbeat_loop(name: str, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            s = RunState.load(name)
            s.last_heartbeat = time.time()
            s.save()
        except (FileNotFoundError, ValueError, OSError):
            pass
        stop_event.wait(HEARTBEAT_INTERVAL_S)


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, _handler)


def run_with_wrapper(
    name: str,
    driver_spec: str,
    config_path: str | Path,
    overrides: list[str],
) -> int:
    _install_sigterm_handler()

    stop = threading.Event()
    hb = threading.Thread(target=_heartbeat_loop, args=(name, stop), daemon=True)
    hb.start()

    rc = 0
    try:
        mod_path, func_name = driver_spec.split(":", 1)
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, func_name)
        argv = ["--config", str(config_path), *overrides]
        result = fn(argv)
        if isinstance(result, int):
            rc = result
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else (1 if e.code else 0)
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        rc = 1
    finally:
        stop.set()
        try:
            final = RunState.load(name)
            final.status = "completed" if rc == 0 else "failed"
            final.exit_code = rc
            final.last_heartbeat = time.time()
            final.save()
        except Exception:
            pass

    return rc


def main() -> int:
    p = argparse.ArgumentParser(prog="areal-exec", add_help=False)
    p.add_argument("--name", required=True)
    p.add_argument("--driver", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs=argparse.REMAINDER)
    args = p.parse_args()

    overrides = args.overrides or []
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]

    state = RunState.load(args.name)
    state.pid = os.getpid()
    state.last_heartbeat = time.time()
    state.save()

    return run_with_wrapper(args.name, args.driver, args.config, overrides)


if __name__ == "__main__":
    sys.exit(main())
