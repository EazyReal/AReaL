# SPDX-License-Identifier: Apache-2.0

"""Process management for ``areal inf`` — gateway + router only.

There is intentionally **no supervisor process**.  ``run`` spawns the gateway
and router as detached subprocesses (each in its own process group via
``start_new_session``), writes their PIDs to ``services/<name>.json``, and
exits.  ``stop`` reads those PIDs back, sends SIGTERM, and escalates to
SIGKILL after the grace period.

If the CLI process dies between fork and write, the subprocesses become
orphans owned by ``init``; the next ``run --force`` (or manual cleanup) is
expected to reconcile.  This is the ``no hidden manager`` choice from the
design doc.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from areal.experimental.cli.state import pid_alive


HEALTH_POLL_INTERVAL_S = 0.5


def spawn_router(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    poll_interval: float,
    routing_strategy: str,
    log_level: str,
    log_file: Path,
) -> int:
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.router",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--poll-interval", str(poll_interval),
        "--routing-strategy", routing_strategy,
        "--log-level", log_level,
    ]
    return _spawn_detached(cmd, log_file)


def spawn_gateway(
    *,
    host: str,
    port: int,
    admin_api_key: str,
    router_host: str,
    router_port: int,
    router_timeout: float,
    forward_timeout: float,
    log_level: str,
    log_file: Path,
) -> int:
    # Workers reach the router via this address; if router bound to all
    # interfaces, fall back to loopback for the gateway -> router link.
    rh = "127.0.0.1" if router_host in ("0.0.0.0", "::") else router_host
    cmd = [
        sys.executable, "-m", "areal.experimental.inference_service.gateway",
        "--host", host,
        "--port", str(port),
        "--admin-api-key", admin_api_key,
        "--router-addr", f"http://{rh}:{router_port}",
        "--router-timeout", str(router_timeout),
        "--forward-timeout", str(forward_timeout),
        "--log-level", log_level,
    ]
    return _spawn_detached(cmd, log_file)


def _spawn_detached(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    lf = open(log_file, "ab", buffering=0)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=lf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    return proc.pid


# ---- shutdown ------------------------------------------------------------


def signal_pid(pid: int, sig: int) -> bool:
    """Best-effort kill the whole process group, falling back to the pid."""
    if pid <= 0:
        return False
    try:
        os.killpg(os.getpgid(pid), sig)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        pass
    try:
        os.kill(pid, sig)
        return True
    except ProcessLookupError:
        return False


def wait_dead(pids: list[int], deadline: float) -> bool:
    while time.time() < deadline:
        if not any(pid_alive(p) for p in pids):
            return True
        time.sleep(0.2)
    return False


def kill_pids(pids: list[int], grace_s: float) -> None:
    """SIGTERM all, wait grace_s, SIGKILL leftovers."""
    pids = [p for p in pids if p > 0]
    if not pids:
        return
    for p in pids:
        if pid_alive(p):
            signal_pid(p, signal.SIGTERM)
    if not wait_dead(pids, time.time() + grace_s):
        for p in pids:
            if pid_alive(p):
                signal_pid(p, signal.SIGKILL)
        wait_dead(pids, time.time() + 5.0)
