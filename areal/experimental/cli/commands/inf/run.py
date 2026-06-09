# SPDX-License-Identifier: Apache-2.0

"""``areal inf run`` — launch the inference service (detached).

Spawns a gateway and a router as detached subprocesses, polls the gateway's
``/health`` until it answers, optionally registers an external model
inline, persists ``ServiceState`` (and ``ModelState`` if a model was
registered) under ``~/.areal/inf/``, and exits.

There is **no supervisor process**.  Subsequent commands (``stop``,
``status``, ``ps``, ``logs``) reconcile via ``pid_alive`` + gateway
``/health``.

Examples:
  areal inf run
  areal inf run --service demo --gateway-port 18080 --router-port 18081
  areal inf run --model gpt-4o --api-url https://api.openai.com/v1 \\
                --provider-api-key-env OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


_DESCRIPTION = __doc__


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "run",
        help="Launch the inference service (detached).",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Service arguments (design 11.2)
    p.add_argument("--service", default="default", help="Service instance name.")
    p.add_argument("--gateway-host", default="127.0.0.1")
    p.add_argument("--gateway-port", type=int, default=8080)
    p.add_argument("--router-host", default="127.0.0.1")
    p.add_argument("--router-port", type=int, default=8081)
    p.add_argument("--admin-api-key", default="areal-admin-key")
    p.add_argument(
        "--routing-strategy", default="round_robin",
        choices=["round_robin", "least_busy"],
    )
    p.add_argument("--poll-interval", type=float, default=5.0)
    p.add_argument("--router-timeout", type=float, default=2.0)
    p.add_argument("--forward-timeout", type=float, default=120.0)
    p.add_argument(
        "--log-level", default="info",
        choices=["debug", "info", "warning", "error"],
    )
    p.add_argument("--launch-timeout", type=float, default=30.0)
    p.add_argument(
        "--config", type=Path, default=None,
        help="Optional TOML override file merged over ~/.areal/inf/config.toml.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Stop an existing healthy service with the same name first.",
    )

    # Inline external model registration (phase 1b only supports external).
    # Internal model flags (--backend, --model-path, ...) land in phase 3.
    p.add_argument(
        "--model", default=None,
        help="Model name to register at startup. Triggers inline registration.",
    )
    p.add_argument(
        "--api-url", default=None,
        help="External provider URL.  Presence marks the model as external.",
    )
    p.add_argument("--provider-api-key", default=None)
    p.add_argument(
        "--provider-api-key-env", default=None,
        help="Name of an environment variable holding the provider API key.",
    )
    p.add_argument(
        "--provider-model", default=None,
        help="Upstream model name to send to the provider (defaults to --model).",
    )

    p.set_defaults(func=_handle)


# ---- helpers --------------------------------------------------------------


def _resolve_provider_api_key(args: argparse.Namespace) -> str:
    if args.provider_api_key:
        return args.provider_api_key
    if args.provider_api_key_env:
        v = os.environ.get(args.provider_api_key_env)
        if not v:
            raise SystemExit(
                f"--provider-api-key-env={args.provider_api_key_env!r} "
                f"is not set in the environment."
            )
        return v
    raise SystemExit(
        "External model registration requires either --provider-api-key or "
        "--provider-api-key-env."
    )


def _refuse_or_replace(name: str, force: bool) -> None:
    """If a service with this name is healthy, refuse (or replace if --force)."""
    from areal.experimental.cli.commands.inf.launcher import kill_pids
    from areal.experimental.cli.commands.inf.state import (
        ServiceState,
        gateway_alive,
        router_alive,
        service_state_path,
    )

    p = service_state_path(name)
    if not p.exists():
        return
    try:
        existing = ServiceState.load(name)
    except (FileNotFoundError, ValueError, TypeError):
        # Stale or corrupt state — overwrite silently.
        p.unlink()
        return

    alive = gateway_alive(existing) or router_alive(existing)
    if alive and not force:
        raise SystemExit(
            f"Service {name!r} is already running "
            f"(gateway pid={existing.gateway_pid}, router pid={existing.router_pid}). "
            f"Use --force to replace it, or `areal inf stop {name}` first."
        )
    if alive:
        kill_pids([existing.gateway_pid, existing.router_pid], grace_s=10.0)

    existing.remove()
    # Models file from previous incarnation is also stale; nuke it.
    from areal.experimental.cli.commands.inf.state import models_state_path
    mp = models_state_path(name)
    if mp.exists():
        mp.unlink()


def _wait_health(client, supervisor_pids: list[int], deadline: float) -> None:
    from areal.experimental.cli.commands.inf.gateway_client import GatewayUnreachable
    from areal.experimental.cli.state import pid_alive

    last_err: Exception | None = None
    while time.time() < deadline:
        if not all(pid_alive(p) for p in supervisor_pids):
            raise SystemExit(
                "Gateway or router subprocess died during startup. "
                "Check the logs printed above for the reason."
            )
        try:
            client.health()
            return
        except GatewayUnreachable as e:
            last_err = e
            time.sleep(0.5)
    raise SystemExit(
        f"Service did not become healthy within timeout. "
        f"Last error: {last_err}"
    )


def _register_external_inline(
    *, args: argparse.Namespace, service_name: str, gateway_url: str
) -> None:
    """Register an external model right after the service is healthy.

    Persists a ``ModelState`` and promotes it to default if first.
    """
    from areal.experimental.cli.commands.inf.gateway_client import (
        GatewayClient,
        GatewayHTTPError,
        GatewayUnreachable,
    )
    from areal.experimental.cli.commands.inf.state import (
        ModelState,
        ServiceModels,
    )

    api_key = _resolve_provider_api_key(args)
    payload = {
        "model": args.model,
        "url": args.api_url,
        "api_key": api_key,
        "data_proxy_addrs": [],
    }
    if args.provider_model:
        payload["provider_model"] = args.provider_model

    client = GatewayClient(
        gateway_url, admin_api_key=args.admin_api_key, timeout=10.0
    )
    try:
        client.register_model(payload)
    except (GatewayUnreachable, GatewayHTTPError) as e:
        raise SystemExit(
            f"Inline register of model {args.model!r} failed: {e}"
        ) from e

    models = ServiceModels.load(service_name)
    models.add(
        ModelState(
            name=args.model,
            kind="external",
            api_url=args.api_url,
            provider_model=args.provider_model or args.model,
            registered_at=time.time(),
        )
    )
    models.save()


# ---- main entry ----------------------------------------------------------


def _handle(args: argparse.Namespace) -> int:
    # Validate inline-model flags up front: phase 1 only supports external.
    if args.model and not args.api_url:
        raise SystemExit(
            "Phase 1 only supports external model registration on `inf run`. "
            "Internal models (--backend / --model-path) will be supported in "
            "a later phase. Pass --api-url <url> to register an external model."
        )
    if args.api_url and not args.model:
        raise SystemExit("--api-url requires --model.")

    from areal.experimental.cli.commands.inf.gateway_client import GatewayClient
    from areal.experimental.cli.commands.inf.launcher import (
        kill_pids,
        spawn_gateway,
        spawn_router,
    )
    from areal.experimental.cli.commands.inf.state import (
        ServiceState,
        get_current_service,
        service_logs_dir,
        set_current_service,
    )

    service = args.service
    _refuse_or_replace(service, force=args.force)

    logs = service_logs_dir(service)
    print(f"Starting service {service!r} ...", file=sys.stderr)
    print(f"  logs: {logs}", file=sys.stderr)

    router_pid = spawn_router(
        host=args.router_host,
        port=args.router_port,
        admin_api_key=args.admin_api_key,
        poll_interval=args.poll_interval,
        routing_strategy=args.routing_strategy,
        log_level=args.log_level,
        log_file=logs / "router.log",
    )
    print(f"  router pid:  {router_pid}", file=sys.stderr)

    # Give the router a head start so the gateway can dial it.
    time.sleep(0.3)

    gateway_pid = spawn_gateway(
        host=args.gateway_host,
        port=args.gateway_port,
        admin_api_key=args.admin_api_key,
        router_host=args.router_host,
        router_port=args.router_port,
        router_timeout=args.router_timeout,
        forward_timeout=args.forward_timeout,
        log_level=args.log_level,
        log_file=logs / "gateway.log",
    )
    print(f"  gateway pid: {gateway_pid}", file=sys.stderr)

    state = ServiceState(
        name=service,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        router_host=args.router_host,
        router_port=args.router_port,
        gateway_pid=gateway_pid,
        router_pid=router_pid,
        admin_api_key=args.admin_api_key,
        routing_strategy=args.routing_strategy,
        log_level=args.log_level,
        created_at=time.time(),
    )

    # Use the state's gateway_url helper so 0.0.0.0 collapses to 127.0.0.1.
    client = GatewayClient(
        state.gateway_url, admin_api_key=args.admin_api_key, timeout=2.0
    )
    try:
        _wait_health(client, [router_pid, gateway_pid],
                     deadline=time.time() + args.launch_timeout)
    except SystemExit:
        kill_pids([gateway_pid, router_pid], grace_s=5.0)
        raise

    state.save()

    if args.model:
        try:
            _register_external_inline(
                args=args, service_name=service, gateway_url=state.gateway_url
            )
        except SystemExit:
            # If inline registration fails, tear the empty service down so
            # the user isn't left with a half-started instance.
            kill_pids([gateway_pid, router_pid], grace_s=5.0)
            state.remove()
            raise

    if get_current_service() is None:
        set_current_service(service)

    print(f"\nService {service!r} ready.")
    print(f"  gateway:  {state.gateway_url}")
    print(f"  router:   {state.router_url}")
    print(f"  pids:     gateway={gateway_pid}, router={router_pid}")
    if args.model:
        print(f"  default model: {args.model} (external)")
    print(f"  log dir:  {logs}")
    return 0
