# SPDX-License-Identifier: Apache-2.0

"""Local state for ``areal inf`` services and models.

Layout under ``$AREAL_HOME`` (default ``~/.areal``)::

    inf/
      config.toml                  user defaults (managed by user)
      current-service              single-line: name of the "current" service
      services/<name>.json         ServiceState (gateway+router pids/addrs)
      models/<name>.json           dict of registered ModelState by model name
      logs/<name>/
        gateway.log                gateway subprocess stdout/stderr
        router.log                 router subprocess stdout/stderr
        <model>.log                per-model subprocess output (phase 3)
      chats/<service>/<id>.jsonl   chat session transcripts (phase 3)
      collections/<service>/...    collected trajectories (phase 3)

There is no supervisor process.  ``areal inf run`` writes these files and
exits; later commands reconcile via ``pid_alive`` + gateway ``/health``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from areal.experimental.cli.state import (
    areal_home,
    atomic_write_json,
    pid_alive,
    sanitize_name,
)


# ---- directory helpers ----------------------------------------------------


def inf_root() -> Path:
    d = areal_home() / "inf"
    d.mkdir(parents=True, exist_ok=True)
    return d


def services_dir() -> Path:
    d = inf_root() / "services"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    d = inf_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_logs_dir(name: str) -> Path:
    d = inf_root() / "logs" / sanitize_name(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def service_state_path(name: str) -> Path:
    return services_dir() / f"{sanitize_name(name)}.json"


def models_state_path(service: str) -> Path:
    return models_dir() / f"{sanitize_name(service)}.json"


def current_service_file() -> Path:
    return inf_root() / "current-service"


# ---- ServiceState (gateway + router) -------------------------------------


@dataclass
class ServiceState:
    name: str
    gateway_host: str
    gateway_port: int
    router_host: str
    router_port: int
    gateway_pid: int
    router_pid: int
    admin_api_key: str
    routing_strategy: str = "round_robin"
    log_level: str = "info"
    created_at: float = 0.0

    @property
    def gateway_url(self) -> str:
        host = "127.0.0.1" if self.gateway_host in ("0.0.0.0", "::") else self.gateway_host
        return f"http://{host}:{self.gateway_port}"

    @property
    def router_url(self) -> str:
        host = "127.0.0.1" if self.router_host in ("0.0.0.0", "::") else self.router_host
        return f"http://{host}:{self.router_port}"

    def save(self) -> None:
        atomic_write_json(service_state_path(self.name), asdict(self))

    @classmethod
    def load(cls, name: str) -> ServiceState:
        p = service_state_path(name)
        if not p.exists():
            raise FileNotFoundError(f"No service state for {name!r} at {p}")
        with open(p) as f:
            return cls(**json.load(f))

    def remove(self) -> None:
        p = service_state_path(self.name)
        if p.exists():
            p.unlink()


def gateway_alive(state: ServiceState) -> bool:
    return pid_alive(state.gateway_pid)


def router_alive(state: ServiceState) -> bool:
    return pid_alive(state.router_pid)


# ---- ModelState (per-service registry) -----------------------------------


@dataclass
class ModelState:
    name: str
    kind: str  # "external" | "internal"
    backend_spec: str = ""  # e.g. "sglang:tp=2,dp=2" (internal only)
    api_url: str = ""  # external only
    provider_model: str = ""  # external only (upstream model name)
    data_proxy_addrs: list[str] = field(default_factory=list)
    inference_server_addrs: list[str] = field(default_factory=list)
    worker_pids: list[int] = field(default_factory=list)  # internal only
    metadata: dict = field(default_factory=dict)
    registered_at: float = 0.0


@dataclass
class ServiceModels:
    """All models registered with one service, plus the default model name."""

    service: str
    default_model: str = ""
    models: dict = field(default_factory=dict)  # name -> ModelState dict

    def save(self) -> None:
        atomic_write_json(models_state_path(self.service), asdict(self))

    @classmethod
    def load(cls, service: str) -> ServiceModels:
        p = models_state_path(service)
        if not p.exists():
            return cls(service=service)
        with open(p) as f:
            data = json.load(f)
        return cls(**data)

    def add(self, model: ModelState) -> None:
        if model.name in self.models:
            raise ValueError(f"Model {model.name!r} already registered.")
        self.models[model.name] = asdict(model)
        if not self.default_model:
            self.default_model = model.name

    def remove(self, name: str) -> ModelState | None:
        entry = self.models.pop(name, None)
        if entry is None:
            return None
        if self.default_model == name:
            self.default_model = next(iter(self.models), "")
        return ModelState(**entry)

    def get(self, name: str) -> ModelState | None:
        entry = self.models.get(name)
        return ModelState(**entry) if entry else None

    def list_all(self) -> list[ModelState]:
        return [ModelState(**v) for v in self.models.values()]


# ---- "current service" ---------------------------------------------------


def get_current_service() -> str | None:
    p = current_service_file()
    if not p.exists():
        return None
    name = p.read_text().strip()
    return name or None


def set_current_service(name: str | None) -> None:
    p = current_service_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    if name is None:
        if p.exists():
            p.unlink()
        return
    p.write_text(name + "\n")


def resolve_service(name_arg: str | None) -> str:
    """Service resolution rules per design 11.1.

    Order:
      1. explicit ``--service`` arg
      2. ``current-service`` file
      3. exactly one healthy service exists -> use it
      4. otherwise fail
    """
    if name_arg:
        return name_arg
    cur = get_current_service()
    if cur:
        return cur
    services = [p.stem for p in services_dir().glob("*.json")]
    if len(services) == 1:
        return services[0]
    if not services:
        raise SystemExit(
            "No --service given and no services tracked under ~/.areal/inf/services/."
        )
    raise SystemExit(
        f"No --service given and multiple services exist: {services}. "
        f"Pass --service <name> or set current-service via `areal inf run`."
    )
