# SPDX-License-Identifier: Apache-2.0

"""TOML user defaults for ``areal inf``.

Per design 12.1 the user-managed config lives at ``~/.areal/inf/config.toml``
with sections ``[default]``, ``[launch]``, ``[register.internal]``,
``[collect]``, ``[chat]``.  CLI flags override the config; the config
overrides hard-coded defaults.

Only what's needed for phase 1 is exercised here (sections ``[default]`` and
``[launch]``); the rest is read in later phases on the same plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from areal.experimental.cli.commands.inf.state import inf_root


def config_path() -> Path:
    return inf_root() / "config.toml"


def load_user_config(extra: Path | None = None) -> dict[str, Any]:
    """Load user config + optional ``--config`` override file.

    Parses with stdlib ``tomllib`` (3.11+) so no third-party dep required.
    Returns a dict-of-dicts; missing sections become empty dicts.  An
    extra override file is shallow-merged section-by-section over the user
    config (last write wins inside each section).
    """
    import tomllib

    merged: dict[str, dict] = {
        "default": {},
        "launch": {},
        "register": {"internal": {}},
        "collect": {},
        "chat": {},
    }

    paths: list[Path] = []
    p = config_path()
    if p.exists():
        paths.append(p)
    if extra is not None:
        extra = extra.expanduser().resolve()
        if not extra.exists():
            raise SystemExit(f"--config file not found: {extra}")
        paths.append(extra)

    for path in paths:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        for section, values in data.items():
            if section not in merged:
                merged[section] = {}
            if isinstance(values, dict):
                _shallow_merge(merged[section], values)
            else:
                merged[section] = values

    return merged


def _shallow_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _shallow_merge(dst[k], v)
        else:
            dst[k] = v
