# SPDX-License-Identifier: Apache-2.0

"""Cross-cutting filesystem helpers for the CLI.

Currently only ``inf`` uses these, but they're intentionally generic so
future namespaces (``train``, ``agent``, ``weight-update``) can build on
the same primitives without dragging in any namespace-specific shape.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def areal_home() -> Path:
    env = os.environ.get("AREAL_HOME")
    root = Path(env).expanduser() if env else Path.home() / ".areal"
    root.mkdir(parents=True, exist_ok=True)
    return root


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent) + "\n")


def sanitize_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")
