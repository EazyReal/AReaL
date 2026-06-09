# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only HTTP client for the inference gateway.

Used by ``areal inf {run,status,register,deregister,...}``.  Stays on
``urllib`` to keep the CLI's import surface light — bringing in ``httpx`` /
``aiohttp`` here would push hot-path imports up by hundreds of ms with
nothing to show for it (the CLI never streams large payloads except in
``chat``, which uses its own client).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class GatewayUnreachable(RuntimeError):
    pass


class GatewayHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}")
        self.status = status
        self.body = body


class GatewayClient:
    def __init__(
        self,
        base_url: str,
        *,
        admin_api_key: str | None = None,
        timeout: float = 5.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key
        self.timeout = timeout

    # ---- low level -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        timeout: float | None = None,
        admin: bool = False,
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if json_body is not None:
            data = json.dumps(json_body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if admin and self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise GatewayHTTPError(e.code, e.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise GatewayUnreachable(str(e)) from e
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return body

    # ---- public ----------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def list_models(self) -> dict:
        return self._request("GET", "/models")

    def register_model(self, payload: dict) -> dict:
        return self._request("POST", "/register_model", json_body=payload, admin=True)

    def deregister_model(self, model: str) -> dict:
        return self._request(
            "POST",
            "/deregister_model",
            json_body={"model": model},
            admin=True,
        )
