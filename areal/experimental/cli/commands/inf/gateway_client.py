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
        return self._request("GET", "/models", admin=True)

    def register_model(self, payload: dict) -> dict:
        return self._request("POST", "/register_model", json_body=payload, admin=True)

    def deregister_model(self, model: str) -> dict:
        return self._request(
            "POST",
            "/deregister_model",
            json_body={"model": model},
            admin=True,
        )


class RouterClient:
    """Direct HTTP client for the router (for worker registration).

    The router's worker pool is populated by ``POST /register {worker_addr}``;
    nothing in the data-proxy or gateway calls this for us, so the CLI does
    it explicitly when it spawns model backends.
    """

    def __init__(
        self, base_url: str, *, admin_api_key: str | None = None, timeout: float = 5.0
    ):
        self.base_url = base_url.rstrip("/")
        self.admin_api_key = admin_api_key
        self.timeout = timeout

    def register_worker(self, worker_addr: str) -> dict:
        url = f"{self.base_url}/register"
        body = json.dumps({"worker_addr": worker_addr}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise GatewayHTTPError(e.code, e.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise GatewayUnreachable(str(e)) from e
        return json.loads(payload) if payload else {}

    def unregister_worker(self, worker_addr: str | None = None, worker_id: str | None = None) -> dict:
        url = f"{self.base_url}/unregister"
        body_dict = {}
        if worker_addr:
            body_dict["worker_addr"] = worker_addr
        if worker_id:
            body_dict["worker_id"] = worker_id
        body = json.dumps(body_dict).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.admin_api_key:
            headers["Authorization"] = f"Bearer {self.admin_api_key}"
        req = urllib.request.Request(url, data=body, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            raise GatewayHTTPError(e.code, e.read().decode("utf-8", errors="replace"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise GatewayUnreachable(str(e)) from e
        return json.loads(payload) if payload else {}
