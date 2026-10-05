# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""WebDriverAgent HTTP client for physical iOS UI automation.

Xcode's ``DeviceInteraction*`` MCP tools accept simulators only, so physical
devices are driven through WebDriverAgent — the same XCUITest bridge Appium
uses. The client speaks plain HTTP to the WDA server running on the device
and has no third-party dependencies; every call is offloaded to a thread so
the driver stays fully async.

Reaching the device-side server needs one of:

- ``ARTEMIS_IOS_WDA_URL``: an explicit endpoint such as
  ``http://127.0.0.1:8100`` for ``iproxy``/``pymobiledevice3`` forwards or
  ``http://<device-ip>:8100`` when the phone shares the LAN.
- The CoreDevice tunnel address from ``devicectl device info details`` —
  paired devices already hold a managed IPv6 tunnel that routes TCP.
- ``ARTEMIS_IOS_WDA_HOST``: just a host/IP; port 8100 is assumed.
"""

import asyncio
import base64
import json
import os
from typing import Any
import urllib.error
import urllib.request

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

WDA_URL_ENV = "ARTEMIS_IOS_WDA_URL"
WDA_HOST_ENV = "ARTEMIS_IOS_WDA_HOST"
WDA_DEFAULT_PORT = 8100
WDA_REQUEST_TIMEOUT = 30.0


class WdaUnavailableError(RuntimeError):
    """The WebDriverAgent server could not be reached or did not respond."""


class WdaClient:
    """Minimal WebDriverAgent client covering Artemis's interaction surface."""

    def __init__(self, base_url: str, timeout: float = WDA_REQUEST_TIMEOUT):
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session_id: str | None = None

    @property
    def base_url(self) -> str:
        return self._base

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def connected(self) -> bool:
        return self._session_id is not None

    def _sync_request(
        self, method: str, path: str, payload: dict[str, Any] | None, timeout: float
    ) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            f"{self._base}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            raise RuntimeError(f"WebDriverAgent {method} {path} failed: HTTP {error.code} {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise WdaUnavailableError(
                f"WebDriverAgent at {self._base} is unreachable: {error}. "
                "Check the device connection, the WDA runner process, and any "
                "port forwarding (iproxy or 'pymobiledevice3 remote')."
            )
        try:
            payload_out = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(payload_out, dict) and "value" in payload_out:
            value = payload_out["value"]
            if isinstance(value, dict) and value.get("error"):
                message = value.get("message") or value["error"]
                raise RuntimeError(f"WebDriverAgent {method} {path} failed: {message}")
            return value
        return payload_out

    async def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self._sync_request, method, path, payload, timeout or self._timeout
        )

    # --- Session lifecycle ---

    async def status(self, timeout: float = 5.0) -> dict[str, Any] | None:
        try:
            value = await self._request("GET", "/status", timeout=timeout)
        except (RuntimeError, WdaUnavailableError, OSError) as error:
            logger.debug(f"WDA status probe failed for {self._base}: {error}")
            return None
        return value if isinstance(value, dict) else {"value": value}

    async def open_session(self) -> str:
        payload = {
            "capabilities": {
                "alwaysMatch": {"platformName": "iOS"},
                "firstMatch": [{}],
            }
        }
        value = await self._request("POST", "/session", payload, timeout=60.0)
        session_id = None
        if isinstance(value, dict):
            session_id = value.get("sessionId")
            if session_id is None and isinstance(value.get("capabilities"), dict):
                session_id = value.get("capabilities", {}).get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("WebDriverAgent did not return a session id.")
        self._session_id = session_id
        return session_id

    async def close_session(self) -> None:
        session_id, self._session_id = self._session_id, None
        if session_id is None:
            return
        try:
            await self._request("DELETE", f"/session/{session_id}", timeout=10.0)
        except (RuntimeError, WdaUnavailableError, OSError) as error:
            logger.debug(f"WDA session delete failed: {error}")

    # --- Observation ---

    async def screenshot_png(self) -> bytes:
        value = await self._request("GET", "/screenshot")
        if not isinstance(value, str) or not value:
            raise RuntimeError("WebDriverAgent returned no screenshot data.")
        return base64.b64decode(value)

    async def source_json(self) -> dict[str, Any]:
        value = await self._request("GET", "/source?format=json")
        if not isinstance(value, dict):
            raise RuntimeError("WebDriverAgent returned a non-JSON hierarchy.")
        return value

    async def window_size(self) -> tuple[float, float]:
        path = f"/session/{self._session_id}/window/size" if self._session_id else "/window/size"
        value = await self._request("GET", path)
        if isinstance(value, dict) and "width" in value and "height" in value:
            return float(value["width"]), float(value["height"])
        raise RuntimeError(f"WebDriverAgent returned an unexpected window size: {value!r}")

    async def active_app(self) -> str | None:
        path = "/wda/activeAppInfo"
        if self._session_id:
            path = f"/session/{self._session_id}/wda/activeAppInfo"
        try:
            value = await self._request("GET", path)
        except (RuntimeError, WdaUnavailableError):
            return None
        if isinstance(value, dict):
            bundle = value.get("bundleId") or value.get("bundleIdentifier")
            return bundle if isinstance(bundle, str) and bundle else None
        return None

    # --- Input ---

    async def _actions(self, pointer_actions: list[dict[str, Any]]) -> None:
        session = self._require_session()
        payload = {
            "actions": [
                {
                    "type": "pointer",
                    "id": "artemis-finger",
                    "parameters": {"pointerType": "touch"},
                    "actions": pointer_actions,
                }
            ]
        }
        await self._request("POST", f"/session/{session}/actions", payload)

    def _require_session(self) -> str:
        if not self._session_id:
            raise RuntimeError("WebDriverAgent session is not open.")
        return self._session_id

    async def tap(self, x: float, y: float, hold_ms: int = 0) -> None:
        actions: list[dict[str, Any]] = [
            {"type": "pointerMove", "duration": 0, "x": x, "y": y},
            {"type": "pointerDown", "button": 0},
        ]
        if hold_ms > 0:
            actions.append({"type": "pause", "duration": hold_ms})
        actions.append({"type": "pointerUp", "button": 0})
        await self._actions(actions)

    async def swipe(self, sx: float, sy: float, ex: float, ey: float, duration_ms: int) -> None:
        await self._actions(
            [
                {"type": "pointerMove", "duration": 0, "x": sx, "y": sy},
                {"type": "pointerDown", "button": 0},
                {"type": "pause", "duration": 50},
                {"type": "pointerMove", "duration": duration_ms, "x": ex, "y": ey},
                {"type": "pointerUp", "button": 0},
            ]
        )

    async def type_text(self, text: str) -> None:
        session = self._require_session()
        # ``value`` is a list of Unicode code points — newlines and non-ASCII
        # text carry their literal values to the focused field.
        payload = {"value": list(text)}
        await self._request("POST", f"/session/{session}/wda/keys", payload)

    async def press_button(self, name: str) -> bool:
        """WDA hardware buttons: home, volumeUp, volumeDown, power."""
        session = self._require_session()
        try:
            await self._request(
                "POST", f"/session/{session}/wda/pressButton", {"name": name}
            )
            return True
        except (RuntimeError, WdaUnavailableError):
            return False

    async def homescreen(self) -> None:
        await self._request("POST", "/wda/homescreen")

    async def lock(self) -> None:
        await self._request("POST", "/wda/lock")

    async def unlock(self) -> None:
        await self._request("POST", "/wda/unlock")


def normalize_wda_url(raw: str) -> str:
    """Accept bare hosts, IPv6 literals, or full URLs and return a base URL."""
    value = raw.strip()
    if not value:
        return value
    if "://" not in value:
        # Bare IPv6 literals need brackets once a port is attached.
        if value.count(":") > 1 and not value.startswith("["):
            value = f"[{value}]"
        value = f"http://{value}"
    value = value.rstrip("/")
    authority = value.split("://", 1)[1]
    host_port = authority.split("/", 1)[0]
    # A port is present when ':' follows the host (or the IPv6 ']' bracket).
    has_port = (
        host_port.rsplit("]", 1)[-1].startswith(":")
        if "]" in host_port
        else ":" in host_port
    )
    if not has_port:
        value = f"{value}:{WDA_DEFAULT_PORT}"
    return value


def wda_url_candidates(
    env_url: str | None = None,
    env_host: str | None = None,
    tunnel_ip: str | None = None,
) -> list[str]:
    """Ordered endpoints to probe for a running WebDriverAgent server."""
    candidates: list[str] = []
    for raw in (
        env_url if env_url is not None else os.environ.get(WDA_URL_ENV),
        env_host if env_host is not None else os.environ.get(WDA_HOST_ENV),
        f"[{tunnel_ip}]:{WDA_DEFAULT_PORT}" if tunnel_ip else None,
        f"127.0.0.1:{WDA_DEFAULT_PORT}",
    ):
        if not raw:
            continue
        url = normalize_wda_url(raw)
        if url and url not in candidates:
            candidates.append(url)
    return candidates


async def probe_wda(
    candidates: list[str], timeout: float = 5.0
) -> WdaClient | None:
    """Return a client bound to the first endpoint that answers ``/status``."""
    for url in candidates:
        client = WdaClient(url)
        status = await client.status(timeout=timeout)
        if status is not None:
            return client
    return None


_ELEMENT_TYPE_PREFIX = "XCUIElementType"


def parse_wda_elements(
    node: dict[str, Any],
    scale: tuple[float, float],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Flatten a WDA ``/source?format=json`` tree into Artemis ui_elements.

    Mirrors ``parse_hierarchy``'s output: ``text``, ``resource_id``, ``class``,
    ``bounds``, ``parsed_bounds``, and ``hit_point`` in screenshot pixels.
    WDA's ``isVisible`` is advisory; elements with usable geometry are kept so
    downstream consumers see everything XCTest reports.
    """
    elements: list[dict[str, Any]] = []

    def visit(entry: dict[str, Any]) -> None:
        rect = entry.get("rect")
        if isinstance(rect, dict):
            x = float(rect.get("x") or 0.0)
            y = float(rect.get("y") or 0.0)
            w = float(rect.get("width") or 0.0)
            h = float(rect.get("height") or 0.0)
        else:
            x = y = w = h = 0.0
        if w > 0 and h > 0:
            left, top = round(x * scale[0]), round(y * scale[1])
            right, bottom = round((x + w) * scale[0]), round((y + h) * scale[1])
            if right > 0 and bottom > 0 and left < width and top < height:
                label = entry.get("label")
                value = entry.get("value")
                name = entry.get("name")
                text = ""
                if isinstance(label, str) and label:
                    text = label
                elif value is not None and str(value):
                    text = str(value)
                element_type = str(entry.get("type") or "")
                element: dict[str, Any] = {
                    "text": text,
                    "resource_id": name if isinstance(name, str) else "",
                    "class": element_type.removeprefix(_ELEMENT_TYPE_PREFIX),
                    "bounds": f"[{left},{top}][{right},{bottom}]",
                    "parsed_bounds": {
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                    },
                    "hit_point": [(left + right) // 2, (top + bottom) // 2],
                    "visible": bool(entry.get("isVisible", True)),
                }
                if value is not None and str(value) != text:
                    element["value"] = str(value)
                elements.append(element)
        children = entry.get("children")
        if isinstance(children, list):
            for child in children:
                if isinstance(child, dict):
                    visit(child)

    visit(node)
    return elements
