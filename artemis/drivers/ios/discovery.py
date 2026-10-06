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

"""Shared ``simctl`` device enumeration for iOS Simulator support.

Both the Xcode driver (target validation) and the runtime pools (discovery,
explicit-serial validation, auto-selection) read the same
``xcrun simctl list devices --json`` output through this module so parsing
stays in one place. Every function fails closed to ``None``/``[]`` on
missing tooling so Android-only hosts never see iOS errors.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import time
from typing import Any

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

SIMCTL_QUERY_TIMEOUT = 15.0

# ``simctl list devices`` takes seconds on a busy host and every iOS consumer
# (readiness probe, device pool validation, /api/devices) enumerates it. Share
# one result briefly so polling UIs do not spawn back-to-back simctl calls.
_SIMULATOR_CACHE_TTL = 10.0
_simulator_cache: list[dict[str, Any]] | None = None
_simulator_cache_time = 0.0


def clear_ios_simulator_cache() -> None:
    """Drop the cached enumeration (e.g. after the driver boots a simulator)."""
    global _simulator_cache, _simulator_cache_time
    _simulator_cache = None
    _simulator_cache_time = 0.0


def _cache_simulators(devices: list[dict[str, Any]]) -> list[dict[str, Any]]:
    global _simulator_cache, _simulator_cache_time
    _simulator_cache = devices
    _simulator_cache_time = time.monotonic()
    return devices


def _cached_simulators() -> list[dict[str, Any]] | None:
    if _simulator_cache is None:
        return None
    if time.monotonic() - _simulator_cache_time > _SIMULATOR_CACHE_TTL:
        return None
    return _simulator_cache


async def run_xcrun(*arguments: str, timeout: float = 30.0) -> bytes:
    """Run argv directly, reporting native errors and reaping cancelled children."""
    process = await asyncio.create_subprocess_exec(
        "xcrun",
        *arguments,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()
        raise
    if process.returncode:
        raise RuntimeError(
            f"xcrun {' '.join(arguments[:3])} failed: {stderr.decode(errors='replace').strip()}"
        )
    return stdout


async def plist_to_json(payload: bytes, timeout: float = 30.0) -> bytes:
    """Convert an OpenStep/XML/binary plist payload to JSON via ``plutil``.

    ``simctl listapps`` emits OpenStep (ASCII) plists that ``plistlib``
    cannot read; ``plutil -convert json`` accepts every plist flavor.
    """
    process = await asyncio.create_subprocess_exec(
        "plutil",
        "-convert",
        "json",
        "-o",
        "-",
        "--",
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(input=payload), timeout)
    except (TimeoutError, asyncio.CancelledError):
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.communicate()
        raise
    if process.returncode != 0:
        raise RuntimeError(
            f"plutil conversion failed: {stderr.decode(errors='replace').strip()}"
        )
    return stdout


def simctl_available() -> bool:
    """Whether this host can enumerate iOS simulators at all."""
    return sys.platform == "darwin" and shutil.which("xcrun") is not None


def parse_simctl_devices(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten ``simctl list devices --json`` into available iOS entries."""
    return [
        {
            "udid": device.get("udid"),
            "name": device.get("name"),
            "state": device.get("state"),
            "runtime": runtime,
        }
        for runtime, entries in payload.get("devices", {}).items()
        if ".iOS-" in runtime
        for device in entries
        if device.get("isAvailable") and device.get("udid")
    ]


async def list_ios_simulators(
    force_refresh: bool = False,
) -> list[dict[str, Any]] | None:
    """All available iOS simulators, or ``None`` when enumeration fails.

    Successful enumerations are cached for ``_SIMULATOR_CACHE_TTL`` seconds;
    failures are never cached so callers retry against live simctl.
    """
    if not simctl_available():
        return None
    if not force_refresh:
        cached = _cached_simulators()
        if cached is not None:
            return cached
    try:
        raw = await run_xcrun("simctl", "list", "devices", "--json", timeout=SIMCTL_QUERY_TIMEOUT)
    except (OSError, RuntimeError, TimeoutError) as exc:
        logger.debug(f"simctl device enumeration failed: {exc}")
        return None
    try:
        return _cache_simulators(parse_simctl_devices(json.loads(raw)))
    except (ValueError, TypeError) as exc:
        logger.debug(f"simctl device list parse failed: {exc}")
        return None


def list_ios_simulators_sync(
    force_refresh: bool = False,
) -> list[dict[str, Any]] | None:
    """Synchronous variant for non-async callers (e.g. replay device lists)."""
    if not simctl_available():
        return None
    if not force_refresh:
        cached = _cached_simulators()
        if cached is not None:
            return cached
    try:
        completed = subprocess.run(
            ["xcrun", "simctl", "list", "devices", "--json"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=SIMCTL_QUERY_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug(f"simctl device enumeration failed: {exc}")
        return None
    if completed.returncode != 0:
        return None
    try:
        return _cache_simulators(parse_simctl_devices(json.loads(completed.stdout)))
    except (ValueError, TypeError) as exc:
        logger.debug(f"simctl device list parse failed: {exc}")
        return None
