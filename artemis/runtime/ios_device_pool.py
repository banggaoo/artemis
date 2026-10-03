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

"""iOS Simulator device pool: discovery, lock status, and target selection.

The iOS counterpart of :class:`~artemis.runtime.device_pool.DevicePool`.
Enumeration rides ``xcrun simctl list devices``; lock ownership shares the
same :class:`DeviceExecutionLock` registry under the ``ios`` scope so a
simulator and an Android serial can never share a lock identity. All methods
fail open on missing Xcode tooling or an enumeration error, mirroring the
Android pool's admission contract.
"""

from __future__ import annotations

from artemis.drivers.ios.discovery import list_ios_simulators, list_ios_simulators_sync
from artemis.runtime.adb_endpoint import IOS_LOCK_SCOPE
from artemis.runtime.device_lock import DeviceExecutionLock
from artemis.runtime.device_pool import DeviceStatus
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# simctl states the driver can use as-is ("device"/Booted) or boot itself
# ("Shutdown"). Anything else (e.g. "Creating") is rejected on explicit picks.
ACCEPTABLE_STATES = frozenset({"device", "Shutdown"})


class IosDevicePool:
    """Discovers iOS simulators and reports their Artemis lock state."""

    @staticmethod
    def _build_statuses(raw_devices: list[dict]) -> list[DeviceStatus]:
        active_owners = DeviceExecutionLock.get_active_owners()
        # Owner map keys are only scope-prefixed for multi-owner collisions, so
        # match by owner payload: an iOS lock always carries lock_scope="ios",
        # and an Android lock on the same text must never mark a sim busy.
        ios_owners = {
            DeviceExecutionLock._normalize_device_id(o.device_id): o
            for o in active_owners.values()
            if o and getattr(o, "lock_scope", None) == IOS_LOCK_SCOPE
        }
        devices: list[DeviceStatus] = []
        for device in raw_devices:
            udid = str(device.get("udid") or "")
            runtime = str(device.get("runtime") or "")
            state = (
                "device"
                if device.get("state") == "Booted"
                else str(device.get("state") or "unknown")
            )
            owner = ios_owners.get(DeviceExecutionLock._normalize_device_id(udid))
            devices.append(
                DeviceStatus(
                    serial=udid,
                    state=state,
                    model=device.get("name"),
                    product=runtime.removeprefix("com.apple.CoreSimulator.SimRuntime.").replace(
                        "-", " "
                    ),
                    is_emulator=True,
                    is_busy=owner is not None,
                    active_pid=owner.pid if owner else None,
                    active_task_desc=owner.description if owner else None,
                    active_session_id=owner.session_id if owner else None,
                    acquired_at=owner.acquired_at if owner else None,
                    platform="ios",
                )
            )
        return devices

    async def list_devices_async(self) -> list[DeviceStatus]:
        """All available simulators; an enumeration failure returns an empty list."""
        return self._build_statuses(await list_ios_simulators() or [])

    async def try_list_devices_async(self) -> list[DeviceStatus] | None:
        """Like list_devices_async, but ``None`` when simctl could not answer."""
        raw = await list_ios_simulators()
        return None if raw is None else self._build_statuses(raw)

    async def validate_explicit_serial_async(self, requested_serial: str) -> str | None:
        """Reject an explicitly requested UDID, or ``None`` when usable.

        Mirrors the Android validator: only a successful, non-empty
        enumeration may reject. ``Shutdown`` simulators are valid targets —
        the driver boots them on connect.
        """
        devices = await self.try_list_devices_async()
        if not devices:
            return None
        by_udid = {d.serial.lower(): d for d in devices}
        match = by_udid.get(str(requested_serial).lower())
        if match is None:
            return (
                f"iOS simulator '{requested_serial}' is not available. "
                f"Known simulators: {sorted(d.serial for d in devices)}."
            )
        if match.state not in ACCEPTABLE_STATES:
            return (
                f"iOS simulator '{requested_serial}' is in state '{match.state}' "
                "and cannot be used."
            )
        return None

    def validate_explicit_serial(self, requested_serial: str) -> str | None:
        """Synchronous validator for non-async admission paths (MCP tools)."""
        devices = list_ios_simulators_sync()
        if not devices:
            return None
        statuses = {d["udid"].lower(): d.get("state") for d in devices}
        state = statuses.get(str(requested_serial).lower())
        if state is None:
            return (
                f"iOS simulator '{requested_serial}' is not available. "
                f"Known simulators: {sorted(statuses)}."
            )
        if state != "Booted" and state != "Shutdown":
            return f"iOS simulator '{requested_serial}' is in state '{state}' and cannot be used."
        return None

    async def select_device_async(self, preferred_serial: str | None = None) -> str | None:
        """Pick a simulator UDID for task execution.

        An explicit serial wins as-is. Otherwise prefer an idle booted
        simulator, then any booted one, then the only available simulator
        when none are booted (the driver boots it on connect). ``None``
        when no simulator exists; strict ``booted`` ambiguity rejection is
        the driver's job for users who literally target "booted".
        """
        if preferred_serial:
            return preferred_serial
        devices = await self.try_list_devices_async() or []
        booted = [d for d in devices if d.state == "device"]
        if len(booted) == 1:
            return booted[0].serial
        if not booted and len(devices) == 1:
            return devices[0].serial
        idle_booted = [d for d in booted if not d.is_busy]
        return idle_booted[0].serial if idle_booted else (booted[0].serial if booted else None)


ios_device_pool = IosDevicePool()
