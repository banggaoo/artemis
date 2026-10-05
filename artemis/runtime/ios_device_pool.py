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

"""iOS device pool: simulator and physical discovery, locks, selection.

The iOS counterpart of :class:`~artemis.runtime.device_pool.DevicePool`.
Enumeration rides ``xcrun simctl list devices`` for simulators and
``xcrun devicectl list devices`` for paired physical hardware; lock ownership
shares the same :class:`DeviceExecutionLock` registry under the ``ios`` scope
so a simulator, a physical device, and an Android serial can never share a
lock identity. Auto-selection only ever picks simulators — physical hardware
requires an explicit serial. All methods fail open on missing Xcode tooling
or an enumeration error, mirroring the Android pool's admission contract.
"""

from __future__ import annotations

from artemis.drivers.ios.discovery import (
    is_physical_ios,
    list_core_devices,
    list_core_devices_sync,
    list_ios_simulators,
    list_ios_simulators_sync,
)
from artemis.runtime.adb_endpoint import IOS_LOCK_SCOPE
from artemis.runtime.device_lock import DeviceExecutionLock
from artemis.runtime.device_pool import DeviceStatus
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

# simctl states the driver can use as-is ("device"/Booted) or boot itself
# ("Shutdown"). Anything else (e.g. "Creating") is rejected on explicit picks.
ACCEPTABLE_STATES = frozenset({"device", "Shutdown"})

# devicectl states that allow driving a physical device right now.
PHYSICAL_ACCEPTABLE_STATES = frozenset({"device"})


class IosDevicePool:
    """Discovers iOS simulators and physical devices and reports lock state."""

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

    @staticmethod
    def _build_physical_statuses(raw_devices: list[dict]) -> list[DeviceStatus]:
        """CoreDevice entries -> statuses; reachable devices read as "device"."""
        active_owners = DeviceExecutionLock.get_active_owners()
        ios_owners = {
            DeviceExecutionLock._normalize_device_id(o.device_id): o
            for o in active_owners.values()
            if o and getattr(o, "lock_scope", None) == IOS_LOCK_SCOPE
        }
        devices: list[DeviceStatus] = []
        for device in raw_devices:
            if not is_physical_ios(device):
                continue
            udid = str(device.get("udid") or "")
            paired = device.get("pairing_state") == "paired"
            connected = device.get("connection_state") == "connected"
            state = "device" if connected and paired else "unpaired" if not paired else "offline"
            owner = ios_owners.get(DeviceExecutionLock._normalize_device_id(udid))
            devices.append(
                DeviceStatus(
                    serial=udid,
                    state=state,
                    model=device.get("name"),
                    product=f"iOS {device.get('os_version') or '?'} physical",
                    is_emulator=False,
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
        """All simulators plus physical devices; enumeration failures degrade to []."""
        simulators = self._build_statuses(await list_ios_simulators() or [])
        physical = self._build_physical_statuses(await list_core_devices() or [])
        return simulators + physical

    async def try_list_devices_async(self) -> list[DeviceStatus] | None:
        """Like list_devices_async, but ``None`` when enumeration cannot answer."""
        raw = await list_ios_simulators()
        core = await list_core_devices()
        if raw is None and core is None:
            return None
        return self._build_statuses(raw or []) + self._build_physical_statuses(core or [])

    async def validate_explicit_serial_async(self, requested_serial: str) -> str | None:
        """Reject an explicitly requested UDID or device name, ``None`` when usable.

        Mirrors the Android validator: only a successful, non-empty
        enumeration may reject, and an enumeration that cannot answer (None)
        must not reject. ``Shutdown`` simulators are valid targets — the
        driver boots them on connect. Physical devices must be paired and
        connected (``"device"``); offline devices fail here.
        """
        raw = await list_ios_simulators()
        core = await list_core_devices()
        if raw is None and core is None:
            return None
        devices = self._build_statuses(raw or []) + self._build_physical_statuses(
            core or []
        )
        needle = str(requested_serial).lower()
        match = next(
            (
                d
                for d in devices
                if d.serial.lower() == needle or (d.model or "") == requested_serial
            ),
            None,
        )
        if match is None:
            # A failed enumeration cannot prove the serial is absent —
            # defer to the driver's own resolution rather than reject.
            if raw is None or core is None:
                return None
            return (
                f"iOS device '{requested_serial}' is not available. "
                f"Known devices: {sorted(d.serial for d in devices)}."
            )
        acceptable = (
            ACCEPTABLE_STATES if match.is_emulator else PHYSICAL_ACCEPTABLE_STATES
        )
        if match.state not in acceptable:
            kind = "simulator" if match.is_emulator else "physical device"
            return (
                f"iOS {kind} '{requested_serial}' is in state '{match.state}' "
                "and cannot be used."
            )
        return None

    def validate_explicit_serial(self, requested_serial: str) -> str | None:
        """Synchronous validator for non-async admission paths (MCP tools)."""
        devices = list_ios_simulators_sync()
        statuses = {d["udid"].lower(): d.get("state") for d in devices or []}
        needle = str(requested_serial).lower()
        state = statuses.get(needle)
        if state is not None:
            if state != "Booted" and state != "Shutdown":
                return (
                    f"iOS simulator '{requested_serial}' is in state '{state}' "
                    "and cannot be used."
                )
            return None
        core = list_core_devices_sync()
        physical = [d for d in core or [] if is_physical_ios(d)]
        match = next(
            (
                d
                for d in physical
                if str(d.get("udid") or "").lower() == needle
                or d.get("name") == requested_serial
            ),
            None,
        )
        if match is None:
            # An enumeration that could not answer must not reject the serial.
            if devices is None or core is None:
                return None
            if not statuses and not physical:
                return None
            return (
                f"iOS device '{requested_serial}' is not available. "
                f"Known devices: {sorted([*statuses, *[d['udid'] for d in physical]])}."
            )
        if match.get("pairing_state") != "paired":
            return f"iOS device '{requested_serial}' is not paired with this Mac."
        if match.get("connection_state") != "connected":
            return f"iOS device '{requested_serial}' is offline; attach it and retry."
        return None

    async def select_device_async(self, preferred_serial: str | None = None) -> str | None:
        """Pick a simulator UDID for task execution.

        An explicit serial wins as-is. Otherwise prefer an idle booted
        simulator, then any booted one, then the only available simulator
        when none are booted (the driver boots it on connect). Physical
        devices are never auto-selected — hardware always needs an explicit
        serial. ``None`` when no simulator exists; strict ``booted``
        ambiguity rejection is the driver's job for users who literally
        target "booted".
        """
        if preferred_serial:
            return preferred_serial
        devices = await self.try_list_devices_async() or []
        simulators = [d for d in devices if d.is_emulator]
        booted = [d for d in simulators if d.state == "device"]
        if len(booted) == 1:
            return booted[0].serial
        if not booted and len(simulators) == 1:
            return simulators[0].serial
        idle_booted = [d for d in booted if not d.is_busy]
        return idle_booted[0].serial if idle_booted else (booted[0].serial if booted else None)


ios_device_pool = IosDevicePool()
