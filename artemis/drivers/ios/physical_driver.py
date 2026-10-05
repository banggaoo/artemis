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

"""Physical iPhone/iPad support via ``devicectl`` plus Xcode's native session.

Lifecycle operations (install, launch, terminate, app list, URL open) run
through ``xcrun devicectl``; observation and input reuse the same Xcode 27
``DeviceInteraction*`` session as the simulator driver — Apple's interaction
layer distinguishes ``deviceIsSimulator`` per session and drives paired
physical devices identically once the device is attached, trusted, and in
Developer Mode. Recording polls ``devicectl`` screenshots instead of
``simctl io recordVideo``, which does not exist for hardware.

Prerequisites surface as actionable errors: the device must appear in
``devicectl list devices`` as ``paired`` and ``connected`` (USB or network);
unpaired, offline, or non-iOS targets fail before any session starts.
"""

import json
from pathlib import Path
import plistlib
import re
from typing import Any

from artemis.drivers.ios.discovery import (
    is_physical_ios,
    list_core_devices,
    run_xcrun,
)
from artemis.drivers.ios.physical_recording import (
    IosPhysicalRecordingSession,
    PhysicalIosRecorder,
)
from artemis.drivers.ios.xcode_driver import XcodeSimulatorDriver
from third_party.mobile_use.utils.logger import get_logger
from third_party.mobile_use.utils.video import get_active_session, remove_active_session

logger = get_logger(__name__)

DEVICECTL_QUERY_TIMEOUT = 30.0
DEVICECTL_LAUNCH_TIMEOUT = 60.0
DEVICECTL_INSTALL_TIMEOUT = 300.0


class PhysicalIosDriver(XcodeSimulatorDriver):
    """One paired physical iOS device and one native interaction session.

    Inherits screenshots, hierarchy, taps, swipes, text, and keys from
    ``XcodeSimulatorDriver`` — the Xcode device-interaction session is shared.
    Only lifecycle plumbing differs: CoreDevice handles discovery, app
    install/launch/terminate, URL opening, and frame capture.
    """

    def __init__(
        self,
        device_id: str,
        width: int = 0,
        height: int = 0,
        *,
        workspace_path: str | Path | None = None,
    ):
        super().__init__(
            device_id=device_id, width=width, height=height, workspace_path=workspace_path
        )
        self._physical_recorder: PhysicalIosRecorder | None = None
        self._launched_pids: dict[str, int] = {}

    async def _resolve_device(self) -> dict[str, Any]:
        self._validate_workspace()
        if self._device_id in ("", "booted"):
            raise ValueError(
                "Physical iOS devices require --device-serial <device UDID>; "
                "'booted' only selects simulators. Find UDIDs via 'xcrun devicectl list devices'."
            )
        await self._require_ios_host()
        devices = await list_core_devices(force_refresh=True)
        if devices is None:
            raise RuntimeError(
                "Could not enumerate physical devices; 'xcrun devicectl list devices' failed."
            )
        needle = self._device_id.lower()
        matches = [
            device
            for device in devices
            if is_physical_ios(device)
            and (
                device.get("udid", "").lower() == needle
                or device.get("name") == self._device_id
            )
        ]
        if not matches:
            simulator = [
                device
                for device in devices
                if device.get("udid", "").lower() == needle
                and device.get("reality") == "simulated"
            ]
            if simulator:
                raise ValueError(
                    f"UDID {self._device_id} is an iOS Simulator, not a physical device."
                )
            raise ValueError(
                f"No paired physical iOS device matches {self._device_id!r}. "
                "Attach it, trust this Mac, and verify 'xcrun devicectl list devices'."
            )
        candidate = matches[0]
        self._device_id = candidate["udid"]
        return candidate

    async def _prepare_device(self, candidate: dict[str, Any]) -> None:
        """Verify the paired device is reachable instead of booting it."""
        if candidate.get("pairing_state") != "paired":
            raise RuntimeError(
                f"iOS device {self._device_id} is not paired. Connect it and tap Trust."
            )
        if candidate.get("connection_state") != "connected":
            raise RuntimeError(
                f"iOS device {self._device_id} ({candidate.get('name') or 'unknown'}) is not "
                "connected. Attach it over USB or ensure network pairing is reachable; "
                "on iOS 16+ also enable Developer Mode in Settings > Privacy & Security."
            )

    def _session_label(self) -> str:
        return f"Artemis Device {self._device_id[:8]}"

    def _validate_session_device(self, session: dict[str, Any]) -> None:
        if session.get("deviceIsSimulator"):
            raise RuntimeError(
                "Xcode opened a simulator session for a physical UDID; refusing to interact."
            )
        if session.get("deviceUUID", "").lower() != self._device_id.lower():
            raise RuntimeError("Xcode selected a different device; refusing to interact.")

    async def launch_app(self, package_name: str) -> bool:
        async with self._operation_lock:
            self._require_connected()
            raw = await run_xcrun(
                "devicectl",
                "device",
                "process",
                "launch",
                "--device",
                self._device_id,
                "--terminate-existing",
                "--json-output",
                "-",
                package_name,
                timeout=DEVICECTL_LAUNCH_TIMEOUT,
            )
            pid = self._launched_pid(raw)
            if pid is not None:
                self._launched_pids[package_name] = pid
        return True

    @staticmethod
    def _launched_pid(raw: bytes) -> int | None:
        """Best-effort process id from devicectl's launch JSON or text output."""
        try:
            payload = json.loads(raw)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict):
            result = payload.get("result", {})
            for key in ("processIdentifier", "pid"):
                pid = result.get(key) or payload.get(key)
                if isinstance(pid, int):
                    return pid
                if isinstance(pid, str) and pid.isdigit():
                    return int(pid)
        match = re.search(r"pid[:= ]+(\d+)", raw.decode(errors="replace"))
        return int(match.group(1)) if match else None

    async def _resolve_pid(self, package_name: str) -> int:
        """Find a running process id for the bundle, tracking launches first."""
        tracked = self._launched_pids.get(package_name)
        if tracked is not None:
            return tracked
        try:
            raw = await run_xcrun(
                "devicectl",
                "device",
                "info",
                "processes",
                "--device",
                self._device_id,
                "--json-output",
                "-",
                timeout=DEVICECTL_QUERY_TIMEOUT,
            )
            payload = json.loads(raw)
            for process in payload.get("result", {}).get("runningProcesses", []):
                executable = str(
                    process.get("executable") or process.get("bundleIdentifier") or ""
                )
                if package_name in executable and isinstance(
                    process.get("processIdentifier"), int
                ):
                    return process["processIdentifier"]
        except (OSError, RuntimeError, TimeoutError, ValueError, TypeError) as exc:
            logger.debug(f"devicectl process enumeration failed: {exc}")
        raise ValueError(
            f"No running process found for {package_name!r} on {self._device_id}; "
            "launch it with launch_app before stopping."
        )

    async def stop_app(self, package_name: str) -> bool:
        async with self._operation_lock:
            self._require_connected()
            pid = await self._resolve_pid(package_name)
            await run_xcrun(
                "devicectl",
                "device",
                "process",
                "terminate",
                "--device",
                self._device_id,
                "--pid",
                str(pid),
                timeout=DEVICECTL_QUERY_TIMEOUT,
            )
            self._launched_pids.pop(package_name, None)
        return True

    async def install_app(self, app_path: Path) -> str:
        self._require_connected()
        path = app_path.expanduser().resolve()
        if path.suffix == ".ipa":
            if not path.is_file():
                raise ValueError("iOS installation requires an existing .ipa file.")
            # The bundle id lives inside the archive; the filename stem is the
            # best available label until the app is installed and listed.
            bundle = path.stem
        elif path.suffix == ".app" and path.is_dir():
            with (path / "Info.plist").open("rb") as stream:
                bundle = plistlib.load(stream).get("CFBundleIdentifier")
            if not isinstance(bundle, str) or not bundle:
                raise ValueError("The .app has no CFBundleIdentifier in Info.plist.")
        else:
            raise ValueError(
                "Physical iOS installation requires a signed .app directory or .ipa built "
                "for a device (arm64) with a valid provisioning profile."
            )
        async with self._operation_lock:
            self._require_connected()
            await run_xcrun(
                "devicectl",
                "device",
                "install",
                "app",
                "--device",
                self._device_id,
                str(path),
                timeout=DEVICECTL_INSTALL_TIMEOUT,
            )
        return bundle

    async def list_apps(self) -> dict[str, str]:
        async with self._operation_lock:
            self._require_connected()
            raw = await run_xcrun(
                "devicectl",
                "device",
                "info",
                "apps",
                "--device",
                self._device_id,
                "--json-output",
                "-",
                timeout=DEVICECTL_QUERY_TIMEOUT,
            )
        payload = json.loads(raw)
        apps = payload.get("result", {}).get("apps", [])
        result = {}
        for app in apps:
            bundle = app.get("bundleIdentifier") or app.get("bundleID")
            if not bundle:
                continue
            result[bundle] = app.get("name") or bundle
        return result

    async def open_url(self, url: str) -> bool:
        async with self._operation_lock:
            self._require_connected()
            await run_xcrun(
                "devicectl",
                "device",
                "process",
                "openURL",
                "--device",
                self._device_id,
                url,
                timeout=DEVICECTL_QUERY_TIMEOUT,
            )
        return True

    async def execute_shell(self, command: str, timeout_seconds: float = 15.0) -> str:
        raise NotImplementedError("Android shell commands are unavailable on iOS devices.")

    # --- Recording (devicectl screenshot polling) ---

    @property
    def recording_session(self) -> IosPhysicalRecordingSession | None:
        if self._physical_recorder is None:
            return None
        return self._physical_recorder.session

    async def start_video_recording(
        self, output_dir: Path | None = None, max_duration_seconds: int = 900
    ) -> None:
        self._require_connected()
        if self._physical_recorder is None:
            self._physical_recorder = PhysicalIosRecorder(self._device_id)
        await self._physical_recorder.start(output_dir, max_duration_seconds)

    async def seal_recording_segment(self, through_time: float | None = None) -> None:
        if self._physical_recorder is not None:
            await self._physical_recorder.seal(through_time)

    async def stop_video_recording(self) -> str | None:
        if self._physical_recorder is None:
            return None
        path = await self._physical_recorder.stop()
        return str(path) if path is not None else None

    async def disconnect(self) -> None:
        try:
            if self._physical_recorder is not None:
                session = self._physical_recorder.session
                if session is not None:
                    if session.is_active:
                        try:
                            await self._physical_recorder.stop()
                        except Exception as exc:
                            logger.error(
                                f"Physical iOS recording finalization failed during disconnect: {exc}"
                            )
                    if get_active_session(self._device_id) is session:
                        remove_active_session(self._device_id)
        finally:
            await super().disconnect()
