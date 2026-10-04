# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Artemis System Diagnostics & Readiness Engine."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from artemis.core.diagnostics.engine import ReadinessEngine
from artemis.core.diagnostics.probes.adb_probe import AdbDeviceProbe
from artemis.core.diagnostics.probes.credentials_probe import (
    LLMCredentialsProbe,
    VisionOCRProbe,
)
from artemis.core.diagnostics.probes.runtime_probe import (
    PythonRuntimeProbe,
    SystemConfigProbe,
)
from artemis.core.diagnostics.probes.toolchain_probe import ToolchainProbe
from artemis.core.diagnostics.schema import (
    ProbeCategory,
    ProbeResult,
    ProbeStatus,
    SystemReadinessReport,
)


@pytest.mark.asyncio
async def test_readiness_engine_run_all():
    """Verify that ReadinessEngine aggregates probes and builds structured report."""
    engine = ReadinessEngine()
    report: SystemReadinessReport = await engine.run_all()

    assert isinstance(report, SystemReadinessReport)
    assert report.blocker_count >= 3
    assert isinstance(report.probes, list)
    assert len(report.probes) >= 5

    probe_ids = [p.id for p in report.probes]
    assert "python_runtime" in probe_ids
    assert "system_config" in probe_ids
    assert "android_adb" in probe_ids
    assert "gemini_api_key" in probe_ids
    assert "vision_ocr_key" in probe_ids
    assert "toolchain" in probe_ids


@pytest.mark.asyncio
async def test_python_runtime_probe_structure():
    """Verify PythonRuntimeProbe returns valid runtime inspection result."""
    probe = PythonRuntimeProbe()
    assert probe.probe_id == "python_runtime"
    assert probe.category == ProbeCategory.RUNTIME
    assert probe.is_blocker is True

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status in (ProbeStatus.PASS, ProbeStatus.FAIL)
    assert "version" in result.metadata
    assert "executable" in result.metadata


@pytest.mark.asyncio
async def test_system_config_probe_structure():
    """Verify SystemConfigProbe checks configuration file health."""
    probe = SystemConfigProbe()
    assert probe.probe_id == "system_config"
    assert probe.category == ProbeCategory.RUNTIME
    assert probe.is_blocker is True

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status in (ProbeStatus.PASS, ProbeStatus.FAIL)
    assert "valid" in result.metadata


@pytest.mark.asyncio
async def test_vision_ocr_probe_structure():
    """Verify VisionOCRProbe returns optional non-blocker status."""
    probe = VisionOCRProbe()
    assert probe.probe_id == "vision_ocr_key"
    assert probe.category == ProbeCategory.CREDENTIALS
    assert probe.is_blocker is False

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status == ProbeStatus.PASS
    assert "configured" in result.metadata


@pytest.mark.asyncio
async def test_adb_probe_structure():
    """Verify AdbDeviceProbe returns correct category, blocker status, and schema."""
    probe = AdbDeviceProbe()
    assert probe.probe_id == "android_adb"
    assert probe.category == ProbeCategory.DEVICE
    assert probe.is_blocker is True

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status in (ProbeStatus.PASS, ProbeStatus.WARN, ProbeStatus.FAIL)
    assert "installed" in result.metadata


@pytest.mark.parametrize(
    ("policy_output", "trust_output", "expected"),
    [
        (
            "KeyguardServiceDelegate\n  showing=true\n  occluded=false\n",
            'User "Owner" (current): deviceLocked=1',
            True,
        ),
        (
            "KeyguardServiceDelegate\n  showing=false\n  occluded=false\n",
            'User "Owner" (current): deviceLocked=0',
            False,
        ),
        (
            "mShowingLockscreen=true mKeyguardOccluded=false",
            "",
            True,
        ),
        ("", "", None),
    ],
)
def test_adb_probe_parses_device_lock_state(policy_output, trust_output, expected):
    """Keyguard and current-user trust signals produce a fail-safe lock state."""
    assert AdbDeviceProbe._parse_device_lock_state(policy_output, trust_output) is expected


def test_modern_unlock_state_is_not_overridden_by_legacy_fields():
    policy = (
        "KeyguardServiceDelegate\n  showing=false\nmShowingLockscreen=true mKeyguardOccluded=false"
    )
    trust = 'User "Owner" (current): deviceLocked=0'

    assert AdbDeviceProbe._parse_device_lock_state(policy, trust) is False


@pytest.mark.asyncio
async def test_positive_lock_state_requires_confirmation(monkeypatch):
    probe = AdbDeviceProbe()
    raw_probe = AsyncMock(side_effect=[True, None])
    monkeypatch.setattr(probe, "_get_device_lock_state", raw_probe)

    result = await probe._get_confirmed_device_lock_state("adb", "device-1")

    assert result is None
    assert raw_probe.await_count == 2


@pytest.mark.asyncio
async def test_dashboard_reuses_recent_confirmed_state_on_one_timeout(monkeypatch):
    probe = AdbDeviceProbe()
    confirmed_probe = AsyncMock(side_effect=[False, None])
    monkeypatch.setattr(probe, "_get_confirmed_device_lock_state", confirmed_probe)

    assert await probe._get_dashboard_lock_state("adb", "device-1") is False
    assert await probe._get_dashboard_lock_state("adb", "device-1") is False
    assert probe._lock_state_sources["device-1"] == "recent_confirmed"


@pytest.mark.asyncio
async def test_readiness_engine_coalesces_concurrent_full_scans():
    engine = ReadinessEngine()
    result = ProbeResult(
        id="test_probe",
        category=ProbeCategory.RUNTIME,
        title="Test",
        status=ProbeStatus.PASS,
        is_blocker=True,
        summary="Ready",
        description="Ready",
    )

    async def slow_probe():
        await asyncio.sleep(0.01)
        return result

    probe = Mock()
    probe.probe = AsyncMock(side_effect=slow_probe)
    engine._probes = {"test_probe": probe}

    reports = await asyncio.gather(*(engine.run_all() for _ in range(8)))

    assert probe.probe.await_count == 1
    assert all(report.overall_ready for report in reports)


@pytest.mark.asyncio
async def test_readiness_engine_reuses_cache_until_forced():
    engine = ReadinessEngine()
    result = ProbeResult(
        id="test_probe",
        category=ProbeCategory.RUNTIME,
        title="Test",
        status=ProbeStatus.PASS,
        is_blocker=True,
        summary="Ready",
        description="Ready",
    )
    probe = Mock()
    probe.probe = AsyncMock(return_value=result)
    engine._probes = {"test_probe": probe}

    first = await engine.run_all()
    cached = await engine.run_all()
    refreshed = await engine.run_all(force_refresh=True)

    assert probe.probe.await_count == 2
    assert cached.timestamp == first.timestamp
    assert refreshed.timestamp >= first.timestamp


@pytest.mark.asyncio
async def test_invalidation_prevents_in_flight_report_from_becoming_shared_cache():
    engine = ReadinessEngine()
    result = ProbeResult(
        id="test_probe",
        category=ProbeCategory.RUNTIME,
        title="Test",
        status=ProbeStatus.PASS,
        is_blocker=True,
        summary="Ready",
        description="Ready",
    )
    started = asyncio.Event()
    release = asyncio.Event()

    async def controlled_probe():
        started.set()
        await release.wait()
        return result

    probe = Mock()
    probe.probe = AsyncMock(side_effect=controlled_probe)
    engine._probes = {"test_probe": probe}

    old_scan = asyncio.create_task(engine.run_all())
    await started.wait()
    engine.invalidate_cache()
    release.set()
    await old_scan
    await engine.run_all()

    assert probe.probe.await_count == 2


@pytest.mark.asyncio
async def test_submission_probe_skips_full_device_enrichment(monkeypatch):
    probe = AdbDeviceProbe(target_serial="device-2")
    get_states = AsyncMock(return_value=[("device-1", "device"), ("device-2", "device")])
    get_lock_state = AsyncMock(return_value=False)
    full_probe = AsyncMock()
    monkeypatch.setattr(
        "artemis.core.diagnostics.probes.adb_probe.toolchain.resolve",
        lambda name: "adb",
    )
    monkeypatch.setattr(probe, "_get_device_states", get_states)
    monkeypatch.setattr(probe, "_get_device_lock_state", get_lock_state)
    monkeypatch.setattr(probe, "_parse_adb_devices", full_probe)

    result = await probe.probe_submission_readiness()

    assert result.summary == "Connected"
    assert result.metadata["submission_probe"] is True
    get_states.assert_awaited_once_with("adb")
    get_lock_state.assert_awaited_once_with("adb", "device-2", timeout_seconds=1.0)
    full_probe.assert_not_awaited()


@pytest.mark.asyncio
async def test_submission_probe_fails_closed_when_lock_state_is_unknown(monkeypatch):
    probe = AdbDeviceProbe()
    monkeypatch.setattr(
        "artemis.core.diagnostics.probes.adb_probe.toolchain.resolve",
        lambda name: "adb",
    )
    monkeypatch.setattr(
        probe,
        "_get_device_states",
        AsyncMock(return_value=[("device-1", "device")]),
    )
    monkeypatch.setattr(
        probe,
        "_get_device_lock_state",
        AsyncMock(return_value=None),
    )

    result = await probe.probe_submission_readiness()

    assert result.summary == "Lock State Unknown"
    assert result.status == ProbeStatus.WARN


@pytest.mark.asyncio
async def test_submission_probe_falls_back_to_unlocked_device(monkeypatch):
    """When the first device is locked but a second device is unlocked, submission probe falls back."""
    probe = AdbDeviceProbe()
    monkeypatch.setattr(
        probe,
        "_get_device_states",
        AsyncMock(return_value=[("device-locked", "device"), ("device-unlocked", "device")]),
    )

    async def mock_lock_state(adb_path, serial, timeout_seconds=1.0):
        return True if serial == "device-locked" else False

    monkeypatch.setattr(probe, "_get_confirmed_device_lock_state", mock_lock_state)

    result = await probe.probe_submission_readiness()

    assert result.status == ProbeStatus.PASS
    assert result.summary == "Connected"
    assert result.metadata["active_device"]["serial"] == "device-unlocked"
    assert result.metadata["active_device"]["is_locked"] is False


@pytest.mark.asyncio
async def test_adb_probe_prefers_unlocked_device_when_one_is_locked(monkeypatch):
    """When multiple ready devices exist, probe() should pick the unlocked one as active."""
    from artemis.core.diagnostics.schema import DeviceInfo

    probe = AdbDeviceProbe()
    monkeypatch.setattr(probe, "_locate_adb", lambda: "/usr/bin/adb")
    monkeypatch.setattr(probe, "_get_adb_version", AsyncMock(return_value="1.0.41"))
    monkeypatch.setattr(probe, "_locate_emulator", lambda: "/usr/bin/emulator")
    monkeypatch.setattr(probe, "_list_installed_avds", lambda _: [])

    devices = [
        DeviceInfo(serial="dev-locked-1", state="device", model="Pixel 7", is_locked=True),
        DeviceInfo(serial="dev-unlocked-2", state="device", model="Pixel 8", is_locked=False),
    ]
    monkeypatch.setattr(probe, "_parse_adb_devices", AsyncMock(return_value=devices))

    result = await probe.probe()

    assert result.status == ProbeStatus.PASS
    assert result.summary == "Connected"
    assert result.metadata["active_device"]["serial"] == "dev-unlocked-2"
    assert result.metadata["active_device"]["is_locked"] is False


@pytest.mark.asyncio
async def test_llm_credentials_probe_structure():
    """Verify LLMCredentialsProbe returns correct category and schema."""
    probe = LLMCredentialsProbe()
    assert probe.probe_id == "gemini_api_key"
    assert probe.category == ProbeCategory.CREDENTIALS
    assert probe.is_blocker is True

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status in (ProbeStatus.PASS, ProbeStatus.FAIL)
    assert "configured_count" in result.metadata


@pytest.mark.asyncio
async def test_toolchain_probe_structure():
    """Verify ToolchainProbe returns valid probe category, metadata, and schema."""
    probe = ToolchainProbe()
    assert probe.probe_id == "toolchain"
    assert probe.category == ProbeCategory.TOOLCHAIN
    assert probe.is_blocker is False

    result: ProbeResult = await probe.probe()
    assert isinstance(result, ProbeResult)
    assert result.status in (ProbeStatus.PASS, ProbeStatus.FAIL)
    assert "ffmpeg" in result.metadata
    assert "scrcpy" in result.metadata


@pytest.mark.asyncio
async def test_probe_target_serial_forwards_to_adb_probe():
    """Verify the probe target preference reaches the ADB probe and can be cleared."""
    engine = ReadinessEngine()
    engine.set_probe_target_serial("test-emulator-1234")
    assert engine._adb_probe._target_serial == "test-emulator-1234"
    engine.set_probe_target_serial(None)
    assert engine._adb_probe._target_serial is None


@pytest.mark.asyncio
async def test_credentials_probe_and_dynamic_update():
    """Verify dynamic API key updates and metadata reflection."""
    from artemis.config import settings

    settings.set_api_key("google", "test_gemini_key_1234567890", persist_to_env=False)

    probe = LLMCredentialsProbe()
    result = await probe.probe()
    assert result.status == ProbeStatus.PASS
    assert "current_key" in result.metadata
    assert result.metadata["current_key"] == "test_gemini_key_1234567890"
    assert "api_keys" in result.metadata
    assert result.metadata["api_keys"]["google"] == "test_gemini_key_1234567890"


@pytest.mark.asyncio
async def test_emulator_manager_lifecycle():
    """Verify EmulatorManager status querying, validation, and dismissal."""
    from artemis.core.diagnostics.emulator_manager import (
        EmulatorLaunchStage,
        EmulatorManager,
    )

    manager = EmulatorManager()
    status = manager.get_status()
    assert status.status == EmulatorLaunchStage.IDLE

    # Invalid empty AVD name
    empty_res = await manager.launch("   ")
    assert empty_res.status == EmulatorLaunchStage.FAILED
    assert "empty" in (empty_res.error or "").lower()

    # Dismiss state
    dismiss_res = manager.dismiss()
    assert dismiss_res["success"] is True
    assert manager.get_status().status == EmulatorLaunchStage.IDLE


def _fake_proc(pid: int = 1111, alive_polls: int = 5, exit_code: int = 1):
    """Popen-shaped mock that reports alive for `alive_polls` polls then exits."""
    proc = Mock()
    proc.pid = pid
    calls = {"n": 0}

    def _poll():
        calls["n"] += 1
        return None if calls["n"] <= alive_polls else exit_code

    proc.poll.side_effect = _poll
    proc.stdout.readline.return_value = ""
    return proc


@pytest.mark.asyncio
async def test_emulator_spawn_passes_extra_args(monkeypatch):
    """_spawn_emulator appends extra emulator CLI args and tracks the proc."""
    import subprocess
    import time

    from artemis.core.diagnostics.emulator_manager import EmulatorManager

    manager = EmulatorManager()
    spawned = {}

    def _fake_popen(cmd, **kwargs):
        spawned["cmd"] = cmd
        return _fake_proc()

    monkeypatch.setattr(subprocess, "Popen", _fake_popen)
    proc = manager._spawn_emulator("/sdk/emulator", "Pixel_9", ["-no-snapshot-load"])

    assert spawned["cmd"] == ["/sdk/emulator", "-avd", "Pixel_9", "-no-snapshot-load"]
    assert manager._proc is proc


def test_emulator_snapshot_failure_detection():
    from artemis.core.diagnostics.emulator_manager import EmulatorManager

    manager = EmulatorManager()
    assert manager._log_reports_snapshot_failure() is False

    manager._log_buffer.append("INFO | Advertising in: /tmp/pid_1.ini")
    assert manager._log_reports_snapshot_failure() is False

    manager._log_buffer.append(
        "WARNING | Device 'encrypt' does not have the requested snapshot 'default_boot'"
    )
    assert manager._log_reports_snapshot_failure() is True

    manager._log_buffer.clear()
    manager._log_buffer.append("WARNING | Failed to load snapshot 'default_boot'")
    assert manager._log_reports_snapshot_failure() is True


@pytest.mark.asyncio
async def test_emulator_snapshot_retry_restarts_without_load(monkeypatch):
    """A corrupted snapshot triggers one restart with -no-snapshot-load."""
    import time

    from artemis.core.diagnostics.emulator_manager import EmulatorManager

    manager = EmulatorManager()
    old_proc = _fake_proc(pid=1)
    new_proc = _fake_proc(pid=2)
    calls = {}

    monkeypatch.setattr(manager, "_locate_emulator", lambda: "/sdk/emulator")
    monkeypatch.setattr(manager, "_terminate_proc", lambda p: calls.setdefault("term", p))

    def _spawn(path, avd, extra):
        calls["spawn"] = (path, avd, extra)
        return new_proc

    monkeypatch.setattr(manager, "_spawn_emulator", _spawn)
    tracked = AsyncMock()
    monkeypatch.setattr(manager, "_track_boot_lifecycle", tracked)

    await manager._retry_without_snapshot("Pixel_9", old_proc)

    assert calls["term"] is old_proc
    assert calls["spawn"] == ("/sdk/emulator", "Pixel_9", ["-no-snapshot-load"])
    assert manager._current_state.pid == 2
    tracked.assert_awaited_once()
    args, kwargs = tracked.await_args
    assert args[0] == "Pixel_9" and args[1] is new_proc
    assert kwargs["allow_snapshot_retry"] is False
    assert kwargs["max_wait_seconds"] == 480


@pytest.mark.asyncio
async def test_emulator_tracker_retries_on_snapshot_failure(monkeypatch):
    """Phase-1 detection invokes the snapshot retry and returns."""
    import time

    from artemis.core.diagnostics.emulator_manager import EmulatorManager

    manager = EmulatorManager()
    manager._log_buffer.append("WARNING | Failed to load snapshot 'default_boot'")
    retry = AsyncMock()
    monkeypatch.setattr(manager, "_retry_without_snapshot", retry)
    monkeypatch.setattr(manager, "_locate_adb", lambda: "adb")

    proc = _fake_proc(alive_polls=999)
    await manager._track_boot_lifecycle("Pixel_9", proc, time.time())

    retry.assert_awaited_once_with("Pixel_9", proc)


@pytest.mark.asyncio
async def test_emulator_tracker_ignores_offline_serial(monkeypatch):
    """`adb devices` entries in `offline` state do not count as connected."""
    import asyncio
    import time

    from artemis.core.diagnostics.emulator_manager import (
        EmulatorLaunchStage,
        EmulatorManager,
    )

    manager = EmulatorManager()
    monkeypatch.setattr(manager, "_locate_adb", lambda: "adb")

    adb_proc = Mock()
    adb_proc.communicate = AsyncMock(return_value=(b"emulator-5554\toffline\n", b""))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=adb_proc))

    # Alive through Phase 1, dies on the third Phase-2 poll.
    proc = _fake_proc(alive_polls=7, exit_code=1)
    await manager._track_boot_lifecycle("Pixel_9", proc, time.time(), max_wait_seconds=60)

    state = manager._current_state
    assert state.status == EmulatorLaunchStage.FAILED
    assert state.serial is None
    assert "terminated unexpectedly" in (state.error or "")


@pytest.mark.asyncio
async def test_emulator_tracker_accepts_device_serial(monkeypatch):
    """`adb devices` entries in `device` state advance to the boot phase."""
    import asyncio
    import time

    from artemis.core.diagnostics.emulator_manager import (
        EmulatorLaunchStage,
        EmulatorManager,
    )

    manager = EmulatorManager()
    monkeypatch.setattr(manager, "_locate_adb", lambda: "adb")

    adb_proc = Mock()
    adb_proc.communicate = AsyncMock(
        return_value=(
            b"emulator-5554\tdevice product:sdk model:sdk device:emu64a\n",
            b"",
        )
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=adb_proc))

    # Alive through Phase 1 + first Phase-2 poll, dies in Phase 3.
    proc = _fake_proc(alive_polls=7, exit_code=1)
    await manager._track_boot_lifecycle("Pixel_9", proc, time.time(), max_wait_seconds=60)

    state = manager._current_state
    assert state.status == EmulatorLaunchStage.FAILED
    assert state.serial == "emulator-5554"
    assert "crashed during OS boot" in (state.error or "")


@pytest.mark.asyncio
async def test_build_report_turns_crashing_probe_into_fail_result():
    engine = ReadinessEngine()
    healthy = ProbeResult(
        id="healthy",
        category=ProbeCategory.RUNTIME,
        title="Healthy",
        status=ProbeStatus.PASS,
        is_blocker=True,
        summary="Ready",
        description="Ready",
    )
    good = Mock()
    good.probe_id = "healthy"
    good.category = ProbeCategory.RUNTIME
    good.is_blocker = True
    good.probe = AsyncMock(return_value=healthy)

    bad = Mock()
    bad.probe_id = "integration_host"
    bad.category = ProbeCategory.RUNTIME
    bad.is_blocker = True
    bad.probe = AsyncMock(side_effect=PermissionError(13, "Permission denied", "/ro/traces"))
    engine._probes = {"healthy": good, "integration_host": bad}

    report = await engine._build_report()

    assert report.overall_ready is False
    assert report.blocker_count == 2
    assert report.passed_blocker_count == 1
    by_id = {r.id: r for r in report.probes}
    assert by_id["healthy"].status is ProbeStatus.PASS
    crashed = by_id["integration_host"]
    assert crashed.status is ProbeStatus.FAIL
    assert crashed.is_blocker is True
    assert crashed.category is ProbeCategory.RUNTIME
    assert crashed.summary == "Probe crashed"
    assert "PermissionError" in crashed.description
    assert "Permission denied" in crashed.description
    assert crashed.metadata["exception_type"] == "PermissionError"


@pytest.mark.asyncio
async def test_build_report_turns_hung_probe_into_fail_result(monkeypatch):
    """A probe that never returns (wedged ADB server) is cut off at the
    engine's per-probe deadline and reported as a FAIL with the restart step,
    so no surface running the report can hang."""
    monkeypatch.setattr(ReadinessEngine, "PROBE_TIMEOUT_SECONDS", 0.05)
    engine = ReadinessEngine()

    async def never_returns():
        await asyncio.sleep(10)

    hung = Mock()
    hung.probe_id = "android_adb"
    hung.category = ProbeCategory.DEVICE
    hung.is_blocker = True
    hung.probe = never_returns
    engine._probes = {"android_adb": hung}

    report = await engine._build_report()

    result = report.probes[0]
    assert result.status is ProbeStatus.FAIL
    assert result.summary == "Probe timed out"
    assert result.metadata["exception_type"] == "TimeoutError"
    assert any("adb kill-server" in a.payload for a in result.actions)
    assert report.overall_ready is False


@pytest.mark.asyncio
async def test_build_report_does_not_swallow_cancellation():
    engine = ReadinessEngine()
    probe = Mock()
    probe.probe_id = "cancelled"
    probe.category = ProbeCategory.RUNTIME
    probe.is_blocker = True
    probe.probe = AsyncMock(side_effect=asyncio.CancelledError())
    engine._probes = {"cancelled": probe}

    with pytest.raises(asyncio.CancelledError):
        await engine._build_report()
