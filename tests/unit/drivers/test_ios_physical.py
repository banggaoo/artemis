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

"""Physical iOS device behavior tested without attached hardware."""

import asyncio
import io
import json
from pathlib import Path
import plistlib
from types import SimpleNamespace

import pytest

from artemis.drivers.ios import discovery, physical_driver, physical_recording, wda
from artemis.drivers.ios.physical_driver import PhysicalIosDriver
from artemis.drivers.ios.physical_recording import PhysicalIosRecorder
from artemis.drivers.ios.wda import WdaClient


IPHONE_UDID = "00008130-000245193C60001C"
IPAD_UDID = "00008103-00197094013B001E"
SIM_UDID = "DE345DD3-5792-4DAD-B863-144682629565"
WATCH_UDID = "00008301-D8814D603680202E"


def _devicectl_device(
    udid,
    name="Device",
    os_version="27.0",
    platform="iOS",
    reality="physical",
    tunnel="connected",
    pairing="paired",
    visibility="default",
):
    """One device entry in devicectl's deprecated split-properties shape."""
    return {
        "deviceProperties": {"name": name, "osVersionNumber": os_version},
        "hardwareProperties": {"udid": udid, "platform": platform, "reality": reality},
        "connectionProperties": {"tunnelState": tunnel, "pairingState": pairing},
        "visibilityClass": visibility,
    }


def _devicectl_payload(*devices):
    return {"info": {}, "result": {"devices": list(devices)}}


def _parsed(*devices):
    """Flattened entries as `parse_devicectl_devices` emits them."""
    return discovery.parse_devicectl_devices(_devicectl_payload(*devices))


PHYSICAL_IPHONE = _devicectl_device(IPHONE_UDID, name="Jane's iPhone")
OFFLINE_IPHONE = _devicectl_device(IPHONE_UDID, name="iPhone", tunnel="disconnected")
UNPAIRED_IPHONE = _devicectl_device(IPHONE_UDID, name="iPhone", pairing="unpaired")
SIMULATOR = _devicectl_device(
    SIM_UDID, name="iPhone 18 Pro", reality="simulated", visibility="simulators"
)
WATCH = _devicectl_device(WATCH_UDID, name="Watch", platform="watchOS")


def test_devicectl_devices_flatten_physical_and_simulator_entries():
    devices = discovery.parse_devicectl_devices(
        _devicectl_payload(PHYSICAL_IPHONE, SIMULATOR, WATCH)
    )
    by_udid = {device["udid"]: device for device in devices}
    assert by_udid[IPHONE_UDID]["reality"] == "physical"
    assert by_udid[IPHONE_UDID]["platform"] == "iOS"
    assert by_udid[SIM_UDID]["reality"] == "simulated"
    assert by_udid[WATCH_UDID]["platform"] == "watchOS"


def test_devicectl_devices_read_the_nested_properties_shape():
    """devicectl is migrating split property dicts into one `properties` map."""
    nested = {
        "properties": {
            "deviceProperties": {"name": "iPhone", "osVersionNumber": "27.0"},
            "hardwareProperties": {"udid": IPHONE_UDID, "platform": "iOS", "reality": "physical"},
            "connectionProperties": {"tunnelState": "connected", "pairingState": "paired"},
        }
    }
    devices = discovery.parse_devicectl_devices(_devicectl_payload(nested))
    assert devices == [
        {
            "udid": IPHONE_UDID,
            "name": "iPhone",
            "os_version": "27.0",
            "platform": "iOS",
            "reality": "physical",
            "product_type": None,
            "connection_state": "connected",
            "pairing_state": "paired",
            "visibility": None,
        }
    ]


def test_devicectl_devices_skip_entries_without_udid():
    devices = discovery.parse_devicectl_devices(
        _devicectl_payload({"hardwareProperties": {"platform": "iOS"}}, PHYSICAL_IPHONE)
    )
    assert [device["udid"] for device in devices] == [IPHONE_UDID]


def test_physical_classification_requires_ios_and_hardware():
    assert discovery.is_physical_ios({"platform": "iOS", "reality": "physical"})
    assert not discovery.is_physical_ios({"platform": "iOS", "reality": "simulated"})
    assert not discovery.is_physical_ios({"platform": "watchOS", "reality": "physical"})
    assert not discovery.is_physical_ios({"platform": "iOS"})


def test_find_physical_device_matches_udid_case_insensitively(monkeypatch):
    monkeypatch.setattr(
        discovery, "list_core_devices_sync", lambda force_refresh=False: _parsed(PHYSICAL_IPHONE, SIMULATOR)
    )
    found = discovery.find_physical_ios_device_sync(IPHONE_UDID.lower())
    assert found is not None and found["udid"] == IPHONE_UDID
    assert discovery.find_physical_ios_device_sync("Jane's iPhone") is not None
    assert discovery.find_physical_ios_device_sync(SIM_UDID) is None
    assert discovery.find_physical_ios_device_sync("missing") is None


def test_find_physical_device_survives_enumeration_failure(monkeypatch):
    monkeypatch.setattr(discovery, "list_core_devices_sync", lambda force_refresh=False: None)
    assert discovery.find_physical_ios_device_sync(IPHONE_UDID) is None


@pytest.fixture
def driver():
    return PhysicalIosDriver(device_id=IPHONE_UDID)


@pytest.fixture(autouse=True)
def _clean_wda_env(monkeypatch):
    for key in (
        'ARTEMIS_IOS_WDA_URL',
        'ARTEMIS_IOS_WDA_HOST',
        'ARTEMIS_IOS_WDA_XCTESTRUN',
        'ARTEMIS_IOS_WDA_BUNDLE_ID',
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
async def test_physical_resolve_rejects_booted_and_empty_serials(driver):
    driver._device_id = "booted"
    with pytest.raises(ValueError, match="device-serial"):
        await driver._resolve_device()


@pytest.mark.asyncio
async def test_physical_resolve_pins_the_udid(driver, monkeypatch):
    monkeypatch.setattr(physical_driver, "list_core_devices", _async_devices(_parsed(PHYSICAL_IPHONE, SIMULATOR)))
    candidate = await driver._resolve_device()
    assert candidate["udid"] == IPHONE_UDID
    assert driver.device_id == IPHONE_UDID


@pytest.mark.asyncio
async def test_physical_resolve_rejects_simulator_udids(driver, monkeypatch):
    driver._device_id = SIM_UDID
    monkeypatch.setattr(physical_driver, "list_core_devices", _async_devices(_parsed(SIMULATOR)))
    with pytest.raises(ValueError, match="Simulator"):
        await driver._resolve_device()


@pytest.mark.asyncio
async def test_physical_resolve_rejects_non_ios_hardware(driver, monkeypatch):
    driver._device_id = WATCH_UDID
    monkeypatch.setattr(physical_driver, "list_core_devices", _async_devices(_parsed(WATCH)))
    with pytest.raises(ValueError, match="paired physical iOS"):
        await driver._resolve_device()


@pytest.mark.asyncio
async def test_physical_resolve_reports_enumeration_failure(driver, monkeypatch):
    monkeypatch.setattr(physical_driver, "list_core_devices", _async_devices(None))
    with pytest.raises(RuntimeError, match="devicectl"):
        await driver._resolve_device()


@pytest.mark.asyncio
async def test_offline_device_fails_with_actionable_error(driver):
    with pytest.raises(RuntimeError, match="not connected|Developer Mode"):
        await driver._prepare_device(_parsed(OFFLINE_IPHONE)[0])


@pytest.mark.asyncio
async def test_unpaired_device_fails_with_trust_guidance(driver):
    with pytest.raises(RuntimeError, match="not paired|Trust"):
        await driver._prepare_device(_parsed(UNPAIRED_IPHONE)[0])


@pytest.mark.asyncio
async def test_connected_device_needs_no_boot(driver):
    await driver._prepare_device(_parsed(PHYSICAL_IPHONE)[0])


def _async_devices(result):
    async def _list(force_refresh=False):
        return result

    return _list


def _xcrun_payload_writer(payloads):
    """Fake run_xcrun that honors ``--json-output <file>`` vs ``-``.

    ``payloads`` maps a devicectl subcommand ("apps", "processes", "details",
    "launch", ...) to the dict written into the output file. Anything else
    lands on stdout so ``--json-output -`` callers get JSON bytes.
    """

    async def fake(*arguments, timeout=30.0):
        args = list(arguments)
        if "--json-output" in args:
            target = args[args.index("--json-output") + 1]
            keys = [a for a in args if a in payloads]
            payload = payloads.get(keys[0], {}) if keys else {}
            if target == "-":
                return json.dumps(payload).encode()
            Path(target).write_text(json.dumps(payload), encoding="utf-8")
            return b""
        return b""

    return fake


class _FakeWda:
    """In-memory WebDriverAgent stand-in for driver interaction tests."""

    def __init__(self, window=(100.0, 200.0), tree=None):
        self.session_id = None
        self.base_url = "http://fake-wda:8100"
        self.window = window
        self.tree = tree if tree is not None else {"type": "Application", "children": []}
        self.tapped: list[tuple[float, float, int]] = []
        self.swiped: list[tuple[float, float, float, float, int]] = []
        self.typed: list[str] = []
        self.buttons: list[str] = []
        self.homescreen_calls = 0
        self._png = self._make_png()
        self._closed = 0

    @staticmethod
    def _make_png(width: int = 300, height: int = 600) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (width, height), color="red").save(buffer, format="PNG")
        return buffer.getvalue()

    async def open_session(self):
        self.session_id = "wda-session"
        return self.session_id

    async def close_session(self):
        self._closed += 1
        self.session_id = None

    async def screenshot_png(self):
        return self._png

    async def window_size(self):
        return self.window

    async def source_json(self):
        return self.tree

    async def active_app(self):
        return "com.example.foreground"

    async def tap(self, x, y, hold_ms=0):
        self.tapped.append((x, y, hold_ms))

    async def swipe(self, sx, sy, ex, ey, duration_ms):
        self.swiped.append((sx, sy, ex, ey, duration_ms))

    async def type_text(self, text):
        self.typed.append(text)

    async def press_button(self, name):
        self.buttons.append(name)
        return True

    async def homescreen(self):
        self.homescreen_calls += 1


@pytest.fixture
def connected_driver():
    """A driver with a stub WDA session and a known 300x600 observation."""
    driver = PhysicalIosDriver(device_id=IPHONE_UDID)
    driver._session_key = "wda-session"
    driver._wda = _FakeWda()
    # Pretend one capture already ran: the orientation guard compares sizes.
    driver._width, driver._height = 300, 600
    return driver


@pytest.mark.asyncio
async def test_launch_records_the_pid_from_devicectl_json(connected_driver, monkeypatch):
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return json.dumps(
            {"result": {"process": {"processIdentifier": 4242,
                                    "executable": "file:///Apps/Example.app/Example"}}}
        ).encode()

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.launch_app("com.example.app")
    assert calls[0][:5] == ("devicectl", "device", "process", "launch", "--device")
    assert IPHONE_UDID in calls[0]
    assert calls[0][-1] == "com.example.app"
    assert connected_driver._launched_pids == {"com.example.app": 4242}


@pytest.mark.asyncio
async def test_stop_app_terminates_the_tracked_pid(connected_driver, monkeypatch):
    connected_driver._launched_pids["com.example.app"] = 4242
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return b"{}"

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.stop_app("com.example.app")
    terminate = calls[0]
    assert terminate[:5] == ("devicectl", "device", "process", "terminate", "--device")
    assert terminate[terminate.index("--pid") + 1] == "4242"
    assert connected_driver._launched_pids == {}


@pytest.mark.asyncio
async def test_stop_app_without_pid_scans_running_processes(connected_driver, monkeypatch):
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        args = list(arguments)
        calls.append(args)
        if "--json-output" in args and args[args.index("--json-output") + 1] != "-":
            if "apps" in args:
                payload = {
                    "result": {
                        "apps": [
                            {
                                "bundleIdentifier": "com.example.app",
                                "url": "file:///var/containers/X/Example.app/",
                            }
                        ]
                    }
                }
            else:
                payload = {
                    "result": {
                        "runningProcesses": [
                            {
                                "processIdentifier": 7,
                                "executable": "file:///usr/libexec/other",
                            },
                            {
                                "processIdentifier": 42,
                                "executable": "file:///var/containers/X/Example.app/Example",
                            },
                        ]
                    }
                }
            Path(args[args.index("--json-output") + 1]).write_text(json.dumps(payload))
        return b""

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.stop_app("com.example.app")
    assert calls[0][2:4] == ["info", "apps"]
    assert calls[1][2:4] == ["info", "processes"]
    terminate = calls[2]
    assert terminate[:5] == ["devicectl", "device", "process", "terminate", "--device"]
    assert terminate[terminate.index("--pid") + 1] == "42"


@pytest.mark.asyncio
async def test_stop_app_without_process_fails_clearly(connected_driver, monkeypatch):
    monkeypatch.setattr(
        physical_driver,
        "run_xcrun",
        _xcrun_payload_writer({"apps": {"result": {"apps": []}},
                               "processes": {"result": {"runningProcesses": []}}}),
    )
    with pytest.raises(ValueError, match="No running process"):
        await connected_driver.stop_app("com.example.app")


@pytest.mark.asyncio
async def test_install_accepts_signed_app_directories(connected_driver, monkeypatch, tmp_path):
    app = tmp_path / "Writer.app"
    app.mkdir()
    (app / "Info.plist").write_bytes(
        plistlib.dumps({"CFBundleIdentifier": "com.example.writer"})
    )
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return b"{}"

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.install_app(app) == "com.example.writer"
    assert calls[0][:4] == ("devicectl", "device", "install", "app")
    assert str(app) in calls[0]


@pytest.mark.asyncio
async def test_install_accepts_ipa_files(connected_driver, monkeypatch, tmp_path):
    ipa = tmp_path / "Writer.ipa"
    ipa.write_bytes(b"PK")
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return b"{}"

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.install_app(ipa) == "Writer"


@pytest.mark.asyncio
async def test_install_rejects_simulator_built_artifacts(connected_driver, tmp_path):
    with pytest.raises(ValueError, match="signed .app"):
        await connected_driver.install_app(tmp_path / "Writer.zip")


@pytest.mark.asyncio
async def test_open_url_uses_devicectl(connected_driver, monkeypatch):
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return b"{}"

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    assert await connected_driver.open_url("bookwriter://open")
    assert calls[0][:5] == ("devicectl", "device", "process", "openURL", "--device")
    assert calls[0][-1] == "bookwriter://open"


@pytest.mark.asyncio
async def test_list_apps_parses_devicectl_json(connected_driver, monkeypatch):
    payload = {
        "apps": [
            {"bundleIdentifier": "com.example.writer", "name": "Writer"},
            {"bundleIdentifier": "com.example.other"},
        ]
    }

    async def fake_xcrun(*arguments, timeout=30.0):
        args = list(arguments)
        if "--json-output" in args:
            Path(args[args.index("--json-output") + 1]).write_text(
                json.dumps({"result": payload})
            )
        return b""

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    apps = await connected_driver.list_apps()
    assert apps == {"com.example.writer": "Writer", "com.example.other": "com.example.other"}


@pytest.mark.asyncio
async def test_shell_commands_stay_rejected(connected_driver):
    with pytest.raises(NotImplementedError):
        await connected_driver.execute_shell("ls")


@pytest.mark.asyncio
async def test_recording_requires_connection():
    driver = PhysicalIosDriver(device_id=IPHONE_UDID)
    with pytest.raises(RuntimeError, match="Connect"):
        await driver.start_video_recording()


# --- Physical recorder ------------------------------------------------------


def _png(path: Path, width: int = 100, height: int = 200) -> None:
    from PIL import Image

    Image.new("RGB", (width, height), color="red").save(path)


@pytest.mark.asyncio
async def test_recorder_polls_frames_and_assembles_mp4(tmp_path, monkeypatch):
    """Frames become one timestamped segment through the ffconcat demuxer."""
    ffmpeg_calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        # Pace like real devicectl round-trips; an instant fake lets the poll
        # loop flood tmp_path with frames and stalls cleanup for minutes.
        await asyncio.sleep(0.02)
        destination = Path(arguments[arguments.index("--destination") + 1])
        _png(destination)
        return b""

    async def fake_ffmpeg(arguments):
        ffmpeg_calls.append(arguments)
        output = Path(arguments[-1])
        output.write_bytes(b"mp4")
        return 0, b""

    async def fake_probe(path):
        return {"duration": 1.0, "width": 100, "height": 200}

    async def fake_manifest(output_dir, paths, offsets):
        manifest = Path(output_dir) / "manifest.json"
        manifest.write_text("{}")
        return manifest

    monkeypatch.setattr(physical_recording, "run_xcrun", fake_xcrun)
    monkeypatch.setattr(physical_recording, "_run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(physical_recording, "probe_video_segment", fake_probe)
    monkeypatch.setattr(physical_recording, "write_recording_manifest", fake_manifest)
    recorder = PhysicalIosRecorder(IPHONE_UDID)
    session = await recorder.start(tmp_path)
    path = await recorder.stop()
    assert path is not None and path.suffix == ".mp4"
    assert any("frames.txt" in str(arg) for call in ffmpeg_calls for arg in call)


@pytest.mark.asyncio
async def test_recorder_partitions_frames_at_the_seam(tmp_path, monkeypatch):
    async def fake_xcrun(*arguments, timeout=30.0):
        await asyncio.sleep(0.02)
        destination = Path(arguments[arguments.index("--destination") + 1])
        _png(destination)
        return b""

    monkeypatch.setattr(physical_recording, "run_xcrun", fake_xcrun)
    recorder = PhysicalIosRecorder(IPHONE_UDID)
    session = await recorder.start(tmp_path)
    anchor = session.anchor_monotonic
    seam = anchor + 5.0
    session.frames = [
        {"path": tmp_path / "frames_0000" / "f1.png", "at": seam - 2},
        {"path": tmp_path / "frames_0000" / "f2.png", "at": seam - 1},
        {"path": tmp_path / "frames_0000" / "f3.png", "at": seam + 1},
    ]
    record = recorder._seal_current_segment(session, seam)
    session.poll_task.cancel()
    assert len(record["frames"]) == 2
    assert len(session.frames) == 1


@pytest.mark.asyncio
async def test_recorder_fails_closed_when_devicectl_never_delivers(tmp_path, monkeypatch):
    async def fake_xcrun(*arguments, timeout=30.0):
        await asyncio.sleep(0.01)
        raise RuntimeError("device disconnected")

    monkeypatch.setattr(physical_recording, "run_xcrun", fake_xcrun)
    monkeypatch.setattr(physical_recording, "MAX_CONSECUTIVE_FAILURES", 2)
    recorder = PhysicalIosRecorder(IPHONE_UDID)
    with pytest.raises(RuntimeError):
        await recorder.start(tmp_path)
    session = recorder.session
    assert session is not None and not session.is_active
    assert session.errors


# --- WebDriverAgent client and wiring --------------------------------------


def test_normalize_wda_url_accepts_hosts_ips_and_urls():
    assert wda.normalize_wda_url("192.168.0.5") == "http://192.168.0.5:8100"
    assert wda.normalize_wda_url("fd20::1") == "http://[fd20::1]:8100"
    assert wda.normalize_wda_url("http://10.0.0.2:9000/") == "http://10.0.0.2:9000"
    assert wda.normalize_wda_url("localhost:8100") == "http://localhost:8100"


def test_wda_url_candidates_order_env_host_tunnel_localhost(monkeypatch):
    monkeypatch.delenv(wda.WDA_URL_ENV, raising=False)
    monkeypatch.delenv(wda.WDA_HOST_ENV, raising=False)
    candidates = wda.wda_url_candidates(tunnel_ip="fd20:85a4::1")
    assert candidates == ["http://[fd20:85a4::1]:8100", "http://127.0.0.1:8100"]
    monkeypatch.setenv(wda.WDA_URL_ENV, "http://10.1.1.1:8100")
    monkeypatch.setenv(wda.WDA_HOST_ENV, "phone.lan")
    assert wda.wda_url_candidates(tunnel_ip="fd20:85a4::1") == [
        "http://10.1.1.1:8100",
        "http://phone.lan:8100",
        "http://[fd20:85a4::1]:8100",
        "http://127.0.0.1:8100",
    ]


def test_parse_wda_elements_maps_tree_to_ui_elements():
    tree = {
        "type": "XCUIElementTypeApplication",
        "rect": {"x": 0, "y": 0, "width": 100, "height": 200},
        "children": [
            {
                "type": "XCUIElementTypeButton",
                "name": "saveButton",
                "label": "Save",
                "rect": {"x": 10, "y": 20, "width": 30, "height": 10},
                "isVisible": True,
                "children": [],
            },
            {
                "type": "XCUIElementTypeStaticText",
                "label": "Title",
                "value": "hello",
                "rect": {"x": 0, "y": 0, "width": 50, "height": 10},
                "children": [],
            },
            {"type": "XCUIElementTypeOther", "rect": {"x": 0, "y": 0, "width": 0, "height": 0}},
        ],
    }
    elements = wda.parse_wda_elements(tree, (3.0, 3.0), 300, 600)
    assert len(elements) == 3  # root window + two children with geometry
    button = next(e for e in elements if e["resource_id"] == "saveButton")
    assert button["text"] == "Save"
    assert button["class"] == "Button"
    assert button["parsed_bounds"] == {"left": 30, "top": 60, "right": 120, "bottom": 90}
    assert button["hit_point"] == [75, 75]
    label = next(e for e in elements if e["text"] == "Title")
    assert label["value"] == "hello"


@pytest.mark.asyncio
async def test_wda_client_unwraps_values_and_raises_errors(monkeypatch):
    client = WdaClient("http://wda.test:8100")

    class _Response:
        def __init__(self, payload):
            self._raw = json.dumps(payload).encode()

        def read(self):
            return self._raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout):
        if request.full_url.endswith("/status"):
            return _Response({"value": {"ready": True}})
        if request.full_url.endswith("/session"):
            return _Response({"value": {"sessionId": "abc-123"}})
        return _Response({"value": {"error": "no such", "message": "nope"}})

    monkeypatch.setattr(wda.urllib.request, "urlopen", fake_urlopen)
    assert await client.status() == {"ready": True}
    assert await client.open_session() == "abc-123"
    with pytest.raises(RuntimeError, match="nope"):
        await client.window_size()


@pytest.mark.asyncio
async def test_ensure_wda_returns_probed_client(driver, monkeypatch):
    client = _FakeWda()
    monkeypatch.setattr(physical_driver, "wda_url_candidates", lambda **kw: ["http://a:8100"])
    monkeypatch.setattr(physical_driver, "probe_wda", _async_return(client))
    monkeypatch.setattr(driver, "_tunnel_ip", _async_return("fd20::1"))
    assert await driver._ensure_wda() is client


@pytest.mark.asyncio
async def test_ensure_wda_launches_runner_then_probes(driver, monkeypatch):
    client = _FakeWda()
    probes = []

    async def probe(candidates, timeout=5.0):
        probes.append(candidates)
        return client if len(probes) > 1 else None

    async def apps_payload(*arguments):
        return {"apps": [{"bundleIdentifier": "com.artemis.WebDriverAgentRunner.xctrunner"}]}

    monkeypatch.setattr(physical_driver, "wda_url_candidates", lambda **kw: ["http://a:8100"])
    monkeypatch.setattr(physical_driver, "probe_wda", probe)
    monkeypatch.setattr(driver, "_tunnel_ip", _async_return(None))
    monkeypatch.setattr(driver, "_devicectl_json", apps_payload)
    monkeypatch.setattr(driver, "_launch_bundle", _async_return(777))
    assert await driver._ensure_wda() is client
    assert driver._wda_runner_pid == 777


@pytest.mark.asyncio
async def test_ensure_wda_without_runner_or_server_fails(driver, monkeypatch):
    monkeypatch.setattr(physical_driver, "wda_url_candidates", lambda **kw: ["http://a:8100"])
    monkeypatch.setattr(physical_driver, "probe_wda", _async_return(None))
    monkeypatch.setattr(driver, "_tunnel_ip", _async_return(None))
    monkeypatch.setattr(driver, "_devicectl_json", _async_return({"apps": []}))
    with pytest.raises(RuntimeError, match="WebDriverAgent"):
        await driver._ensure_wda()


@pytest.mark.asyncio
async def test_capture_maps_wda_tree_and_screenshot(connected_driver):
    connected_driver._wda = _FakeWda(
        window=(100.0, 200.0),
        tree={
            "type": "XCUIElementTypeApplication",
            "rect": {"x": 0, "y": 0, "width": 100, "height": 200},
            "children": [
                {
                    "type": "XCUIElementTypeButton",
                    "label": "OK",
                    "rect": {"x": 10, "y": 10, "width": 20, "height": 20},
                }
            ],
        },
    )
    data = await connected_driver.get_screen_data()
    assert data.platform == "ios"
    assert (data.width, data.height) == (300, 600)
    assert connected_driver._scale == (3.0, 3.0)
    button = next(e for e in data.ui_elements if e["text"] == "OK")
    assert button["hit_point"] == [60, 60]


@pytest.mark.asyncio
async def test_tap_converts_pixels_to_wda_points(connected_driver):
    await connected_driver.tap(150, 300)
    assert connected_driver._wda.tapped == [(50.0, 100.0, 100)]


@pytest.mark.asyncio
async def test_swipe_converts_endpoints(connected_driver):
    await connected_driver.swipe(30, 300, 150, 60, duration_ms=500)
    assert connected_driver._wda.swiped == [(10.0, 100.0, 50.0, 20.0, 500)]


@pytest.mark.asyncio
async def test_input_text_and_keys_route_to_wda(connected_driver):
    await connected_driver.input_text("hi", clear_existing=False)
    assert connected_driver._wda.typed == ["hi"]
    await connected_driver.press_key("volume_up")
    assert connected_driver._wda.buttons == ["volumeUp"]
    await connected_driver.press_key("home")
    assert connected_driver._wda.buttons == ["volumeUp", "home"]
    with pytest.raises(NotImplementedError, match="typing appends"):
        await connected_driver.input_text("x", clear_existing=True)


@pytest.mark.asyncio
async def test_current_package_uses_wda_active_app(connected_driver):
    assert await connected_driver.get_current_package() == "com.example.foreground"


@pytest.mark.asyncio
async def test_disconnect_closes_wda_and_terminates_runner(connected_driver, monkeypatch):
    connected_driver._wda_runner_pid = 777
    calls = []

    async def fake_xcrun(*arguments, timeout=30.0):
        calls.append(arguments)
        return b""

    monkeypatch.setattr(physical_driver, "run_xcrun", fake_xcrun)
    await connected_driver.disconnect()
    assert connected_driver._wda is None
    assert connected_driver._session_key is None
    assert connected_driver._wda_runner_pid is None
    assert any(
        "terminate" in args and str(777) in args for args in calls
    )


@pytest.mark.asyncio
async def test_connect_establishes_wda_session(driver, monkeypatch):
    client = _FakeWda()
    monkeypatch.setattr(physical_driver, "list_core_devices", _async_devices(_parsed(PHYSICAL_IPHONE)))
    monkeypatch.setattr(driver, "_ensure_wda", _async_return(client))
    monkeypatch.setattr(driver, "_require_ios_host", _async_return(None))
    await driver.connect()
    assert driver._wda is client
    assert driver._session_key == "wda-session"


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


# --- Factory routing --------------------------------------------------------


def test_factory_routes_physical_udids_to_the_physical_driver(monkeypatch):
    import artemis.drivers.factory as factory

    ctx = SimpleNamespace(
        device=SimpleNamespace(
            mobile_platform="ios",
            device_id=IPHONE_UDID,
            device_width=1179,
            device_height=2556,
        ),
        agent_config=None,
    )
    monkeypatch.delenv("ARTEMIS_CLOUD_MODE", raising=False)
    monkeypatch.setattr(
        discovery, "list_ios_simulators_sync", lambda force_refresh=False: _parsed(SIMULATOR)
    )
    monkeypatch.setattr(
        discovery, "find_physical_ios_device_sync", lambda identifier: _parsed(PHYSICAL_IPHONE)[0]
    )
    driver = factory.create_driver(ctx)
    assert isinstance(driver, PhysicalIosDriver)
    assert driver.device_id == IPHONE_UDID


def test_factory_keeps_simulator_udids_on_the_simulator_driver(monkeypatch):
    import artemis.drivers.factory as factory
    from artemis.drivers.ios.xcode_driver import XcodeSimulatorDriver

    ctx = SimpleNamespace(
        device=SimpleNamespace(
            mobile_platform="ios",
            device_id=SIM_UDID,
            device_width=1179,
            device_height=2556,
        ),
        agent_config=None,
    )
    monkeypatch.delenv("ARTEMIS_CLOUD_MODE", raising=False)
    monkeypatch.setattr(
        discovery, "list_ios_simulators_sync", lambda force_refresh=False: _parsed(SIMULATOR)
    )
    driver = factory.create_driver(ctx)
    assert isinstance(driver, XcodeSimulatorDriver)
    assert not isinstance(driver, PhysicalIosDriver)


def test_factory_defaults_unknown_serials_to_simulator_validation(monkeypatch):
    """Unrecognized serials keep the simulator driver so its error path applies."""
    import artemis.drivers.factory as factory
    from artemis.drivers.ios.xcode_driver import XcodeSimulatorDriver

    ctx = SimpleNamespace(
        device=SimpleNamespace(
            mobile_platform="ios",
            device_id="unknown-serial",
            device_width=1179,
            device_height=2556,
        ),
        agent_config=None,
    )
    monkeypatch.delenv("ARTEMIS_CLOUD_MODE", raising=False)
    monkeypatch.setattr(discovery, "list_ios_simulators_sync", lambda force_refresh=False: [])
    monkeypatch.setattr(discovery, "find_physical_ios_device_sync", lambda identifier: None)
    driver = factory.create_driver(ctx)
    assert type(driver) is XcodeSimulatorDriver


def test_factory_cloud_mode_still_rejects_ios(monkeypatch):
    import artemis.drivers.factory as factory

    ctx = SimpleNamespace(
        device=SimpleNamespace(mobile_platform="ios", device_id=IPHONE_UDID),
        agent_config=None,
    )
    monkeypatch.setenv("ARTEMIS_CLOUD_MODE", "1")
    with pytest.raises(ValueError, match="local only"):
        factory.create_driver(ctx)


def test_android_selection_is_unchanged(monkeypatch):
    import artemis.drivers.factory as factory
    from artemis.drivers.android.adb_driver import AndroidAdbDriver

    ctx = SimpleNamespace(
        device=SimpleNamespace(
            mobile_platform="android",
            device_id="emulator-5554",
            device_width=1080,
            device_height=2400,
        ),
        agent_config=None,
        adb_client=None,
        ui_adb_client=None,
    )
    monkeypatch.delenv("ARTEMIS_CLOUD_MODE", raising=False)
    monkeypatch.delenv("ARTEMIS_MOCK_DRIVER", raising=False)
    driver = factory.create_driver(ctx)
    assert isinstance(driver, AndroidAdbDriver)
