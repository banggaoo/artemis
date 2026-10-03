# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MCP bridge ownership and failure behavior without launching Xcode."""

import asyncio
from contextlib import asynccontextmanager
import json
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent
import pytest

from artemis.drivers.ios import bridge
from artemis.drivers.ios.bridge import XcodeBridge


def response(data=None, *, text=None, error=False):
    return CallToolResult(
        content=[TextContent(type="text", text=text)] if text is not None else [],
        structuredContent=data,
        isError=error,
    )


class NativeSession:
    def __init__(self):
        self.events = []
        self.tools_calls = []
        self.tool_calls = []
        self.initialize_hook = None
        self.call_hook = None
        self.pages = {
            None: (["DeviceInteractionStartSession"], "second"),
            "second": (["DeviceInteractionSynthesize", "DeviceInteractionEndSession"], None),
        }

    async def __aenter__(self):
        self.events.append(("session-enter", asyncio.current_task()))
        return self

    async def __aexit__(self, *args):
        self.events.append(("session-exit", asyncio.current_task()))

    async def initialize(self):
        self.events.append(("initialize", asyncio.current_task()))
        if self.initialize_hook:
            await self.initialize_hook()

    async def list_tools(self, cursor=None):
        self.tools_calls.append(cursor)
        names, next_cursor = self.pages[cursor]
        return SimpleNamespace(
            tools=[SimpleNamespace(name=name) for name in names], nextCursor=next_cursor
        )

    async def call_tool(self, name, arguments):
        self.tool_calls.append((name, arguments))
        if self.call_hook:
            return await self.call_hook(name, arguments)
        return response({"userMessage": "OK"})


@pytest.fixture
def native_transport(monkeypatch):
    session = NativeSession()
    events = []
    parameters = []

    @asynccontextmanager
    async def stdio(params):
        parameters.append(params)
        events.append(("stdio-enter", asyncio.current_task()))
        try:
            yield object(), object()
        finally:
            events.append(("stdio-exit", asyncio.current_task()))

    monkeypatch.setattr(bridge, "stdio_client", stdio)
    monkeypatch.setattr(bridge, "ClientSession", lambda reader, writer: session)
    return SimpleNamespace(session=session, events=events, parameters=parameters)


def assert_resources_closed_in_owner(native):
    assert [event for event, task in native.events] == ["stdio-enter", "stdio-exit"]
    assert [event for event, task in native.session.events if event.startswith("session-")] == [
        "session-enter",
        "session-exit",
    ]
    owners = {task for event, task in native.events + native.session.events}
    assert len(owners) == 1


@pytest.mark.asyncio
async def test_initializes_discovers_paginated_tools_and_preserves_xcode_selection(
    native_transport, monkeypatch
):
    monkeypatch.setenv("DEVELOPER_DIR", "/Applications/Selected Xcode.app/Contents/Developer")
    monkeypatch.setenv("MCP_XCODE_PID", "1234")
    client = XcodeBridge()
    await client.start()
    assert client.connected
    assert client.tools == {
        "DeviceInteractionStartSession",
        "DeviceInteractionSynthesize",
        "DeviceInteractionEndSession",
    }
    assert native_transport.session.tools_calls == [None, "second"]
    params = native_transport.parameters[0]
    assert params.command == "xcrun" and params.args == ["mcpbridge"]
    assert params.env["DEVELOPER_DIR"] == "/Applications/Selected Xcode.app/Contents/Developer"
    assert params.env["MCP_XCODE_PID"] == "1234"
    assert await client.call("DeviceInteractionEndSession", {"interactionSessionKey": "key"}) == {
        "userMessage": "OK"
    }
    await client.close()
    assert not client.connected
    assert_resources_closed_in_owner(native_transport)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "native_response",
    [
        response({"screenshotPath": "/tmp/native.png"}),
        response(text=json.dumps({"screenshotPath": "/tmp/native.png"})),
    ],
)
async def test_reads_structured_and_legacy_json_results(native_transport, native_response):
    async def call(name, arguments):
        return native_response

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    assert await client.call("DeviceInteractionSynthesize", {}) == {
        "screenshotPath": "/tmp/native.png"
    }
    await client.close()


@pytest.mark.asyncio
async def test_native_tool_errors_propagate_without_retrying_or_destroying_connection(
    native_transport,
):
    results = iter(
        [response(text="Agent is not approved", error=True), response({"userMessage": "Closed"})]
    )

    async def call(name, arguments):
        return next(results)

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    with pytest.raises(
        RuntimeError, match="DeviceInteractionStartSession failed: Agent is not approved"
    ):
        await client.call("DeviceInteractionStartSession", {})
    assert await client.call("DeviceInteractionEndSession", {}) == {"userMessage": "Closed"}
    assert [name for name, arguments in native_transport.session.tool_calls] == [
        "DeviceInteractionStartSession",
        "DeviceInteractionEndSession",
    ]
    await client.close()


@pytest.mark.asyncio
async def test_unstructured_success_is_rejected(native_transport):
    async def call(name, arguments):
        return response(text="not JSON")

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    with pytest.raises(RuntimeError, match="no structured result"):
        await client.call("DeviceInteractionSynthesize", {})
    await client.close()


@pytest.mark.asyncio
async def test_simulator_calls_are_serialized(native_transport):
    first_started, release_first = asyncio.Event(), asyncio.Event()

    async def call(name, arguments):
        if name == "first":
            first_started.set()
            await release_first.wait()
        return response({"name": name})

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    first = asyncio.create_task(client.call("first", {}))
    await first_started.wait()
    second = asyncio.create_task(client.call("second", {}))
    await asyncio.sleep(0)
    assert [name for name, arguments in native_transport.session.tool_calls] == ["first"]
    release_first.set()
    assert await asyncio.gather(first, second) == [{"name": "first"}, {"name": "second"}]
    await client.close()


@pytest.mark.asyncio
async def test_initialization_timeout_releases_all_resources(native_transport):
    async def initialize():
        await asyncio.Event().wait()

    native_transport.session.initialize_hook = initialize
    client = XcodeBridge(timeout_seconds=0.01)
    with pytest.raises(TimeoutError):
        await client.start()
    assert_resources_closed_in_owner(native_transport)
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call("DeviceInteractionSynthesize", {})


@pytest.mark.asyncio
async def test_initialization_cancellation_releases_all_resources(native_transport):
    started = asyncio.Event()

    async def initialize():
        started.set()
        await asyncio.Event().wait()

    native_transport.session.initialize_hook = initialize
    client = XcodeBridge()
    task = asyncio.create_task(client.start())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert_resources_closed_in_owner(native_transport)


@pytest.mark.asyncio
async def test_initialization_failure_releases_all_resources(native_transport):
    async def initialize():
        raise OSError("Cannot open native service")

    native_transport.session.initialize_hook = initialize
    client = XcodeBridge()
    with pytest.raises(
        RuntimeError, match="Cannot connect to Xcode MCP: Cannot open native service"
    ):
        await client.start()
    assert not client.connected
    assert_resources_closed_in_owner(native_transport)


@pytest.mark.asyncio
async def test_input_timeout_retires_bridge_without_repeating_possibly_executed_action(
    native_transport,
):
    async def call(name, arguments):
        await asyncio.Event().wait()

    native_transport.session.call_hook = call
    client = XcodeBridge(timeout_seconds=0.01)
    await client.start()
    with pytest.raises(TimeoutError):
        await client.call("DeviceInteractionSynthesize", {"interactionCommand": "t 10 20"})
    assert len(native_transport.session.tool_calls) == 1
    assert_resources_closed_in_owner(native_transport)
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call("DeviceInteractionSynthesize", {})


@pytest.mark.asyncio
async def test_cancelled_input_retires_bridge_and_fails_queued_action(native_transport):
    started = asyncio.Event()

    async def call(name, arguments):
        started.set()
        await asyncio.Event().wait()

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    first = asyncio.create_task(client.call("first", {}))
    await started.wait()
    second = asyncio.create_task(client.call("second", {}))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    with pytest.raises(RuntimeError, match="bridge has closed"):
        await second
    assert [name for name, arguments in native_transport.session.tool_calls] == ["first"]
    assert_resources_closed_in_owner(native_transport)


@pytest.mark.asyncio
async def test_transport_failure_retires_bridge_and_fails_queued_action(native_transport):
    started, fail = asyncio.Event(), asyncio.Event()

    async def call(name, arguments):
        started.set()
        await fail.wait()
        raise OSError("pipe disconnected")

    native_transport.session.call_hook = call
    client = XcodeBridge()
    await client.start()
    first = asyncio.create_task(client.call("first", {}))
    await started.wait()
    second = asyncio.create_task(client.call("second", {}))
    await asyncio.sleep(0)
    fail.set()
    with pytest.raises(RuntimeError, match="connection failed: pipe disconnected"):
        await first
    with pytest.raises(RuntimeError, match="bridge has closed"):
        await second
    await client.close()
    assert_resources_closed_in_owner(native_transport)


@pytest.mark.asyncio
async def test_bridge_can_reconnect_after_timeout_for_session_cleanup(native_transport):
    async def call(name, arguments):
        await asyncio.Event().wait()

    native_transport.session.call_hook = call
    client = XcodeBridge(timeout_seconds=0.01)
    await client.start()
    with pytest.raises(TimeoutError):
        await client.call("DeviceInteractionSynthesize", {"interactionCommand": "t 10 20"})
    assert not client.connected
    native_transport.session.call_hook = None
    await client.start()
    assert client.connected
    assert await client.call(
        "DeviceInteractionEndSession", {"interactionSessionKey": "known-key"}
    ) == {"userMessage": "OK"}
    assert [name for name, arguments in native_transport.session.tool_calls] == [
        "DeviceInteractionSynthesize",
        "DeviceInteractionEndSession",
    ]
    await client.close()
    assert [event for event, task in native_transport.events] == [
        "stdio-enter",
        "stdio-exit",
        "stdio-enter",
        "stdio-exit",
    ]
