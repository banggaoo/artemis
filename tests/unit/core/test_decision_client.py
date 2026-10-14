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

"""Unit tests for the System-One decision client."""

import io
import json
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error

import pytest

from artemis.core.decision_client import DecisionClient


def _mock_response(body: dict):
    return io.BytesIO(json.dumps(body).encode("utf-8"))


class _Closing(io.BytesIO):
    """BytesIO that survives use as a urlopen context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _mock_ctx(body: dict):
    return _Closing(json.dumps(body).encode("utf-8"))


def test_noul_yes_verdict():
    client = DecisionClient(url="http://x/v1/decide", model="laya")
    body = {"answers": {"q": {"noul": 0.92, "confidence": 0.9}}}
    with patch("urllib.request.urlopen", return_value=_mock_ctx(body)) as mock:
        result = client.noul("state text", "is it ok?")
    assert result.answered
    assert result.verdict is True
    assert result.confidence == 0.9
    assert result.probabilities["yes"] == pytest.approx(0.92)
    payload = json.loads(mock.call_args.args[0].data.decode())
    assert payload["questions"]["q"]["type"] == "noul"
    assert payload["model"] == "laya"


def test_noul_no_verdict():
    client = DecisionClient(url="http://x/v1/decide")
    body = {"answers": {"q": {"noul": 0.1, "confidence": 0.88}}}
    with patch("urllib.request.urlopen", return_value=_mock_ctx(body)):
        result = client.noul("s", "q?")
    assert result.verdict is False
    assert result.probabilities["no"] == pytest.approx(0.9)


def test_choice_verdict():
    client = DecisionClient(url="http://x/v1/decide")
    body = {
        "answers": {
            "q": {
                "choice": "settings",
                "probabilities": {"settings": 0.8, "chrome": 0.2},
                "confidence": 0.8,
            }
        }
    }
    with patch("urllib.request.urlopen", return_value=_mock_ctx(body)):
        result = client.choose("s", "pick", {"settings": "a", "chrome": "b"})
    assert result.verdict == "settings"
    assert result.probabilities["settings"] == 0.8


def test_network_error_fails_closed():
    client = DecisionClient(url="http://x/v1/decide")
    with patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("down")
    ):
        result = client.noul("s", "q?")
    assert not result.answered
    assert result.verdict is None
    assert result.confidence == 0.0


def test_malformed_body_fails_closed():
    client = DecisionClient(url="http://x/v1/decide")
    with patch("urllib.request.urlopen", return_value=_Closing(b"not json")):
        result = client.noul("s", "q?")
    assert not result.answered


def test_missing_answer_fields_fail_closed():
    client = DecisionClient(url="http://x/v1/decide")
    body = {"answers": {"q": {}}}
    with patch("urllib.request.urlopen", return_value=_mock_ctx(body)):
        result = client.noul("s", "q?")
    assert result.verdict is None


@pytest.mark.asyncio
async def test_anoul_async_wrapper():
    client = DecisionClient(url="http://x/v1/decide")
    body = {"answers": {"q": {"noul": 0.7, "confidence": 0.95}}}
    with patch("urllib.request.urlopen", return_value=_mock_ctx(body)):
        result = await client.anoul("s", "q?")
    assert result.verdict is True
    assert result.confidence == 0.95


# --- Planner wiring (_try_decision_layer) -------------------------------------


@pytest.mark.asyncio
async def test_planner_decision_layer_disabled_returns_none():
    from artemis.agents.planner.planner import _try_decision_layer

    ctx = SimpleNamespace(agent_config=SimpleNamespace(
        decision=SimpleNamespace(enabled=False)))
    assert await _try_decision_layer(ctx, "goal", "before", "after") is None


@pytest.mark.asyncio
async def test_planner_decision_layer_answers_above_threshold():
    from artemis.agents.planner.planner import _try_decision_layer
    from artemis.core.decision_client import DecisionResult

    ctx = SimpleNamespace(agent_config=SimpleNamespace(
        decision=SimpleNamespace(
            enabled=True, planner_validation=True,
            url="http://x", model="laya",
            confidence_threshold=0.85, timeout_seconds=5.0)))
    with patch(
        "artemis.core.decision_client.DecisionClient.anoul",
        return_value=DecisionResult(verdict=True, confidence=0.9),
    ):
        out = await _try_decision_layer(ctx, "goal", "before", "after")
    assert out["status"] == "success"
    assert "approved" in out["feedback"]


@pytest.mark.asyncio
async def test_planner_decision_layer_escalates_below_threshold():
    from artemis.agents.planner.planner import _try_decision_layer
    from artemis.core.decision_client import DecisionResult

    ctx = SimpleNamespace(agent_config=SimpleNamespace(
        decision=SimpleNamespace(
            enabled=True, planner_validation=True,
            url="http://x", model="laya",
            confidence_threshold=0.85, timeout_seconds=5.0)))
    with patch(
        "artemis.core.decision_client.DecisionClient.anoul",
        return_value=DecisionResult(verdict=True, confidence=0.4),
    ):
        assert await _try_decision_layer(ctx, "goal", "before", "after") is None


@pytest.mark.asyncio
async def test_planner_decision_layer_escalates_on_failure():
    from artemis.agents.planner.planner import _try_decision_layer
    from artemis.core.decision_client import DecisionResult

    ctx = SimpleNamespace(agent_config=SimpleNamespace(
        decision=SimpleNamespace(
            enabled=True, planner_validation=True,
            url="http://x", model="laya",
            confidence_threshold=0.85, timeout_seconds=5.0)))
    with patch(
        "artemis.core.decision_client.DecisionClient.anoul",
        return_value=DecisionResult(verdict=None, confidence=0.0),
    ):
        assert await _try_decision_layer(ctx, "goal", "before", "after") is None
