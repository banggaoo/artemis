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

"""Hybrid Flash loop (decision.flash_loop): VLM describes, decision model acts.

The describe call binds no tools, so malformed tool-call envelopes cannot be
emitted; the decision model picks from a fixed menu (element taps +
meta-actions). Any inconclusive verdict — unreachable endpoint, low
confidence, or a key outside the menu — escalates to the standard
tool-calling model turn.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from artemis.agents.flash.runner import FlashRunner, _HYBRID_META_ACTIONS
from artemis.core.decision_client import DecisionResult
from artemis.llm.reliability import Failure, FailureCategory, LLMCallError


_FAILURE = Failure(FailureCategory.UNKNOWN, retryable=True, should_fallback=True)


def _runner(goal: str = "enable dark theme") -> FlashRunner:
    r = FlashRunner.__new__(FlashRunner)
    r.goal = goal
    r._last_action_note = None
    return r


def _cfg(threshold: float = 0.85):
    return SimpleNamespace(
        url="http://127.0.0.1:11434/v1/decide",
        model="laya",
        timeout_seconds=5.0,
        confidence_threshold=threshold,
    )


def _state(elements):
    return SimpleNamespace(indexed_elements=elements)


def _verdict(verdict, confidence=0.9):
    return DecisionResult(verdict=verdict, confidence=confidence)


# --- action menu --------------------------------------------------------------


def test_menu_lists_element_taps_and_meta_actions():
    state = _state(
        [
            {"index": 1, "text": "Settings"},
            {"index": 2, "content_desc": "Search"},
            {"index": 3, "resource_id": "id/more"},
        ]
    )
    menu = FlashRunner._hybrid_action_menu(state)
    assert menu["tap_1"] == "tap element 1: Settings"
    assert menu["tap_2"] == "tap element 2: Search"
    assert menu["tap_3"] == "tap element 3: id/more"
    for key in _HYBRID_META_ACTIONS:
        assert key in menu


def test_menu_handles_missing_elements():
    assert "done" in FlashRunner._hybrid_action_menu(_state(None))
    assert "tap_" not in " ".join(FlashRunner._hybrid_action_menu(_state(None)))


# --- menu -> tool call mapping -------------------------------------------------


def test_choice_maps_element_tap_to_click():
    call = FlashRunner._menu_choice_to_call("tap_4", "desc")
    assert call["name"] == "click"
    assert call["args"] == {"target": 4}
    assert call["id"]


def test_choice_maps_swipe_and_keys():
    assert FlashRunner._menu_choice_to_call("swipe_up", "")["args"] == {
        "direction": "up"
    }
    back = FlashRunner._menu_choice_to_call("press_back", "")
    assert back["name"] == "press_key" and back["args"] == {"key": "back"}


def test_choice_maps_done_and_fail_to_report():
    done = FlashRunner._menu_choice_to_call("done", "all set")
    assert done["name"] == "report_task_status"
    assert done["args"]["status"] == "completed"
    assert done["args"]["explanation"] == "all set"
    fail = FlashRunner._menu_choice_to_call("fail", "blocked")
    assert fail["args"]["status"] == "failed"


def test_choice_rejects_unknown_key():
    assert FlashRunner._menu_choice_to_call("launch_rocket", "") is None


# --- executed call -> menu key mapping ------------------------------------------


def test_menu_key_for_call_maps_device_actions():
    f = FlashRunner._menu_key_for_call
    assert f("click", {"target": 7, "target_description": "x"}) == "tap_7"
    assert f("swipe", {"direction": "up"}) == "swipe_up"
    assert f("swipe", {"gesture": "left"}) == "swipe_left"
    assert f("press_key", {"key": "BACK"}) == "press_back"
    assert f("press_key", {"key": "home"}) == "press_home"
    assert f("wait_for_delay", {"time_in_ms": 500}) == "wait"
    assert f("report_task_status", {"status": "completed"}) == "done"
    assert f("report_task_status", {"status": "failed"}) == "fail"


def test_menu_key_for_call_rejects_non_menu_calls():
    f = FlashRunner._menu_key_for_call
    assert f("click", {"target": "not-an-int"}) is None
    assert f("swipe", {"start": [0, 0], "end": [1, 1]}) is None
    assert f("input_text", {"text": "hi"}) is None


# --- _hybrid_decide ------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_decide_returns_call_on_confident_verdict():
    runner = _runner()
    state = _state([{"index": 1, "text": "Dark theme"}])
    describe_resp = AIMessage(content="Settings list; [1] Display ~[500,300]")

    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=describe_resp),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = AsyncMock(
            return_value=_verdict("tap_1")
        )
        out = await runner._hybrid_decide(
            AsyncMock(), state, HumanMessage(content="obs"), "plan", _cfg()
        )

    assert out is not None
    response, tool_calls = out
    assert tool_calls[0]["name"] == "click"
    assert tool_calls[0]["args"] == {"target": 1}
    # The choice is stamped onto the describe message so ToolMessages keep a
    # legitimate parent when the ledger replays them.
    assert response.tool_calls[0]["name"] == "click"
    # The describe response is pure text — no envelope parsing needed.
    assert "Settings list" in response.content


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_on_low_confidence():
    runner = _runner()
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = AsyncMock(
            return_value=_verdict("tap_1", confidence=0.10)
        )
        out = await runner._hybrid_decide(
            AsyncMock(), _state([{"index": 1, "text": "Dark theme"}]), HumanMessage(content="x"), None, _cfg()
        )
    assert out is None


@pytest.mark.asyncio
async def test_hybrid_decide_gates_on_answer_confidence_not_trust_score():
    """laya-serve returns near-zero `confidence` on menu-shaped spreads;
    the actionable signal is `answer_confidence` (top-pick probability)."""
    runner = _runner()
    verdict = DecisionResult(
        verdict="tap_1",
        confidence=0.008,  # calibrated trust score — near zero
        raw={"answer_confidence": 0.62},
    )
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = AsyncMock(return_value=verdict)
        out = await runner._hybrid_decide(
            AsyncMock(), _state([{"index": 1, "text": "Dark theme"}]), HumanMessage(content="x"), None, _cfg()
        )
    assert out is not None
    assert out[1][0]["name"] == "click"


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_when_unanswered():
    runner = _runner()
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = AsyncMock(
            return_value=DecisionResult(verdict=None)
        )
        out = await runner._hybrid_decide(
            AsyncMock(), _state([]), HumanMessage(content="x"), None, _cfg()
        )
    assert out is None


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_on_out_of_menu_verdict():
    runner = _runner()
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = AsyncMock(
            return_value=_verdict("tap_99", confidence=0.95)
        )
        out = await runner._hybrid_decide(
            AsyncMock(), _state([]), HumanMessage(content="x"), None, _cfg()
        )
    assert out is None


@pytest.mark.asyncio
async def test_hybrid_decide_masks_tried_menu_keys():
    """Elements already tapped are removed from the menu by (index, label)
    identity, so the decision model cannot re-pick a dead element."""
    runner = _runner(goal="tap Alpha Beta")
    state = _state([{"index": 1, "text": "Alpha"}, {"index": 2, "text": "Beta"}])
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        choose = AsyncMock(return_value=_verdict("tap_2"))
        MockClient.return_value.achoose = choose
        out = await runner._hybrid_decide(
            AsyncMock(),
            state,
            HumanMessage(content="x"),
            None,
            _cfg(),
            tried_taps={(1, "Alpha")},
            tried_labels=set(),
        )
    assert out is not None
    menu_sent = choose.call_args.args[2]
    assert "tap_1" not in menu_sent
    assert "tap_2" in menu_sent
    # Terminal meta-actions are never masked — wait/done/fail stay available.
    assert "wait" in menu_sent
    assert "done" in menu_sent
    # The decision state discloses what was masked so the pick has context.
    state_sent = choose.call_args.args[0]
    assert "tap_1" in state_sent


@pytest.mark.asyncio
async def test_hybrid_decide_masks_reindexed_label():
    """A distinctive label stays masked even when the element list re-indexes
    (the volatile-text problem that defeats screen signatures)."""
    runner = _runner(goal="open Settings Beta")
    # Same "Settings" row, now at index 9 instead of 12.
    state = _state([{"index": 9, "text": "Settings"}, {"index": 2, "text": "Beta"}])
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        choose = AsyncMock(return_value=_verdict("tap_2"))
        MockClient.return_value.achoose = choose
        out = await runner._hybrid_decide(
            AsyncMock(),
            state,
            HumanMessage(content="x"),
            None,
            _cfg(),
            tried_taps={(12, "Settings")},
            tried_labels={"Settings"},
        )
    assert out is not None
    menu_sent = choose.call_args.args[2]
    assert "tap_9" not in menu_sent
    assert "tap_2" in menu_sent


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_when_menu_lacks_goal_target():
    """No element label sharing a goal token → between-screen navigation is
    the VLM path's job; the decision model only picks on-screen elements."""
    runner = _runner()  # goal: "enable dark theme"
    choose = AsyncMock()
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="Gmail setup page")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        MockClient.return_value.achoose = choose
        out = await runner._hybrid_decide(
            AsyncMock(),
            _state([{"index": 1, "text": "GOT IT"}, {"index": 2, "text": "Gmail"}]),
            HumanMessage(content="x"),
            None,
            _cfg(),
        )
    assert out is None
    choose.assert_not_awaited()


@pytest.mark.asyncio
async def test_hybrid_decide_proceeds_when_goal_target_present():
    runner = _runner()
    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="Display screen")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        choose = AsyncMock(return_value=_verdict("tap_2"))
        MockClient.return_value.achoose = choose
        out = await runner._hybrid_decide(
            AsyncMock(),
            _state(
                [{"index": 1, "text": "Brightness"}, {"index": 2, "text": "Dark theme"}]
            ),
            HumanMessage(content="x"),
            None,
            _cfg(),
        )
    assert out is not None
    choose.assert_awaited()


def test_menu_has_goal_target_matching():
    f = FlashRunner._menu_has_goal_target
    menu = {"tap_1": "tap element 1: Gmail", "tap_2": "tap element 2: Dark theme"}
    assert f(menu, "enable dark theme") is True
    assert f(menu, "open the Settings app") is False
    assert f({"swipe_up": "scroll"}, "enable dark theme") is False
    # goals with no matchable tokens never gate (all short/stopword tokens)
    assert f(menu, "do it") is True


@pytest.mark.asyncio
async def test_hybrid_decide_never_executes_terminal_picks():
    """done/fail escalate to the VLM turn — the decision model sees no pixels
    and cannot honestly verify goal completion (observed live declaring
    'done' on a home screen where the target app wasn't open)."""
    runner = _runner()
    for pick in ("done", "fail"):
        with (
            patch(
                "artemis.agents.flash.runner.acomplete",
                new=AsyncMock(return_value=AIMessage(content="desc")),
            ),
            patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
        ):
            MockClient.return_value.achoose = AsyncMock(
                return_value=_verdict(pick, confidence=0.95)
            )
            out = await runner._hybrid_decide(
                AsyncMock(), _state([{"index": 1, "text": "Dark theme"}]), HumanMessage(content="x"),
                None, _cfg(),
            )
        assert out is None, pick


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_on_describe_failure():
    runner = _runner()
    with patch(
        "artemis.agents.flash.runner.acomplete",
        new=AsyncMock(side_effect=LLMCallError("server down", failure=_FAILURE)),
    ):
        out = await runner._hybrid_decide(
            AsyncMock(), _state([]), HumanMessage(content="x"), None, _cfg()
        )
    assert out is None


# --- _hybrid_plan --------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_plan_returns_milestone_text():
    runner = _runner()
    with patch(
        "artemis.agents.flash.runner.acomplete",
        new=AsyncMock(return_value=AIMessage(content="- open Settings\n- tap Display")),
    ):
        plan = await runner._hybrid_plan(AsyncMock())
    assert "open Settings" in plan


@pytest.mark.asyncio
async def test_hybrid_plan_failure_is_non_fatal():
    runner = _runner()
    with patch(
        "artemis.agents.flash.runner.acomplete",
        new=AsyncMock(side_effect=LLMCallError("boom", failure=_FAILURE)),
    ):
        assert await runner._hybrid_plan(AsyncMock()) is None


@pytest.mark.asyncio
async def test_hybrid_decide_state_includes_last_action_outcome():
    """The decision model has no transcript view — the one-line last-action
    note is its only signal that a tap was a dead end vs. real progress."""
    runner = _runner()
    runner._last_action_note = "click → no visible change"
    state = _state([{"index": 1, "text": "Dark theme"}])
    captured = {}

    with (
        patch(
            "artemis.agents.flash.runner.acomplete",
            new=AsyncMock(return_value=AIMessage(content="desc")),
        ),
        patch("artemis.agents.flash.runner.DecisionClient") as MockClient,
    ):
        async def _achoose(state_text, question, menu):
            captured["state"] = state_text
            return _verdict("tap_1")

        MockClient.return_value.achoose = _achoose
        await runner._hybrid_decide(
            AsyncMock(), state, HumanMessage(content="x"), "plan", _cfg()
        )

    assert "click → no visible change" in captured["state"]


@pytest.mark.asyncio
async def test_hybrid_decide_escalates_when_describe_says_target_reached():
    """'TARGET: reached' is a perception claim, not a menu pick — the turn
    escalates so the VLM can verify and report (the terminal gate audits it)."""
    runner = _runner()
    state = _state([{"index": 1, "text": "Dark theme"}])
    describe = AIMessage(content="Dark theme is already on.\nTARGET: reached")

    with patch(
        "artemis.agents.flash.runner.acomplete",
        new=AsyncMock(return_value=describe),
    ), patch("artemis.agents.flash.runner.DecisionClient") as MockClient:
        out = await runner._hybrid_decide(
            AsyncMock(), state, HumanMessage(content="x"), "plan", _cfg()
        )

    assert out is None
    MockClient.return_value.achoose.assert_not_called()


# --- hybrid prompt rendering ---------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_plan_prompt_renders_goal():
    """The plan prompt is rendered with jinja2, so `{{ goal }}` must be
    substituted — a `$goal` placeholder would reach the VLM literally."""
    runner = _runner(goal="enable dark theme")
    complete = AsyncMock(return_value=AIMessage(content="- open Settings"))
    with patch("artemis.agents.flash.runner.acomplete", new=complete):
        await runner._hybrid_plan(AsyncMock())

    prompt = complete.call_args.args[1][0].content[0]["text"]
    assert "GOAL: enable dark theme" in prompt
    assert "$goal" not in prompt


@pytest.mark.asyncio
async def test_hybrid_describe_prompt_renders_goal_and_plan():
    """The describe prompt must carry both the goal and the plan text —
    rendered with jinja2, `$goal`/`$plan` would pass through literally."""
    runner = _runner(goal="enable dark theme")
    describe = AIMessage(content="Settings list.\nTARGET: reached")
    complete = AsyncMock(return_value=describe)
    with patch("artemis.agents.flash.runner.acomplete", new=complete):
        await runner._hybrid_decide(
            AsyncMock(),
            _state([{"index": 1, "text": "Dark theme"}]),
            HumanMessage(content="x"),
            "- open Settings\n- tap Display",
            _cfg(),
        )

    prompt = complete.call_args.args[1][0].content
    assert "GOAL: enable dark theme" in prompt
    assert "- open Settings\n- tap Display" in prompt
    assert "$goal" not in prompt
    assert "$plan" not in prompt
