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

"""Deterministic goal-invariant predicates (artemis.agents.flash.state_predicates).

Verdicts are tri-state: ``unknown`` covers absence, ambiguity and missing
state — never silently treated as failure or success.
"""

from types import SimpleNamespace

import pytest

from artemis.agents.flash.runner import FlashRunner
from artemis.agents.flash.state_predicates import (
    describe_failures,
    evaluate_invariants,
    evaluate_predicate,
)
from artemis.mcp.observation import _control_only_elements

_SWITCH = {
    "index": 14,
    "text": "Dark theme",
    "class": "android.widget.Switch",
    "resource_id": "com.android.settings:id/switchWidget",
    "bounds": [834, 692, 961, 744],
    "center": [897, 718],
    "checked": "true",
    "enabled": "true",
    "is_ocr": False,
}

_PRED_OFF = {
    "id": "dark_theme_off",
    "selector": {
        "resource_id": "com.android.settings:id/switchWidget",
        "class": "android.widget.Switch",
        "text": "Dark theme",
    },
    "attribute": "checked",
    "expected": False,
}


# --- single predicate ----------------------------------------------------------


def test_predicate_unsatisfied_when_state_differs():
    r = evaluate_predicate(_PRED_OFF, [_SWITCH])
    assert r["status"] == "unsatisfied"
    assert r["evidence"]["observed"] is True


def test_predicate_satisfied_when_state_matches():
    el = {**_SWITCH, "checked": "false"}
    r = evaluate_predicate(_PRED_OFF, [el])
    assert r["status"] == "satisfied"


def test_predicate_unknown_when_target_absent():
    r = evaluate_predicate(_PRED_OFF, [{"text": "unrelated"}])
    assert r["status"] == "unknown"
    assert r["reason"] == "target_absent"


def test_predicate_unknown_when_ambiguous():
    loose = {**_PRED_OFF, "selector": {"text": "Dark theme"}}
    r = evaluate_predicate(loose, [_SWITCH, dict(_SWITCH)])
    assert r["status"] == "unknown"
    assert r["reason"] == "ambiguous_target"


def test_predicate_unknown_when_attribute_missing():
    el = {k: v for k, v in _SWITCH.items() if k != "checked"}
    r = evaluate_predicate(_PRED_OFF, [el])
    assert r["status"] == "unknown"
    assert r["reason"] == "attribute_not_known"


def test_predicate_ignores_ocr_elements():
    ocr = {**_SWITCH, "is_ocr": True, "checked": "false"}
    r = evaluate_predicate(_PRED_OFF, [ocr])
    assert r["status"] == "unknown"
    assert r["reason"] == "target_absent"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"selector": {}},
        {**_PRED_OFF, "selector": {"bounds": "[0,0][1,1]"}},
        {**_PRED_OFF, "attribute": "content-desc"},
        {**_PRED_OFF, "expected": "off"},
        "not-a-dict",
    ],
)
def test_predicate_unknown_when_spec_unsupported(bad):
    assert evaluate_predicate(bad, [_SWITCH])["status"] == "unknown"


# --- invariant aggregation -----------------------------------------------------


def test_invariants_satisfied_only_when_all_hold():
    on_pred = {**_PRED_OFF, "id": "on", "expected": True}
    report = evaluate_invariants([_PRED_OFF, on_pred], [_SWITCH])
    assert report["verdict"] == "unsatisfied"  # _PRED_OFF fails
    report = evaluate_invariants([on_pred], [_SWITCH])
    assert report["verdict"] == "satisfied"


def test_invariants_unknown_when_evidence_missing():
    report = evaluate_invariants([_PRED_OFF], [])
    assert report["verdict"] == "unknown"


def test_invariants_empty_is_unknown_never_auto_satisfied():
    assert evaluate_invariants([], [_SWITCH])["verdict"] == "unknown"


def test_describe_failures_names_expected_and_observed():
    report = evaluate_invariants([_PRED_OFF], [_SWITCH])
    text = describe_failures(report)
    assert "dark_theme_off" in text
    assert "expected checked=False" in text


# --- element signature sees state ----------------------------------------------


def test_signature_distinguishes_toggle_states():
    on = [_SWITCH]
    off = [{**_SWITCH, "checked": "false"}]
    assert not FlashRunner._signatures_similar(
        FlashRunner._element_signature(on),
        FlashRunner._element_signature(off),
    )


def test_signature_ignores_volatile_focus():
    a = [dict(_SWITCH)]
    b = [{**_SWITCH, "focused": "true"}]
    assert FlashRunner._signatures_similar(
        FlashRunner._element_signature(a),
        FlashRunner._element_signature(b),
    )


# --- control-only elements (unlabeled switches etc.) ----------------------------

# Raw fused nodes use the uiautomator shape (hyphenated keys, string attrs).
_RAW_SWITCH = {
    "text": "",
    "resource-id": "com.android.settings:id/switchWidget",
    "class": "android.widget.Switch",
    "checkable": "true",
    "checked": "false",
    "enabled": "true",
    "clickable": "true",
    "bounds": "[901,1661][1038,1787]",
}
_RAW_TITLE = {
    "text": "Dark theme",
    "resource-id": "android:id/title",
    "class": "android.widget.TextView",
    "checkable": "false",
    "checked": "false",
    "bounds": "[63,1663][333,1734]",
}


def test_control_elements_expose_unlabeled_switch_state():
    labeled = [
        {
            "index": 1,
            "text": "Dark theme",
            "class": "android.widget.TextView",
            "resource_id": "android:id/title",
            "bounds": [63, 1663, 333, 1734],
            "is_ocr": False,
        }
    ]
    controls = _control_only_elements([_RAW_TITLE, _RAW_SWITCH], labeled)
    assert len(controls) == 1
    el = controls[0]
    assert el["is_control_only"] and not el["is_ocr"]
    assert el["index"] == len(labeled) + 1  # appended after the labeled sequence
    assert el["checked"] == "false"
    # The predicate now sees the bare switch the compact list never showed
    # (unlabeled controls match on resource_id + class — there is no text).
    bare_switch_pred = {
        **_PRED_OFF,
        "selector": {
            "resource_id": "com.android.settings:id/switchWidget",
            "class": "android.widget.Switch",
        },
    }
    r = evaluate_predicate(bare_switch_pred, labeled + controls)
    assert r["status"] == "satisfied"


def test_control_elements_skip_textviews_and_duplicates():
    labeled = [
        {
            "index": 1,
            "text": "x",
            "class": "android.widget.Switch",
            "resource_id": "com.android.settings:id/switchWidget",
            "bounds": [901, 1661, 1038, 1787],
            "is_ocr": False,
        }
    ]
    controls = _control_only_elements([_RAW_TITLE, _RAW_SWITCH], labeled)
    assert controls == []  # title isn't stateful; switch already represented


# --- completion gate ------------------------------------------------------------


def _runner_with_invariants(invariants, final_check=False):
    r = FlashRunner.__new__(FlashRunner)
    r._goal_invariants = invariants
    r._plan_text = None
    r.ctx = SimpleNamespace(
        execution_setup=SimpleNamespace(
            goal_invariants=invariants,
            final_check_enabled=final_check,
        )
    )
    return r


@pytest.mark.asyncio
async def test_verify_completion_rejects_unsatisfied_invariant():
    r = _runner_with_invariants([_PRED_OFF])
    state = SimpleNamespace(indexed_elements=[_SWITCH])  # still ON
    allowed, findings = await r._verify_completion(state)
    assert not allowed
    assert "dark_theme_off" in findings


@pytest.mark.asyncio
async def test_verify_completion_releases_on_satisfied_invariants():
    r = _runner_with_invariants([_PRED_OFF])
    state = SimpleNamespace(indexed_elements=[{**_SWITCH, "checked": "false"}])
    allowed, _ = await r._verify_completion(state)
    assert allowed


@pytest.mark.asyncio
async def test_verify_completion_falls_through_on_unknown():
    # No invariants armed, no checker → existing fail-open behavior.
    r = _runner_with_invariants([], final_check=False)
    state = SimpleNamespace(indexed_elements=[])
    allowed, _ = await r._verify_completion(state)
    assert allowed
