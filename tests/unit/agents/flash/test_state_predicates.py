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


# --- text attribute ------------------------------------------------------------

_DIGITS = {
    "index": 3,
    "text": "555-0155",
    "class": "android.widget.EditText",
    "resource_id": "com.google.android.dialer:id/digits",
    "bounds": [147, 1196, 933, 1340],
    "center": [540, 1268],
    "enabled": "true",
    "is_ocr": False,
}

_PRED_DIGITS = {
    "id": "dialed_5550155",
    "selector": {
        "resource_id": "com.google.android.dialer:id/digits",
        "class": "android.widget.EditText",
    },
    "attribute": "text",
    "expected": "555-0155",
}


def test_text_predicate_satisfied_on_exact_match():
    r = evaluate_predicate(_PRED_DIGITS, [_DIGITS])
    assert r["status"] == "satisfied"
    assert r["evidence"]["observed"] == "555-0155"


def test_text_predicate_unsatisfied_on_partial_input():
    r = evaluate_predicate(_PRED_DIGITS, [{**_DIGITS, "text": "555-0"}])
    assert r["status"] == "unsatisfied"


def test_text_predicate_unknown_when_text_absent():
    r = evaluate_predicate(_PRED_DIGITS, [{k: v for k, v in _DIGITS.items() if k != "text"}])
    assert r["status"] == "unknown"
    assert r["reason"] == "attribute_not_known"


def test_text_predicate_rejects_bool_expected():
    bad = {**_PRED_DIGITS, "expected": True}
    r = evaluate_predicate(bad, [_DIGITS])
    assert r["status"] == "unknown"
    assert r["reason"] == "unsupported_predicate"


def test_bool_predicate_rejects_string_expected():
    bad = {**_PRED_OFF, "expected": "false"}
    r = evaluate_predicate(bad, [_SWITCH])
    assert r["status"] == "unknown"
    assert r["reason"] == "unsupported_predicate"


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


def _sig(elements):
    from artemis.agents.flash.runner import FlashRunner

    return FlashRunner._element_signature(elements)


def test_signature_dense_screen_small_change_counts():
    # 30-node dial pad: appending one digit changes exactly one label —
    # Jaccard 0.8 called this 'unchanged'; strict equality must not.
    pad = [
        {"index": i, "text": str(i % 10), "center": [100 * (i % 3) + 50, 1400 + 190 * (i // 3)], "package": "com.google.android.dialer"}
        for i in range(29)
    ]
    pad.append({**_DIGITS, "package": "com.google.android.dialer"})
    before = _sig(pad)
    after = _sig([*pad[:-1], {**_DIGITS, "text": "555-015", "package": "com.google.android.dialer"}])
    assert before != after


def test_signature_excludes_status_bar_and_ocr_jitter():
    app = [{"index": 1, "text": "five", "center": [540, 1649], "package": "com.google.android.dialer"}]
    noise_a = [
        *app,
        {"index": 2, "text": "19:08", "center": [60, 40], "package": "com.android.systemui"},
        {"index": 3, "text": "OCR read A", "center": [500, 200], "is_ocr": True},
    ]
    noise_b = [
        *app,
        {"index": 2, "text": "19:09", "center": [60, 40], "package": "com.android.systemui"},
        {"index": 3, "text": "OCR read B", "center": [500, 200], "is_ocr": True},
    ]
    assert _sig(noise_a) == _sig(noise_b)


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


# --- pixel-level stale-tree oracle --------------------------------------------


def _png_bytes(width=64, height=64, box=None):
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "black")
    if box:
        ImageDraw.Draw(img).rectangle(box, fill="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_screens_differ_flags_a_small_real_change():
    from artemis.agents.flash.runner import FlashRunner

    before = _png_bytes()
    after = _png_bytes(box=(10, 10, 24, 24))  # one appended digit's worth
    assert FlashRunner._screens_differ(before, after)


def test_screens_differ_stays_quiet_on_identical_frames():
    from artemis.agents.flash.runner import FlashRunner

    a = _png_bytes(box=(5, 5, 30, 30))
    b = _png_bytes(box=(5, 5, 30, 30))
    assert not FlashRunner._screens_differ(a, b)


def test_screens_differ_refuses_missing_or_garbage_bytes():
    from artemis.agents.flash.runner import FlashRunner

    assert not FlashRunner._screens_differ(None, b"x")
    assert not FlashRunner._screens_differ(b"x", None)
    assert not FlashRunner._screens_differ(b"not a png", b"also not")


def test_describe_failures_locates_the_digit_divergence():
    """A text invariant on a formatted field: the finding names the shared
    digit prefix and both remainders — '(555) 001-0555' vs '555-0155' shares
    '5550' then diverges."""
    from artemis.agents.flash.state_predicates import describe_failures, evaluate_invariants

    pred = {
        "id": "dialed",
        "selector": {"resource_id": "com.google.android.dialer:id/digits"},
        "attribute": "text",
        "expected": "555-0155",
    }
    el = {
        "index": 1, "text": "(555) 001-0555",
        "resource_id": "com.google.android.dialer:id/digits",
        "class": "android.widget.EditText", "is_ocr": False,
    }
    report = evaluate_invariants([pred], [el])
    text = describe_failures(report)
    assert "digits 1-4 '5550' match" in text
    assert "expected continues '155'" in text
    assert "observed continues '010555'" in text


def test_digit_prefix_diff_handles_formatting_noise():
    from artemis.agents.flash.state_predicates import _digit_prefix_diff

    assert "digits 1-4" in _digit_prefix_diff("555-0155", "(555) 001-0555")
    assert _digit_prefix_diff("555-0155", "555-0155") is None
    assert _digit_prefix_diff("", "123") is None
