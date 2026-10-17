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

"""Deterministic UI-state predicates for Flash verification.

A predicate pins one element (by selector) to an expected attribute value
and evaluates it against the live element list (``state.indexed_elements``,
the dicts produced by ``format_minimal_list_with_elements``). The verdict is
always one of ``satisfied`` / ``unsatisfied`` / ``unknown`` — absence,
ambiguity or missing state is ``unknown``, never assumed ``unsatisfied``.

A satisfied predicate proves only the condition it declares for the
observation it ran on — it does not by itself prove the whole task.
``evaluate_invariants`` aggregates the declared ``goal_invariants`` into a
single gate used at the pre-action short-circuit and the completion
boundary (``FlashRunner._verify_completion``).

Predicate schema::

    {
        "id": "dark_theme_off",
        "selector": {
            "resource_id": "com.android.settings:id/switchWidget",
            "class": "android.widget.Switch",
            "text": "Dark theme",
        },
        "attribute": "checked",
        "expected": False,
    }
"""

from __future__ import annotations

from typing import Any

_SELECTOR_KEYS = {"resource_id", "class", "text"}
_STATE_ATTRIBUTES = {"checked", "enabled", "selected"}


def _as_bool(value: Any) -> bool | None:
    """Coerce uiautomator attribute strings to bool; None when not boolean."""
    if type(value) is bool:
        return value
    if isinstance(value, str):
        return {"true": True, "false": False}.get(value.strip().lower())
    return None


def evaluate_predicate(predicate: dict[str, Any], elements: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluates one predicate against the current element list.

    Returns ``{"status": ..., "reason": ..., "predicate_id": ..., "evidence": {...}}``
    where status is ``satisfied``, ``unsatisfied`` or ``unknown``.
    """
    predicate_id = predicate.get("id") if isinstance(predicate, dict) else None
    evidence: dict[str, Any] = {"source": "xml"}

    selector = predicate.get("selector") if isinstance(predicate, dict) else None
    attribute = predicate.get("attribute") if isinstance(predicate, dict) else None
    expected = predicate.get("expected") if isinstance(predicate, dict) else None
    if (
        not isinstance(predicate, dict)
        or not isinstance(selector, dict)
        or not selector
        or not set(selector) <= _SELECTOR_KEYS
        or any(not isinstance(v, str) or not v.strip() for v in selector.values())
        or attribute not in _STATE_ATTRIBUTES
        or type(expected) is not bool
    ):
        return {
            "status": "unknown",
            "reason": "unsupported_predicate",
            "predicate_id": predicate_id,
            "evidence": evidence,
        }

    # OCR-sourced elements are not XML state — a match on detected pixels
    # would be a guess about a control the hierarchy never declared.
    matches = [
        el
        for el in elements
        if isinstance(el, dict)
        and not el.get("is_ocr")
        and all(el.get(key) == value for key, value in selector.items())
    ]
    evidence["matches"] = len(matches)
    if len(matches) != 1:
        return {
            "status": "unknown",
            "reason": "target_absent" if not matches else "ambiguous_target",
            "predicate_id": predicate_id,
            "evidence": evidence,
        }

    target = matches[0]
    observed = _as_bool(target.get(attribute))
    evidence.update(
        attribute=attribute,
        observed=observed,
        expected=expected,
        target={
            key: target.get(key) for key in ("index", "resource_id", "class", "text", "bounds")
        },
    )
    if observed is None:
        return {
            "status": "unknown",
            "reason": "attribute_not_known",
            "predicate_id": predicate_id,
            "evidence": evidence,
        }
    return {
        "status": "satisfied" if observed == expected else "unsatisfied",
        "reason": "unique_target_known_value",
        "predicate_id": predicate_id,
        "evidence": evidence,
    }


def evaluate_invariants(
    predicates: list[dict[str, Any]] | None, elements: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregates declared goal invariants into one gate.

    ``verdict`` is ``satisfied`` only when EVERY predicate is satisfied;
    ``unsatisfied`` when ANY is unsatisfied (deterministic contradiction);
    ``unknown`` otherwise (nothing provable either way — including the
    empty-predicate case, which must never auto-close a task).
    """
    results = [evaluate_predicate(p, elements or []) for p in predicates or []]
    if not results:
        verdict = "unknown"
    elif any(r["status"] == "unsatisfied" for r in results):
        verdict = "unsatisfied"
    elif all(r["status"] == "satisfied" for r in results):
        verdict = "satisfied"
    else:
        verdict = "unknown"
    return {"verdict": verdict, "results": results}


def describe_failures(report: dict[str, Any]) -> str:
    """One-line-per-predicate findings for feedback into the model turn."""
    lines = []
    for r in report.get("results", []):
        if r["status"] != "unsatisfied":
            continue
        ev = r.get("evidence", {})
        target = ev.get("target") or {}
        lines.append(
            f"[invariant failed] '{r.get('predicate_id') or target.get('text')}'"
            f" expected {ev.get('attribute')}={ev.get('expected')}"
            f" but observed {ev.get('observed')}"
        )
    return "\n".join(lines) or "Declared goal condition is not satisfied."
