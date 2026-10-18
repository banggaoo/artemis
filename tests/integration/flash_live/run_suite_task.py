#!/usr/bin/env python3
"""Suite variant of run_invariant_task.py — reads task defs from suite_spec.py.

Generalized oracles (text_equals / text_present / element_attr /
element_attr_text / night_mode), per-task seed actions, intent variants
(action / action+data / -n component), and the same frozen protocol:
real local Gemma endpoint only, no cloud prewarm, deterministic checks.
"""
import argparse
import asyncio
import json
import os
import re
import subprocess
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from spec import ADB, MODEL, ROOT, SERIAL
from suite_spec import TASKS

PARSER = argparse.ArgumentParser(description=__doc__)
PARSER.add_argument("--task", choices=TASKS, required=True)
PARSER.add_argument("--output-dir", type=Path, required=True)
ARGS = PARSER.parse_args()
TASK = TASKS[ARGS.task]
assert ARGS.output_dir.parent.is_dir()
ARGS.output_dir.mkdir(exist_ok=True)
os.environ["ARTEMIS_ARTEMIS_JSONC"] = str(ROOT / "local-config.json")
os.environ["ARTEMIS_TRACES_DIR"] = str(ARGS.output_dir / "traces")
os.environ["ARTEMIS_HELPER_AUTO_INSTALL"] = "false"
os.environ["ARTEMIS_CHECKER_ENABLED"] = "false"
os.environ["ARTEMIS_OUTPUTTER_ENABLED"] = "false"
os.environ["ARTEMIS_PLANNER_VALIDATION_ENABLED"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
if TASK.get("hierarchy_backend"):
    os.environ["ARTEMIS_HIERARCHY_BACKEND"] = TASK["hierarchy_backend"]

from artemis.agents.flash.runner import FlashRunner
from artemis.context import DevicePlatform
from artemis.sdk import Agent
import artemis.sdk.agent as sdk_agent
from third_party.mobile_use.main import new_default_config_builder

EVENTS = ARGS.output_dir / "events.jsonl"


def event(kind, **values):
    with EVENTS.open("a") as output:
        output.write(json.dumps({"kind": kind, "monotonic": time.perf_counter(), **values}, default=str) + "\n")


def adb(*parts, timeout=40, check=True):
    return subprocess.run(
        [ADB, "-s", SERIAL, *parts], capture_output=True, check=check, timeout=timeout,
    ).stdout


def launch(task=None):
    task = task or TASK
    pkg = task.get("package", "com.android.settings")
    if task.get("intent") == "main_activity":
        adb("shell", "am", "start", "-n", task["main_activity"], check=False)
    elif task.get("intent_data"):
        adb("shell", "am", "start", "-a", task["intent"], "-d", task["intent_data"], check=False)
    else:
        adb("shell", "am", "start", "-a", task["intent"], check=False)


def dump_xml(remote="/sdcard/suite_seed.xml"):
    """Best-effort hierarchy dump; returns (nodes, element_dicts)."""
    try:
        adb("shell", "uiautomator", "dump", remote, timeout=25, check=False)
        raw = adb("exec-out", "cat", remote, timeout=25, check=False)
    except Exception:
        return [], []
    if b"hierarchy" not in raw:
        return [], []
    return list(ET.fromstring(raw).iter("node")), []


def _center(node):
    m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", node.get("bounds") or "")
    if not m:
        return None
    l, t, r, b = (int(g) for g in m.groups())
    return (l + r) // 2, (t + b) // 2


def run_seed(seed_spec):
    stype = seed_spec["type"]
    if stype == "uimode":
        adb("shell", "cmd", "uimode", "night", seed_spec["value"])
    elif stype == "clear_app_data":
        adb("shell", "pm", "clear", TASK.get("package", ""), check=False)
    elif stype == "stopwatch_reset":
        launch_task = {"intent": "main_activity",
                       "main_activity": "com.google.android.deskclock/com.android.deskclock.DeskClock",
                       "package": "com.google.android.deskclock"}
        launch(launch_task)
        time.sleep(2)
        nodes, _ = dump_xml()
        # Go to the Stopwatch tab — prefer the clickable tab node over the
        # action-bar title that carries the same text.
        tab = next(
            (n for n in nodes
             if ((n.get("text") or "") == "Stopwatch" or (n.get("content-desc") or "") == "Stopwatch")
             and (n.get("clickable") or "") == "true"),
            None,
        ) or next(
            (n for n in nodes
             if (n.get("text") or "") == "Stopwatch" or (n.get("content-desc") or "") == "Stopwatch"),
            None,
        )
        if tab is not None:
            c = _center(tab)
            if c:
                adb("shell", "input", "tap", str(c[0]), str(c[1]), check=False)
                time.sleep(1)
        nodes, _ = dump_xml()
        for label in ("Stop", "Reset", "Pause"):
            for n in nodes:
                if (n.get("text") or "") == label or (n.get("content-desc") or "") == label:
                    c = _center(n)
                    if c:
                        adb("shell", "input", "tap", str(c[0]), str(c[1]), check=False)
                        time.sleep(0.5)
                    break
    elif stype == "ensure_element_attr":
        # Drive the target element's attribute to `expected` by tapping it
        # when the current value differs. Deterministic seed, same dump
        # mechanism the oracle uses.
        launch({"intent": seed_spec.get("intent", TASK["intent"]),
                "main_activity": seed_spec.get("main_activity") or TASK.get("main_activity", ""),
                "package": TASK["package"],
                "intent_data": seed_spec.get("intent_data")})
        time.sleep(2)
        want = seed_spec["expected"]
        for attempt in range(3):
            nodes, _ = dump_xml()
            hit = None
            for n in nodes:
                if seed_spec.get("resource_id") and n.get("resource-id") != seed_spec["resource_id"]:
                    continue
                if seed_spec.get("text") and (n.get("text") or "") != seed_spec["text"] and (n.get("content-desc") or "") != seed_spec["text"]:
                    continue
                hit = n
                break
            if hit is None and "deskclock" in TASK["package"] and seed_spec.get("text"):
                # Alarm rows live on the Alarm tab — prefer the clickable
                # tab node over the action-bar title carrying the same text.
                tab = next(
                    (n for n in nodes
                     if ((n.get("text") or "") == "Alarm" or (n.get("content-desc") or "") == "Alarm")
                     and (n.get("clickable") or "") == "true"),
                    None,
                ) or next(
                    (n for n in nodes
                     if (n.get("text") or "") == "Alarm" or (n.get("content-desc") or "") == "Alarm"),
                    None,
                )
                if tab is not None:
                    c = _center(tab)
                    if c:
                        adb("shell", "input", "tap", str(c[0]), str(c[1]), check=False)
                        time.sleep(1)
                continue
            if hit is None:
                event("seed_warn", seed=stype, detail="element not found")
                return
            if (hit.get(seed_spec["attribute"]) or "false") == want:
                return
            c = _center(hit)
            if not c:
                return
            adb("shell", "input", "tap", str(c[0]), str(c[1]), check=False)
            time.sleep(1)
        event("seed_warn", seed=stype, detail="attribute did not reach expected")


# Inject the declared invariants into every ExecutionSetup the SDK builds.
_ORIG_ES = sdk_agent.ExecutionSetup
def _es_with_invariants(**kwargs):
    inv = TASK.get("invariants") or []
    if inv:
        kwargs["goal_invariants"] = inv
    return _ORIG_ES(**kwargs)
sdk_agent.ExecutionSetup = _es_with_invariants


async def no_cloud_prewarm(self, api_key=None):
    event("prewarm_skipped", reason="Cloud prewarm disabled; all inference uses the real local endpoint.")


Agent._prewarm_llm_connections = no_cloud_prewarm
ORIGINAL_RUN = FlashRunner.run
RUNNER_RESULT = None


async def measured_run(self, state):
    global RUNNER_RESULT
    adb("shell", "am", "force-stop", TASK.get("package", "com.android.settings"))
    launch()
    await asyncio.sleep(1)
    event("ready", invariants=TASK.get("invariants") or [])
    start = time.perf_counter()
    RUNNER_RESULT = await ORIGINAL_RUN(self, state)
    event("runner_result", duration_s=time.perf_counter() - start, result=RUNNER_RESULT)
    return RUNNER_RESULT


FlashRunner.run = measured_run


def _fallback_elements(output_dir):
    """Newest element list from the run's trace DB (ui_tree stores JSON dicts)."""
    import sqlite3

    db = output_dir / "traces" / "data_engine.db"
    if not db.exists():
        return []
    rows = sqlite3.connect(db).execute(
        "select ui_tree from images where ui_tree is not null and ui_tree != ''"
        " order by timestamp desc limit 1"
    ).fetchall()
    if not rows:
        return []
    try:
        elements = json.loads(rows[0][0])
    except Exception:
        return []
    return elements if isinstance(elements, list) else []


def _norm_elements(xml_nodes, json_els):
    """Yield (rid, class, text, desc, attrs) across XML and JSON-fallback sources."""
    for n in xml_nodes:
        yield {
            "rid": n.get("resource-id"), "class": n.get("class"),
            "text": n.get("text"), "desc": n.get("content-desc"),
            "attrs": dict(n.attrib),
        }
    for e in json_els:
        yield {
            "rid": e.get("resource-id"), "class": e.get("class"),
            "text": e.get("text"), "desc": e.get("content-desc"),
            "attrs": e,
        }


def evaluate_oracle(oracle, xml_nodes, json_els, night_after):
    ot = oracle["type"]
    els = list(_norm_elements(xml_nodes, json_els))
    if ot == "night_mode":
        return night_after == oracle["expected"], night_after
    if ot == "system_setting":
        # Device-level truth for settings whose widget is filtered out of the
        # accessibility tree (e.g. Battery percentage: the Switch is
        # clickable=false/focusable=false so neither backend's live tree
        # surfaces it — only raw `uiautomator dump` does). The Settings.System
        # value is synchronous and unambiguous.
        val = adb(
            "shell", "settings", "get",
            oracle.get("namespace", "system"), oracle["key"], check=False,
        ).decode(errors="replace").strip()
        return val == str(oracle["expected"]), val
    if ot == "text_equals":
        hits = [e for e in els if e["rid"] == oracle["resource_id"]]
        observed = hits[0]["text"] if hits else None
        return bool(hits) and observed == oracle["expected"], observed
    if ot == "text_present":
        hits = [e for e in els if oracle["text"] in (e["text"] or "") or oracle["text"] in (e["desc"] or "")]
        return bool(hits), len(hits)
    if ot == "element_attr":
        hits = [e for e in els if e["rid"] == oracle["resource_id"]]
        observed = hits[0]["attrs"].get(oracle["attribute"]) if hits else None
        return bool(hits) and str(observed) == oracle["expected"], observed
    if ot == "element_attr_text":
        hits = [e for e in els if (e["text"] or "") == oracle["text"] or (e["desc"] or "") == oracle["text"]]
        observed = hits[0]["attrs"].get(oracle["attribute"]) if hits else None
        return bool(hits) and str(observed) == oracle["expected"], observed
    raise ValueError(f"unknown oracle type {ot}")


async def main():
    result = {"task": ARGS.task, "mode": "suite", "spec": TASK}
    agent = None
    try:
        for seed_spec in TASK.get("seed") or []:
            run_seed(seed_spec)
        (ARGS.output_dir / "before.png").write_bytes(adb("exec-out", "screencap", "-p", check=False))
        builder = new_default_config_builder()
        builder.for_device(DevicePlatform.ANDROID, SERIAL)
        builder.with_video_recording_tools(enabled=False)
        builder.with_checker(enabled=False)
        builder.with_planner_validation(enabled=False)
        builder.with_outputter(enabled=False)
        builder.with_flash_step_summarizer(enabled=False)
        agent = Agent(config=builder.build())
        await agent.init(retry_count=1, retry_wait_seconds=1)
        returned = await asyncio.wait_for(
            agent.run_task(goal=TASK["goal"], profile="flash", name=f"suite_{ARGS.task}"),
            timeout=TASK.get("timeout", 420),
        )
        result["sdk_result"] = returned
        result["runner_result"] = RUNNER_RESULT
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
    finally:
        # Capture final UI state while the app is still alive: clean(force)
        # force-stops the target, and DB-backed widgets (DeskClock alarms)
        # flush their toggle write asynchronously — a post-clean dump can
        # show reverted state the UI never displayed.
        # uiautomator itself can return a stale attribute on a rebinding view
        # (observed: teardown dump read checked=false while the same-moment
        # screenshot and a later dump both showed the switch on), so the
        # oracle votes over up to three dumps rather than trusting one.
        remote = "/sdcard/artemis_suite_final.xml"

        def _dump_once():
            for _attempt in range(3):
                try:
                    adb("shell", "uiautomator", "dump", remote, timeout=30, check=False)
                    data = adb("exec-out", "cat", remote, timeout=30, check=False)
                    if b"hierarchy" in data:
                        return data
                except Exception as capture_err:
                    event("final_dump_retry", attempt=_attempt, error=str(capture_err))
                time.sleep(2)
            return b""

        def _oracle_on(xml_bytes):
            if b"hierarchy" not in xml_bytes:
                return evaluate_oracle(
                    TASK["oracle"], [], _fallback_elements(ARGS.output_dir), night
                )
            return evaluate_oracle(
                TASK["oracle"],
                list(ET.fromstring(xml_bytes).iter("node")),
                [],
                night,
            )

        night = adb("shell", "cmd", "uimode", "night", check=False).decode(errors="replace").strip()
        # Primary evidence: the newest live-run capture. The run's own
        # observation path (UIAutomator2/helper, re-observed fresh before every
        # invariant decision) proved fresher than a post-run shell `uiautomator
        # dump`, which can serve stale node attributes — observed: two dumps
        # read checked=false while the same-moment screenshot showed the
        # switch on and the in-run captures had been stable true for ~2min.
        trace_els = _fallback_elements(ARGS.output_dir)
        trace_verdict = None
        if trace_els:
            try:
                trace_verdict = evaluate_oracle(TASK["oracle"], [], trace_els, night)
            except Exception as oracle_err:
                result["oracle_error"] = str(oracle_err)
        dumps = [_dump_once()]
        verdicts = []
        try:
            verdicts.append(_oracle_on(dumps[0]))
        except Exception as oracle_err:
            result["oracle_error"] = str(oracle_err)
        while len(dumps) < 3 and (
            len(verdicts) < 2 or verdicts[-1][0] != verdicts[-2][0]
        ):
            time.sleep(2)
            dumps.append(_dump_once())
            try:
                verdicts.append(_oracle_on(dumps[-1]))
            except Exception as oracle_err:
                result["oracle_error"] = str(oracle_err)
                break
        if len(dumps) > 2:
            result["oracle_unstable"] = True
            event("oracle_unstable", verdicts=[str(v) for v in verdicts])
        xml = dumps[-1]
        result["final_xml_source"] = "uiautomator" if b"hierarchy" in xml else "trace_db_fallback"
        result["oracle_dump_votes"] = [bool(v[0]) for v in verdicts]
        result["oracle_trace"] = (
            {"achieved": bool(trace_verdict[0]), "observed": str(trace_verdict[1])}
            if trace_verdict is not None
            else None
        )
        if TASK["oracle"]["type"] == "system_setting":
            ok, obs = evaluate_oracle(TASK["oracle"], [], [], night)
            result["oracle_source"] = "system_setting"
            result["state_achieved"] = bool(ok)
            result["oracle_observed"] = obs
            result["oracle_dump_votes"] = [bool(ok)]
        elif trace_verdict is not None:
            result["oracle_source"] = "trace_last"
            result["state_achieved"] = bool(trace_verdict[0])
            result["oracle_observed"] = trace_verdict[1]
            if verdicts and verdicts[-1][0] != trace_verdict[0]:
                result["oracle_disagreement"] = True
        elif verdicts:
            result["oracle_source"] = "dump_vote"
            result["state_achieved"] = sum(bool(v[0]) for v in verdicts) > len(verdicts) / 2
            for v in reversed(verdicts):
                if bool(v[0]) == result["state_achieved"]:
                    result["oracle_observed"] = v[1]
                    break
        else:
            result["state_achieved"] = False
        (ARGS.output_dir / "final.xml").write_bytes(xml)
        (ARGS.output_dir / "after.png").write_bytes(adb("exec-out", "screencap", "-p", check=False))
        if agent is not None:
            await agent.clean(force=True)
        result["night_after"] = night
        result["reported_completed"] = isinstance(RUNNER_RESULT, dict) and RUNNER_RESULT.get("status") == "completed"
        result["verified_success"] = result["state_achieved"] and result["reported_completed"]
        with (ARGS.output_dir / "result.json").open("x") as output:
            json.dump(result, output, indent=2, default=str)
            output.write("\n")
        print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
