#!/usr/bin/env python3
"""Aggregate the suite campaign into a failure-mode report.

Reads every <out>/<task>__s<i>/result.json + run.log, groups by task and by
failure class, and writes <out>/analysis.json + a printed table.
"""
import json
import re
import statistics
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/suite-out")


def classify(result, log_text):
    if result.get("verified_success"):
        return "verified"
    err = (result.get("error") or "").lower()
    runner = result.get("runner_result") or {}
    status = runner.get("status")
    expl = (runner.get("explanation") or "").lower()
    if "timed out" in err or "timeout" in err:
        return "timeout"
    if status == "failed" and ("suppressed" in expl or "no effect" in expl):
        return "dead_tap_escalation"
    if result.get("reported_completed") and not result.get("state_achieved"):
        return "claimed_without_state"
    if status == "failed":
        return "runner_failed"
    if not result.get("reported_completed"):
        return "incomplete"
    return "other"


rows = []
for d in sorted(OUT.iterdir()):
    rj = d / "result.json"
    if not rj.exists() or not d.is_dir():
        continue
    try:
        r = json.loads(rj.read_text())
    except Exception:
        continue
    log = ""
    lf = d / "run.log"
    if lf.exists():
        try:
            log = lf.read_text(errors="replace")
        except Exception:
            pass
    turns = [int(m) for m in re.findall(r"Turn (\d+)/\d+", log)]
    rows.append({
        "run": d.name,
        "task": r.get("task"),
        "class": classify(r, log),
        "turns": max(turns) if turns else None,
        "oracle_source": r.get("oracle_source"),
        "oracle_disagreement": bool(r.get("oracle_disagreement")),
        "oracle_observed": r.get("oracle_observed"),
    })

by_task = {}
for row in rows:
    t = by_task.setdefault(row["task"], {"runs": 0, "verified": 0, "classes": {}, "turns": []})
    t["runs"] += 1
    t["verified"] += row["class"] == "verified"
    t["classes"][row["class"]] = t["classes"].get(row["class"], 0) + 1
    if row["turns"]:
        t["turns"].append(row["turns"])

for t in by_task.values():
    t["median_turns"] = statistics.median(t["turns"]) if t["turns"] else None
    t.pop("turns", None)

classes = {}
for row in rows:
    classes[row["class"]] = classes.get(row["class"], 0) + 1

report = {
    "runs": len(rows),
    "verified": sum(r["class"] == "verified" for r in rows),
    "rate": round(sum(r["class"] == "verified" for r in rows) / max(1, len(rows)), 3),
    "by_class": classes,
    "by_task": by_task,
    "oracle_disagreements": sum(r["oracle_disagreement"] for r in rows),
    "rows": rows,
}
(OUT / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")

print(f"{'task':28} {'verified':>9}  classes")
for name in sorted(by_task):
    t = by_task[name]
    cls = " ".join(f"{k}:{v}" for k, v in sorted(t["classes"].items()))
    print(f"{name:28} {t['verified']}/{t['runs']:<7} {cls}")
print(f"\nTOTAL {report['verified']}/{report['runs']} = {report['rate']:.1%}  oracle_disagreements={report['oracle_disagreements']}")
print("by_class:", classes)
