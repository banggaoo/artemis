#!/usr/bin/env python3
"""Batch driver: N executions of the complex-UI suite, serial on one emulator.

Usage (from this directory, inside the project venv):
    python spec.py            # once per checkout — writes local-config.json
    python run_batch.py --out /tmp/suite-out --runs-per-task 6

Resumable: skips dirs that already hold result.json. One hung run never
blocks the batch (hard per-run timeout = task timeout + 180s grace).
Seed-major ordering gives every task family coverage in the first pass.
Aggregates report.json incrementally after every run.
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from spec import ADB, API_BASE, ROOT, SERIAL
from suite_spec import TASKS

PY = sys.executable
RUNNER = ROOT / "run_suite_task.py"
ENDPOINT = f"{API_BASE}/models"

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--out", type=Path, required=True)
PARSER.add_argument("--runs-per-task", type=int, default=7)
PARSER.add_argument("--tasks", nargs="*", default=None, help="subset of task names")
ARGS = PARSER.parse_args()

tasks = ARGS.tasks or list(TASKS)
total = len(tasks) * ARGS.runs_per_task
out = ARGS.out
out.mkdir(parents=True, exist_ok=True)


def llm_alive():
    try:
        with urllib.request.urlopen(ENDPOINT, timeout=5) as r:
            return b"gemma" in r.read().lower()
    except Exception:
        return False


def adb_alive():
    try:
        subprocess.run([ADB, "-s", SERIAL, "shell", "echo", "ok"],
                       capture_output=True, timeout=8, check=True)
        return True
    except Exception:
        return False


def write_report():
    rows = []
    for d in sorted(out.iterdir()):
        rj = d / "result.json"
        if not rj.exists():
            continue
        try:
            r = json.loads(rj.read_text())
        except Exception:
            continue
        rows.append({
            "run": d.name,
            "task": r.get("task"),
            "verified_success": r.get("verified_success"),
            "state_achieved": r.get("state_achieved"),
            "reported_completed": r.get("reported_completed"),
            "runner_status": (r.get("runner_result") or {}).get("status") if isinstance(r.get("runner_result"), dict) else None,
            "error": r.get("error"),
            "oracle_observed": r.get("oracle_observed"),
        })
    by_task = {}
    for row in rows:
        t = by_task.setdefault(row["task"], {"runs": 0, "verified": 0, "achieved": 0, "completed": 0})
        t["runs"] += 1
        t["verified"] += 1 if row["verified_success"] else 0
        t["achieved"] += 1 if row["state_achieved"] else 0
        t["completed"] += 1 if row["reported_completed"] else 0
    report = {
        "total_planned": total,
        "runs_finished": len(rows),
        "verified_success": sum(1 for r in rows if r["verified_success"]),
        "by_task": by_task,
        "rows": rows,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def wait_healthy(max_wait=300):
    """Poll the endpoint+device for up to max_wait; True when both answer."""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if llm_alive() and adb_alive():
            return True
        time.sleep(20)
    return False


# Seed-major order: every task family gets one run before the second seed of
# anything starts, so a per-family harness problem surfaces in the first pass
# instead of after hours of one task type.
done = 0
for seed_i in range(ARGS.runs_per_task):
    for task_name in tasks:
        run_dir = out / f"{task_name}__s{seed_i}"
        if (run_dir / "result.json").exists():
            done += 1
            continue
        if not wait_healthy():
            print(f"[{done}/{total}] SKIP {task_name} s{seed_i}: endpoint or device dead", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "driver_error.marker").write_text("llm/adb unreachable >5min\n")
            write_report()
            continue
        timeout = TASKS[task_name].get("timeout", 420) + 180
        print(f"[{done}/{total}] RUN {task_name} s{seed_i} (hard cap {timeout}s)", flush=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        log = (run_dir / "run.log").open("w")
        try:
            subprocess.run(
                [PY, str(RUNNER), "--task", task_name, "--output-dir", str(run_dir)],
                timeout=timeout, stdout=log, stderr=subprocess.STDOUT,
            )
        except subprocess.TimeoutExpired:
            (run_dir / "timeout.marker").write_text(f"killed at {timeout}s\n")
        except Exception as e:
            (run_dir / "driver_error.marker").write_text(str(e) + "\n")
        finally:
            log.close()
        done += 1
        rep = write_report()
        v = rep["verified_success"]
        print(f"    -> {v}/{rep['runs_finished']} verified so far", flush=True)
        time.sleep(4)

final = write_report()
print(json.dumps({k: final[k] for k in ("runs_finished", "verified_success", "by_task")}, indent=2))
