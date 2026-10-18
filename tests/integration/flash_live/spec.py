"""Live-eval configuration: on-device model, emulator, frozen protocol.

Everything machine-specific reads from the environment so the suite runs
on any host with the same pieces:

  ARTEMIS_LIVE_ADB        adb binary        (default: platform-tools adb on PATH)
  ARTEMIS_LIVE_SERIAL     device serial     (default: emulator-5554)
  ARTEMIS_LIVE_MODEL      served model id   (default: local gemma-4-e4b 4-bit)
  ARTEMIS_LIVE_API_BASE   OpenAI-compatible endpoint (default: 127.0.0.1:8097)

`prepare()` writes local-config.json next to this file; run it once per
checkout. No cloud inference: every node is pinned to the local endpoint.
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ADB = os.environ.get(
    "ARTEMIS_LIVE_ADB",
    shutil.which("adb")
    or "/Users/james/Library/Android/sdk/platform-tools/adb",
)
SERIAL = os.environ.get("ARTEMIS_LIVE_SERIAL", "emulator-5554")
MODEL = os.environ.get(
    "ARTEMIS_LIVE_MODEL",
    "/Users/james/.ondevice-agent-platform/models/"
    "mlx-community--gemma-4-e4b-it-4bit__475b9088d29754a3379866cf5aeb6b41acd313c2",
)
API_BASE = os.environ.get("ARTEMIS_LIVE_API_BASE", "http://127.0.0.1:8097/v1")

LLM = {
    "provider": "custom", "model": MODEL, "api_base": API_BASE,
    "temperature": 0.0, "max_tokens": 512, "timeout": 120,
    "coordinate_format": "yx_1000",
}
NODES = (
    "planner", "summarizer", "operator", "operator_summarizer",
    "log_reader_sub_agent", "log_analyzer", "diagnoser", "checker",
    "planner_avatar", "history_analyzer_expert", "diagnoser_expert", "explorer",
    "history_analyzer", "planner_validation", "validator_pixel_safety_net", "output_analyzer",
)


def prepare():
    llm = {**LLM, "fallback": dict(LLM)}
    config = {name: dict(llm) for name in NODES}
    config["utils"] = {
        name: dict(llm) for name in ("outputter", "hopper", "video_analyzer", "object_detector")
    }
    config["agent"] = {
        # 18 turns: a 7-digit dial needs 7 action turns + a report, plus
        # recovery slack for index corrections and backspace repair on the
        # dense pad (observed: two runs burned ~13 turns on mis-dials).
        "flash": {"max_turns": 18, "step_summarizer": {"enabled": False}},
        "decision": {"enabled": False, "flash_loop": False},
        "pro": {
            "checker": {"enabled": False},
            "planner_validation": {"enabled": False},
            "committee": {"enabled": False},
        },
        "video_analyzer": {"enabled": False, "enable_ledger": False},
        "outputter": {"enabled": False},
        "memory": {"chunking": {"enabled": False}},
    }
    with (ROOT / "local-config.json").open("x") as output:
        json.dump(config, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    prepare()
