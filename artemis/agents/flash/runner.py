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

"""Universal Multi-Model FlashRunner for Artemis.

Executes autonomous reactive mobile workflows across Google Gemini,
OpenAI GPT-4o/o3, Anthropic Claude 3.5/3.7, and OpenRouter endpoints.

Conversation shape (shared with the Pro operator, history redesign §3.2):
the prompt is built every turn from a session :class:`TranscriptLedger` —
a byte-stable system prefix, the committed earlier turns (append-only; old UI
lists stripped at depth 1, screenshots resolved to visual summaries at depth
K, long spans chunk-compressed), and a fresh tail carrying the current
observation under a ``# CURRENT OBSERVATION [T+mm:ss]`` header. Actions run
synchronously and their tool messages already carry the outcome; only a turn
whose action failed ends with an ``--- Action Execution Result (T+mm:ss) ---``
message (it carries the error). Every timestamp the model sees is a
session-relative offset — the same clock the video analyzer uses for the
session recording.

Per-turn notices on the tail (the reasoning reminder after a silent turn, the
no-tool-call nudge, the final-turn warning, the user-guidance wrapper) are
marked ephemeral: the ledger's scrub edge deletes them once the message leaves
the recent window, so they never reach the frozen history or a chunk capsule.
The verbatim body of a user instruction is the exception: it stays in the
active window until its turn is chunked, whose user lines then carry it on.
"""

import asyncio
import base64
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import uuid

from jinja2 import Template
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from artemis.agents.flash.summarizer import VisualStepSummarizer, build_focus_context
from artemis.agents.operator.prompts import REASONING_REMINDER, UserGuidance, render_user_guidance
from artemis.agents.validator.tool_declarations import (
    ASK_EXPLORER_TOOL,
    CLICK_SEQUENCE_TOOL,
    LIST_NOTES_TOOL,
    READ_NOTE_TOOL,
    REPORT_TASK_STATUS_TOOL,
    VALIDATOR_TOOLS_DECLARATION,
    capture_screenshot_and_parse_ui,
    prune_intermediate_screenshots,
)
from artemis.config import (
    MemoryRuntimeConfig,
    MemoryTranscriptConfig,
    StepSummarizerConfig,
    load_agent_config,
    settings,
)
from artemis.context import ArtemisContext
from artemis.controllers.unified_controller import UnifiedMobileController
from artemis.data_engine.trace import trace
from artemis.graph.perception import _check_injected_instruction_file
from artemis.graph.state import State
from artemis.llm.router import ModelProvider
from artemis.llm.structured import ParseFailure, parse_structured
from artemis.core.decision_client import DecisionClient
from artemis.agents.flash.state_predicates import (
    describe_failures,
    evaluate_invariants,
)
from artemis.mcp.action_executor import McpActionExecutor
from artemis.utils.image_codec import image_data_uri
from artemis.utils.image_hash import dhash_hex, hamming_distance_hex
from artemis.mcp.observation import observe
from artemis.memory.transcript import PRO_UI_LIST_MARKER, TranscriptLedger, mark_ephemeral
from artemis.services.llm import (
    LLMCallError,
    RobustChatModelWrapper,
    acomplete,
    get_google_llm,
    get_llm,
    invoke_llm_with_timeout_message,
)
from artemis.tools.history import history_tool_declarations
from artemis.tools.tool_wrapper import tool_result_messages
from artemis.utils.coordinates import (
    COORDINATE_SPACE_KEY,
    COORDINATE_SPACE_NORMALIZED,
    parse_swipe_parameters,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

_NO_TOOL_CALL_NOTICE = (
    "You did not call any tools last turn. Please make progress by calling an"
    " action tool or 'report_task_status'."
)

# Hard ceiling on consecutive narration-only turns. Some local VLMs (e.g.
# Gemma 4 E4B under mlx_vlm) can fall into an apology loop — seeing the
# goal, narrating a plan, never emitting a tool call — and with an
# unlimited turn budget that loop never exits on its own.
_MAX_CONSECUTIVE_NO_TOOL_CALLS = 8

# Gemma-family tool-call envelopes observed in the wild: the model emits
# `call:<name>{<args>}` and some servers (mlx_vlm) wrap it into a function
# literally named "call" whose arguments are the unparseable remainder, or
# the envelope leaks into text content.
_GEMMA_CALL_RE = re.compile(r"(?:<\|?tool_call\|?>?\s*)?call:([a-zA-Z_]\w*)\s*\{")
_GEMMA_ENVELOPED_RE = re.compile(r'\s*\{?\s*"?([a-zA-Z_]\w*)"?\s*\{')
_GEMMA_QUOTE_RE = re.compile(r'<\|\\?\"\|>')


def _loads_relaxed_args(raw: str):
    """Best-effort JSON parse of a mangled Gemma `name{key:value}` args object."""
    s = raw.strip()
    s = _GEMMA_QUOTE_RE.sub('"', s)
    # Quote bare keys: {key: -> {"key": and ,key: -> ,"key":
    s = re.sub(r'([{,])\s*"?([a-zA-Z_]\w*)"?\s*:', r'\1"\2":', s)
    try:
        return json.loads(s)
    except ValueError:
        return None


def _parse_call_syntax(raw: str, match=None):
    """Recovers `name{args}` at ``match`` — `call:name{...}` or `{name{...}`."""
    if match is None:
        match = _GEMMA_ENVELOPED_RE.match(raw)
        if not match:
            return None
    name = match.group(1)
    start = match.end() - 1  # opening '{' of the args object
    args = None
    depth = 0
    for i in range(start, len(raw)):
        ch = raw[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                args = _loads_relaxed_args(raw[start:i + 1])
                break
    if depth:
        args = _loads_relaxed_args(raw[start:] + '}')
    if not isinstance(args, dict):
        return None
    return {"name": name, "args": args, "id": str(uuid.uuid4())}


_FINAL_TURN_WARNING = "[WARNING] This is your final turn; only 'report_task_status' is available."

# Hybrid Flash loop (decision.flash_loop): the VLM perceives, the typed
# decision model acts. The describe call carries NO tool declarations, so
# the tool-call envelope syntax that local VLMs mangle can never be
# emitted; action selection is a fixed-space `choose` over the live
# element list plus meta-actions instead.
_HYBRID_PLAN_PROMPT = """You are the planning stage of a UI testing agent.
GOAL: {{ goal }}

Given the attached CURRENT screenshot and element list, decompose the goal
into 3-6 short ordered milestones (one per line, starting with '- '). Each
milestone is a SCREEN STATE to reach or a concrete interaction ON THE GOAL
PATH — e.g. "open the app drawer (swipe up on home)", "Settings main list
open", "Dark theme toggle enabled". Only include steps that move toward the
goal; NEVER enumerate visible elements unrelated to it (do not list "tap
<app icon>" for every icon — a milestone is progress, not an inventory).
If the first target is not visible, the first milestone is how to reach it.
Output only the milestone list."""

_HYBRID_DESCRIBE_PROMPT = """You are the perception stage of a UI testing agent.
GOAL: {{ goal }}
PLAN:
{{ plan }}

Describe the CURRENT screenshot factually in 3-5 short sentences: which app
and screen is shown, what has changed since the previous observation, and
where the elements relevant to the goal sit. If the goal's next target is
NOT visible, say so and note how it would be reached (e.g. open the app
drawer, scroll, navigate back). When a goal-relevant toggle, switch, or
control is visible, state its CURRENT value explicitly (on/off, checked,
enabled, the shown selection) — and if the goal state is already satisfied,
say so plainly rather than describing the screen generically. Then list
EVERY interactive element from the attached list, one per line: [index]
label ~[x,y] — the approximate normalized 0-1000 coordinate is only
required for elements visible on screen but NOT in the list. End with
exactly one marker line:
TARGET: visible — the goal's next interactive target is on this screen
TARGET: not-visible — it is not; navigation is needed first
TARGET: reached — the goal state is already achieved
Never claim an action was taken. Do not decide what to do next — a
separate decision stage reads your description."""

# Menu entries the decision model may always pick, alongside the element
# taps generated from the live index list. Free-text input is excluded by
# design: a fixed answer space cannot produce the text to type, so a goal
# that needs text input escalates to the standard tool-calling turn.
_HYBRID_META_ACTIONS = {
    "swipe_up": "scroll the screen up to reveal content below",
    "swipe_down": "scroll the screen down to reveal content above",
    "swipe_left": "swipe left (e.g. carousel, page forward)",
    "swipe_right": "swipe right (e.g. carousel, page back)",
    "press_back": "press the Android back button",
    "press_home": "press the Android home button",
    "wait": "wait briefly for the screen to settle or load",
    "done": "the goal is satisfied — finish the task as completed",
    "fail": "the goal cannot be achieved from here — finish as failed",
}

# Cap on element-tap entries in the decision menu: the decision model's
# context is short (~512-1024 tokens) and taps beyond the first screenful
# are rarely the right action anyway.
_HYBRID_MAX_MENU_ELEMENTS = 15

# Consecutive turns that re-executed an already-tried menu key before the
# hybrid decide is skipped entirely and the turn escalates to the VLM —
# the masked menu plus the notice usually break the loop first, but the
# standard path is the fallback when even that fails.
_HYBRID_REPEAT_ESCALATE = 2

# Distilled state sent to the decision model is capped for the same reason:
# goal + plan + screen description, truncated tail-first.
_HYBRID_MAX_STATE_CHARS = 2000


@dataclass
class _TurnRecord:
    """What one reactive turn produced, for committing it into the ledger.

    ``step_keys`` are the DataEngine step ids recorded this turn (one per
    executed action; the tool_call_id stands in without a DataEngine) — the
    first keys the observation screenshot to its visual-transition summary,
    all of them feed the chunk ledger. ``actions`` collects the outcome of
    every device action so the turn's execution result can be rendered.
    """

    step_keys: list[str] = field(default_factory=list)
    actions: list[tuple[str, str, str]] = field(default_factory=list)

    def result(self) -> dict | None:
        """The turn's execution result in the validator-report shape.

        ``None`` unless an action failed: Flash executes synchronously and each
        action's tool message already carries its outcome, so a result message
        would only repeat it. A failure is worth its own message because it
        carries the error the turn ended on.
        """
        for name, status, text in self.actions:
            if status != "success":
                # Executor messages name the action themselves ("Error
                # executing click: ..."); prefix only the ones that don't.
                return {"status": "failed", "error": text if name in text else f"{name}: {text}"}
        return None


class FlashRunner:
    """Reactive, ultra-fast agent execution loop supporting all VLM providers."""

    def __init__(self, ctx: ArtemisContext, goal: str, max_turns: int | None = None):
        self.ctx = ctx
        self.goal = goal
        try:
            cfg = load_agent_config()
            self.max_turns = max_turns if max_turns is not None else cfg.flash.max_turns
            self.step_summarizer_cfg = cfg.flash.step_summarizer
            self.memory_runtime_cfg = cfg.memory.runtime
            self.transcript_cfg = cfg.memory.transcript
            self.chunking_cfg = cfg.memory.chunking
        except Exception:
            self.max_turns = max_turns if max_turns is not None else 0
            self.step_summarizer_cfg = StepSummarizerConfig()
            self.memory_runtime_cfg = MemoryRuntimeConfig()
            self.transcript_cfg = MemoryTranscriptConfig()
            self.chunking_cfg = None

        self.controller = UnifiedMobileController(ctx)
        # ``agent_name="flash"`` makes ask_explorer follow ``explorer.flash_mode``
        # (the Flash profile knob) instead of the Pro profile's tier.
        self.executor = McpActionExecutor(ctx, self.controller, agent_name="flash")
        self.summarizer = (
            VisualStepSummarizer(
                ctx,
                model_name=self.step_summarizer_cfg.model,
                retry_limit=self.memory_runtime_cfg.retry_limit,
                max_concurrency=self.memory_runtime_cfg.max_concurrency,
                flush_timeout_s=self.memory_runtime_cfg.flush_timeout_s,
            )
            if self.step_summarizer_cfg.enabled
            else None
        )
        # Publish the service on the composition root (ctx.step_memory) so
        # any co-resident consumer shares this run's summary runtime.
        if self.summarizer is not None and getattr(ctx, "step_memory", None) is None:
            try:
                ctx.step_memory = self.summarizer
            except (AttributeError, TypeError, ValueError):
                pass
        # Terminal-boundary verification state: the hybrid plan text (for the
        # final check's context) and how many 'completed' reports the checker
        # has already rejected this run.
        self._plan_text: str | None = None
        self._completion_rejections = 0
        # Declared success conditions (execution_setup.goal_invariants): when
        # every one evaluates satisfied against the live element list the run
        # may close early; an unsatisfied one vetoes 'completed' reports.
        self._goal_invariants: list[dict] = []
        # One-line "last action → outcome" fed into the decision model's
        # state: Laya has no transcript view, so without it a dead tap and
        # real progress look identical.
        self._last_action_note: str | None = None

    @property
    def turn_limit(self) -> int | None:
        """The reactive turn cap, or ``None`` when the loop is unbounded."""
        try:
            limit = int(self.max_turns) if self.max_turns is not None else 0
        except (TypeError, ValueError):
            return None
        return limit if limit > 0 else None

    def _video_tools_enabled(self) -> bool:
        """Same gate the Pro graph applies before binding ``video_analyzer``."""
        setup = getattr(self.ctx, "execution_setup", None)
        return getattr(setup, "video_recording_tools_enabled", False) is True

    def _get_tools(self) -> list:
        # Flash has no tool that writes notes and its prompt never teaches
        # them, so the Validator set's note readers are left out here.
        tools = [
            t
            for t in VALIDATOR_TOOLS_DECLARATION
            if t is not READ_NOTE_TOOL and t is not LIST_NOTES_TOOL
        ]
        tools.insert(1, CLICK_SEQUENCE_TOOL)
        tools.append(ASK_EXPLORER_TOOL)
        # Helper tools shared with the Pro operator: the history tools are
        # declared from the very same args schemas the LangChain tools bind
        # (same availability gates: a DataEngine session, recall config).
        tools.extend(history_tool_declarations(self.ctx))
        if self._video_tools_enabled():
            from artemis.tools.video_tool import VIDEO_ANALYZER_TOOL

            tools.append(VIDEO_ANALYZER_TOOL)
        tools.append(REPORT_TASK_STATUS_TOOL)
        # With an actuator installed, drop declarations for device actions the
        # backend does not implement (and append its extension tools); without
        # one, the full historical declaration set is kept.
        actuator = getattr(self.ctx, "actuator", None)
        if actuator is not None:
            from artemis.mcp.action_manifest import filter_declarations

            tools = filter_declarations(tools, actuator, "flash")
        return tools

    def _prune_intermediate_screenshots(self, messages: list[BaseMessage]) -> None:
        """Prunes binary screenshot blocks from intermediate observation messages."""
        prune_intermediate_screenshots(messages)

    # ------------------------------------------------------------------
    # run() setup helpers
    # ------------------------------------------------------------------

    def _build_ledger(self) -> TranscriptLedger:
        """Builds the session transcript ledger with the shared Pro history policy.

        The ``T+mm:ss`` clock is anchored to the DataEngine session start so
        the observation headers, the chunk ledger lines and the video
        analyzer's action timeline share one origin. L2/L3 chunk compression
        needs DataEngine step records; without an engine the ledger runs
        scrub-edge-only.
        """
        engine = getattr(self.ctx, "data_engine", None)
        session_start = getattr(engine, "session_start_time", None) if engine else None
        cfg = self.transcript_cfg
        ledger = TranscriptLedger(
            step_memory=self.summarizer,
            prune_history_xml=self.step_summarizer_cfg.prune_history_xml,
            image_scrub_depth=getattr(cfg, "image_scrub_depth", 3),
            pending_grace_steps=getattr(cfg, "pending_grace_steps", 3),
            xml_scrub_depth=getattr(cfg, "xml_scrub_depth", 1),
            image_scrub_depth_relaxed=getattr(cfg, "image_scrub_depth_relaxed", None),
            context_budget_tokens=getattr(cfg, "context_budget_tokens", None),
            start_ratio=getattr(cfg, "start_ratio", None),
            session_start=session_start if isinstance(session_start, (int, float)) else None,
        )
        if engine is not None:
            try:
                from artemis.memory import HistoryChunkManager

                ledger.attach_chunker(
                    HistoryChunkManager(
                        engine=engine,
                        ctx=self.ctx,
                        chunking_config=self.chunking_cfg,
                        transcript_config=cfg,
                        goal=self.goal,
                    )
                )
            except Exception as e:
                logger.error(f"History chunk manager unavailable for FlashRunner: {e}")
        try:
            self.ctx.transcript_ledger = ledger
        except (AttributeError, TypeError, ValueError):
            pass
        return ledger

    #: The model of the current run; decides how screenshot tool results travel.
    _llm = None

    def _init_llm(self):
        """Initializes the Universal LLM via the Service Layer."""
        try:
            return get_llm(self.ctx, name="operator")
        except Exception as e:
            logger.warning(f"Failed to get operator LLM from config, using default: {e}")

            return RobustChatModelWrapper(get_google_llm(model_name="gemini-2.5-flash"), self.ctx)

    # ------------------------------------------------------------------
    # Hybrid loop (decision.flash_loop): VLM describes, decision model acts
    # ------------------------------------------------------------------

    async def _hybrid_plan(self, llm, img_bytes=None, xml_list=None) -> str | None:
        """The one upfront planning call — Pro's planner leg grafted into Flash.

        A plain completion (no tools): the VLM emits a milestone list that the
        per-turn describe prompt carries as context. The call is grounded on
        the first observation — a plan made blind degenerates to boilerplate
        ("load login screen" for a Settings task, observed live). Failure is
        non-fatal — the loop still runs, just without plan guidance.
        """
        prompt = Template(_HYBRID_PLAN_PROMPT).render(goal=self.goal)
        blocks: list[dict] = [{"type": "text", "text": prompt}]
        if img_bytes:
            blocks.append({"type": "text", "text": "--- Current Screenshot ---"})
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri(base64.b64encode(img_bytes).decode())},
                }
            )
        if xml_list:
            blocks.append({"type": "text", "text": f"{PRO_UI_LIST_MARKER}\n{xml_list}"})
        try:
            response = await invoke_llm_with_timeout_message(
                acomplete(llm, [HumanMessage(content=blocks)]),
                timeout_seconds=10,
                hard_timeout=settings.LLM_FLASH_HARD_TIMEOUT_SECONDS,
            )
        except LLMCallError as e:
            logger.warning(f"Hybrid plan call failed, running unplanned: {e}")
            return None
        plan = self._extract_response_text(response).strip()
        logger.info(f"Hybrid plan ({len(plan)} chars): {plan[:200]}")
        return plan or None

    @staticmethod
    def _element_label(el: dict) -> str:
        return str(
            el.get("text")
            or el.get("content_desc")
            or el.get("resource_id")
            or el.get("class_name")
            or "element"
        )[:60]

    @staticmethod
    def _element_signature(elements) -> frozenset:
        """(label, center, state) set used to tell whether the screen changed.

        ``checked``/``enabled`` join the identity so a toggle flipping
        ON→OFF counts as a screen change — with geometry alone the dead-tap
        machinery treated a working switch tap as a no-op (observed live:
        the Dark theme switch flipped while the signature stayed identical).
        Volatile attributes (``focused``/``selected``) are excluded: focus
        moves on scroll/hover and would fake 'changed' on dead taps.
        Inherently volatile elements are excluded entirely — status-bar
        items (``com.android.systemui``: clock, badges) tick on their own
        and OCR text jitters between dumps — so a strict set comparison
        can mean 'unchanged' instead of a similarity ratio that swallows
        small real changes on dense screens (observed live: one changed
        digit label on the 30-node dial pad read 'dead' at Jaccard 0.8).
        """
        return frozenset(
            (
                FlashRunner._element_label(el),
                tuple(el.get("center") or ()),
                el.get("checked"),
                el.get("enabled"),
            )
            for el in elements
            if not el.get("is_ocr")
            and el.get("package") != "com.android.systemui"
        )

    @staticmethod
    def _signatures_similar(a: frozenset, b: frozenset, threshold: float = 0.8) -> bool:
        """Jaccard-similar element signatures — tolerates volatile items like
        the clock, so a screen is 'unchanged' only when it really didn't move."""
        union = a | b
        if not union:
            return True
        return len(a & b) / len(union) >= threshold

    @staticmethod
    def _hybrid_action_menu(
        state: State,
        tried_taps: set[tuple[int, str]] | None = None,
        tried_labels: set[str] | None = None,
    ) -> dict[str, str]:
        """The fixed action menu for one turn: element taps + meta-actions.

        Tap keys embed the element index of the CURRENT observation's list —
        index drift across turns is impossible because the menu is rebuilt
        every turn and the executor resolves the index against the same list
        snapshot the decision was made on.

        ``tried_taps`` masks elements already tapped by (index, label)
        identity; ``tried_labels`` additionally masks any element whose
        distinctive label was tapped before, surviving re-indexing. A screen
        signature can't express "same screen" here — volatile text (clocks,
        badges) perturbs it — so identity matching is used instead.
        """
        elements = getattr(state, "indexed_elements", None) or []
        menu: dict[str, str] = {}
        for el in elements[:_HYBRID_MAX_MENU_ELEMENTS]:
            idx = el.get("index")
            if idx is None:
                continue
            label = FlashRunner._element_label(el)
            if tried_taps and (idx, label) in tried_taps:
                continue
            if tried_labels and len(label) >= 4 and label in tried_labels:
                continue
            menu[f"tap_{idx}"] = f"tap element {idx}: {label}"
        menu.update(_HYBRID_META_ACTIONS)
        return menu

    @staticmethod
    def _menu_choice_to_call(choice: str, description: str) -> dict | None:
        """Maps a menu key to a tool-call dict for the normal execution path."""
        tid = f"call_{uuid.uuid4().hex[:8]}"
        if choice.startswith("tap_"):
            try:
                idx = int(choice[len("tap_"):])
            except ValueError:
                return None
            return {"id": tid, "name": "click", "args": {"target": idx}}
        if choice in ("swipe_up", "swipe_down", "swipe_left", "swipe_right"):
            return {
                "id": tid,
                "name": "swipe",
                "args": {"direction": choice.split("_", 1)[1]},
            }
        if choice in ("press_back", "press_home"):
            return {
                "id": tid,
                "name": "press_key",
                "args": {"key": "back" if choice == "press_back" else "home"},
            }
        if choice == "wait":
            return {"id": tid, "name": "wait_for_delay", "args": {"timeout_ms": 1500}}
        if choice in ("done", "fail"):
            return {
                "id": tid,
                "name": "report_task_status",
                "args": {
                    "status": "completed" if choice == "done" else "failed",
                    "explanation": description[:500] or "Decision layer verdict.",
                },
            }
        return None

    _NAV_LABEL_RE = re.compile(
        r"navigate up|back|close|cancel|done|collapse|dismiss|arrow", re.I
    )

    def _navigation_suggestions(self, elements) -> str:
        """Labeled elements that look like exits/nav — offered as concrete
        alternatives when the model is re-issuing a dead action."""
        hints = []
        for el in elements or []:
            idx = el.get("index")
            label = self._element_label(el)
            if idx is None or not label:
                continue
            if self._NAV_LABEL_RE.search(label):
                hints.append(f"[{idx}] {label}")
        return ", ".join(hints[:6])

    _GOAL_STOPWORDS = frozenset(
        {
            "the", "and", "for", "with", "app", "open", "into", "enable",
            "disable", "turn", "tap", "click", "press", "then", "this",
            "that", "screen", "page", "goto", "select",
        }
    )

    @classmethod
    def _menu_has_goal_target(cls, menu: dict[str, str], goal: str) -> bool:
        """True when some tap label shares a meaningful token with the goal."""
        goal_tokens = {
            t
            for t in re.findall(r"[a-z0-9']+", goal.lower())
            if len(t) >= 4 and t not in cls._GOAL_STOPWORDS
        }
        if not goal_tokens:
            return True
        for key, label in menu.items():
            if not key.startswith("tap_"):
                continue
            label_tokens = set(re.findall(r"[a-z0-9']+", label.lower()))
            if label_tokens & goal_tokens:
                return True
        return False

    @staticmethod
    def _menu_key_for_call(name: str, args: dict) -> str | None:
        """Maps an executed tool call back to its hybrid menu key, so actions
        taken on a screen can be masked from the menu on a later visit."""
        args = args or {}
        if name == "click":
            target = args.get("target")
            return f"tap_{target}" if isinstance(target, int) else None
        if name == "swipe":
            direction = str(args.get("direction") or args.get("gesture") or "").lower()
            return f"swipe_{direction}" if direction in ("up", "down", "left", "right") else None
        if name == "press_key":
            return {"back": "press_back", "home": "press_home"}.get(
                str(args.get("key") or "").lower()
            )
        if name == "wait_for_delay":
            return "wait"
        if name == "report_task_status":
            return "done" if args.get("status") == "completed" else "fail"
        return None

    async def _hybrid_decide(
        self,
        llm,
        state: State,
        tail: HumanMessage,
        plan_text: str | None,
        cfg,
        tried_taps: set[tuple[int, str]] | None = None,
        tried_labels: set[str] | None = None,
    ) -> tuple | None:
        """One hybrid turn: describe → choose → tool call.

        Returns ``(describe_response, [tool_call])`` on a confident decision,
        ``None`` to escalate to the standard tool-calling model turn. The
        describe call binds no tools — the VLM answers with a screen
        narrative (plus approximate coords for unlisted targets) that becomes
        the turn's thought text, so envelope-mangling and narration-only
        stalls cannot occur on this path.

        ``tried_taps``/``tried_labels`` mask menu entries already attempted —
        the decision model has no transcript view and would otherwise re-pick
        a dead element indefinitely. Terminal meta-actions
        (``wait``/``done``/``fail``) are never masked.
        """
        describe_prompt = Template(_HYBRID_DESCRIBE_PROMPT).render(
            goal=self.goal, plan=plan_text or "(no plan — describe and list elements)"
        )
        try:
            response = await invoke_llm_with_timeout_message(
                acomplete(llm, [SystemMessage(content=describe_prompt), tail]),
                timeout_seconds=10,
                hard_timeout=settings.LLM_FLASH_HARD_TIMEOUT_SECONDS,
            )
        except LLMCallError as e:
            logger.warning(f"Hybrid describe call failed, escalating: {e}")
            return None
        description = self._extract_response_text(response)
        logger.info(f"Hybrid describe: {description[:200]}")

        # Goal-achieved marker: a 'reached' describe is a perception claim,
        # not a menu pick — hand the turn to the VLM, which can look at the
        # screen itself and call report_task_status. The terminal-boundary
        # check then audits that claim against device state, so a premature
        # 'reached' costs a verification round-trip, not a false completion.
        if re.search(r"(?im)^\s*target:\s*reached\b", description):
            logger.info(
                "Hybrid describe reports TARGET: reached — escalating to the "
                "VLM turn for verification"
            )
            return None

        menu = self._hybrid_action_menu(
            state, tried_taps=tried_taps, tried_labels=tried_labels
        )
        # Visibility gate: the goal's next target must exist among the menu's
        # element labels for an in-screen pick to be meaningful at all. Token
        # overlap between labels and the goal is the check — describe
        # narration can't be trusted (observed: 'TARGET: visible' on a home
        # screen listing no Settings). No match → the VLM handles the
        # between-screen navigation turn instead.
        if not self._menu_has_goal_target(menu, self.goal):
            tap_labels = [v for k, v in menu.items() if k.startswith("tap_")]
            logger.info(
                "Hybrid menu holds no goal-relevant element — escalating "
                f"the navigation turn to the VLM path (labels: {tap_labels[:12]})"
            )
            return None
        tried_line = ""
        if tried_taps or tried_labels:
            masked = sorted(
                set(self._hybrid_action_menu(state)) - set(menu)
            )
            if masked:
                tried_line = (
                    "\nAlready attempted without moving the task forward and"
                    f" unavailable: {', '.join(masked)}."
                )
        client = DecisionClient(
            cfg.url, model=cfg.model, timeout_seconds=cfg.timeout_seconds
        )
        last_line = (
            f"\nLast action outcome: {self._last_action_note}"
            if self._last_action_note
            else ""
        )
        state_text = (
            f"Goal: {self.goal}\n"
            f"Plan: {plan_text or 'none'}\n"
            f"Current screen: {description}{tried_line}{last_line}"
        )
        verdict = await client.achoose(
            state_text[-_HYBRID_MAX_STATE_CHARS:],
            f"Which single action best advances this goal: {self.goal}",
            menu,
        )
        # For `choice` answers the decision gate is the top-pick probability
        # (answer_confidence): the calibrated `confidence` field measures overall
        # answer trustworthiness and sits near zero on any menu-shaped spread —
        # using it would escalate every turn.
        top_p = verdict.confidence
        raw = getattr(verdict, "raw", None) or {}
        try:
            top_p = float(raw.get("answer_confidence") or top_p)
        except (TypeError, ValueError):
            pass
        floor = getattr(cfg, "flash_choice_threshold", 0.25)
        # Scale the floor to the menu size: on a 20-option menu a uniform
        # spread is 0.05, so a 0.20 top pick is 4x uniform — decisive — yet a
        # fixed 0.25 gate escalates every real-screen verdict (observed: all
        # post-navigation picks landed 0.16–0.23). Require 4x uniform, capped
        # at the configured floor, with an absolute minimum of 0.12.
        effective_floor = max(0.12, min(floor, 4.0 / max(len(menu), 1)))
        if (
            not verdict.answered
            or top_p < effective_floor
            or verdict.verdict not in menu
        ):
            logger.info(
                f"Hybrid decision inconclusive (verdict={verdict.verdict!r} "
                f"top_p={top_p:.2f} floor={effective_floor:.2f}), escalating to tool-call turn"
            )
            return None
        if verdict.verdict in ("done", "fail"):
            # Terminal picks can never execute blind: the decision model sees
            # no pixels and cannot verify goal completion — observed live
            # declaring 'done' on a screen where the target app wasn't even
            # open. The standard turn re-verifies against the screenshot and
            # keeps report_task_status available to do it honestly.
            logger.info(
                f"Hybrid terminal pick {verdict.verdict!r} (top_p={top_p:.2f}) "
                "— escalating to the VLM turn for verification"
            )
            return None
        tool_call = self._menu_choice_to_call(str(verdict.verdict), description)
        if tool_call is None:
            return None
        logger.info(
            f"Hybrid decision: {tool_call['name']} (top_p={top_p:.2f})"
        )
        # Stamp the chosen call onto the describe message: the ToolMessages the
        # executor appends then have a legitimate parent tool_call, which strict
        # providers require when the ledger replays them into later prompts.
        response.tool_calls = [
            {
                "name": tool_call["name"],
                "args": tool_call["args"],
                "id": tool_call["id"],
                "type": "tool_call",
            }
        ]
        response.invalid_tool_calls = []
        return response, [tool_call]

    def _render_system_prompt(self, tools_declaration: list) -> str:
        """Renders the system prompt from the flash_runner.md template.

        Tool-teaching segments are gated on the available tool set so an absent
        tool leaves no trace in the prompt; until an actuator is wired in, the
        full manifest set reproduces the historical prompt.
        """
        prompt_path = Path(__file__).parent / "flash_runner.md"
        prompt_template = prompt_path.read_text(encoding="utf-8")
        available_tools = frozenset(t.name for t in tools_declaration)
        return Template(prompt_template).render(goal=self.goal, available_tools=available_tools)

    # ------------------------------------------------------------------
    # Per-turn helpers (observe / think)
    # ------------------------------------------------------------------

    async def _read_injected_instruction(
        self, offset_label: str | None = None
    ) -> tuple[str | None, UserGuidance | None]:
        """Consumes this turn's injected instruction: ``(instruction, guidance)``.

        ``instruction`` is the user's text verbatim — what the step record,
        the chunk ledger and the visual-transition lens receive (the same
        shape Pro stamps). ``guidance`` is the shared user-guidance pair
        (:func:`render_user_guidance`, the Pro rendering without a plan to
        edit): the ephemeral ``--- User Guidance ---`` wrapper and the
        persistent verbatim body, both riding on the observation tail.
        ``offset_label`` stamps the body with the session offset. Both are
        None when nothing was injected.
        """
        if not (self.ctx.data_engine and self.ctx.data_engine.base_dir):
            return None, None
        try:
            injected_payload = await asyncio.to_thread(
                _check_injected_instruction_file,
                str(self.ctx.data_engine.base_dir),
            )
        except (OSError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to check injected instruction in FlashRunner: {e}")
            return None, None
        if not (injected_payload and injected_payload.get("instruction")):
            return None, None
        instruction = str(injected_payload["instruction"])
        guidance = render_user_guidance(
            instruction,
            release_loop=bool(injected_payload.get("release_loop")),
            has_plan=False,
            offset_label=offset_label,
        )
        return instruction, guidance

    def _build_tail(
        self,
        ledger: TranscriptLedger,
        turns: int,
        img_bytes,
        xml_list,
        *,
        injected: UserGuidance | None = None,
        notices: list[str] | None = None,
        is_final: bool = False,
        previous_turn_silent: bool = False,
    ) -> HumanMessage:
        """Builds this turn's observation tail (Pro observation shape).

        Header, screenshot and UI list carry the same markers as the Pro
        operator's tail so the ledger's scrub edge treats both alike. The
        objective is stated once, in the system prompt; the tail never repeats
        it. Every per-turn notice (nudges, the user-guidance wrapper, the
        final-turn warning, the reasoning reminder after a silent turn) is
        marked ephemeral so it leaves the transcript with the recent window.
        The verbatim body of an injected instruction is the one exception: it
        is a regular block, so a standing instruction stays in the active
        window until the turn is chunked (the chunk's user lines carry it on).
        """
        # ``turns`` is kept for call-site symmetry: the tail has the same shape
        # on every turn.
        blocks: list[dict] = [
            {"type": "text", "text": f"# CURRENT OBSERVATION [{ledger.elapsed_label()}]"}
        ]
        if xml_list:
            blocks.append({"type": "text", "text": f"{PRO_UI_LIST_MARKER}\n{xml_list}"})
        ephemeral: list[int] = []
        # (text, ephemeral) in tail order.
        per_turn: list[tuple[str, bool]] = [(text, True) for text in (notices or [])]
        if injected:
            # The wrapper is per-turn; the verbatim body right after it is a
            # regular block that stays until the turn is chunked.
            per_turn.append((injected.wrapper, True))
            per_turn.append((injected.body, False))
        if is_final:
            per_turn.append((_FINAL_TURN_WARNING, True))
        if previous_turn_silent:
            per_turn.append((REASONING_REMINDER, True))
        for text, is_ephemeral in per_turn:
            if is_ephemeral:
                ephemeral.append(len(blocks))
            blocks.append({"type": "text", "text": text})
        # The screenshot goes last so intermediate-image pruning removes a
        # trailing span: the surviving text prefix stays byte-identical across
        # turns, which prefix-caching servers (APC) can then reuse.
        if img_bytes:
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            blocks.append({"type": "text", "text": "--- Current Screenshot ---"})
            blocks.append({"type": "image_url", "image_url": {"url": image_data_uri(img_b64)}})
        tail = HumanMessage(content=blocks)
        if ephemeral:
            mark_ephemeral(tail, ephemeral)
        return tail

    def _extract_response_text(self, response) -> str:
        """Extracts the natural-language thought text from the model response."""
        raw_text = response.content if isinstance(response.content, str) else ""
        if isinstance(response.content, list):
            raw_text = "".join(
                b.get("text", "") for b in response.content if isinstance(b, dict) and "text" in b
            )
        return raw_text

    def _extract_response_thinking(self, response) -> str:
        """Extracts the model's native thinking blocks (thought summaries), if any.

        Providers running with ``include_thoughts`` return them as ``thinking``
        content blocks next to the ``text`` blocks; they are recorded on the step
        as ``operator_native_thinking`` and never fed back as reply text.
        """
        if not isinstance(response.content, list):
            return ""
        parts = [
            str(b.get("thinking") or "")
            for b in response.content
            if isinstance(b, dict) and b.get("type") == "thinking"
        ]
        return "\n".join(p for p in parts if p.strip())

    def _token_usage_from_response(self, response) -> dict | None:
        """Extracts token usage metadata from the model response, if present."""
        step_token_usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            u = response.usage_metadata
            pr = u.get("input_tokens") or u.get("prompt_tokens") or 0
            co = u.get("output_tokens") or u.get("completion_tokens") or 0
            to = u.get("total_tokens") or (pr + co)
            if to > 0:
                step_token_usage = {
                    "prompt_tokens": int(pr),
                    "completion_tokens": int(co),
                    "total_tokens": int(to),
                }
        elif hasattr(response, "response_metadata") and isinstance(
            response.response_metadata, dict
        ):
            u = response.response_metadata.get("usage_metadata") or response.response_metadata.get(
                "token_usage"
            )
            if isinstance(u, dict):
                pr = (
                    u.get("input_tokens")
                    or u.get("prompt_tokens")
                    or u.get("prompt_token_count")
                    or 0
                )
                co = (
                    u.get("output_tokens")
                    or u.get("completion_tokens")
                    or u.get("candidates_token_count")
                    or 0
                )
                to = u.get("total_tokens") or u.get("total_token_count") or (pr + co)
                if to > 0:
                    step_token_usage = {
                        "prompt_tokens": int(pr),
                        "completion_tokens": int(co),
                        "total_tokens": int(to),
                    }
        return step_token_usage

    def _estimate_token_usage(self, messages: list[BaseMessage], raw_text: str) -> dict:
        """Approximates token usage from message sizes when metadata is absent."""
        calc_prompt_chars = 0
        calc_images = 0
        for msg in messages:
            c = getattr(msg, "content", "")
            if isinstance(c, str):
                calc_prompt_chars += len(c)
            elif isinstance(c, list):
                for block in c:
                    if isinstance(block, dict):
                        if block.get("type") == "image_url" or "image_url" in block:
                            calc_images += 1
                        else:
                            calc_prompt_chars += len(str(block.get("text", "")))
                    else:
                        calc_prompt_chars += len(str(block))
        prompt_tokens = (calc_prompt_chars // 4) + (calc_images * 258)
        completion_tokens = len(raw_text) // 4
        return {
            "prompt_tokens": max(1, prompt_tokens),
            "completion_tokens": max(1, completion_tokens),
            "total_tokens": max(1, prompt_tokens + completion_tokens),
        }

    def _resolve_token_usage(
        self, response, messages: list[BaseMessage], raw_text: str
    ) -> tuple[dict, bool]:
        """Token usage for the call and whether the provider measured it.

        Falls back to an estimate when the response carries no usage
        metadata; the flag lets the caller keep an estimate out of the
        ledger's chars-per-token calibration (an estimate is itself derived
        from character counts, so calibrating on it would be circular).
        """
        step_token_usage = self._token_usage_from_response(response)
        if step_token_usage and step_token_usage.get("total_tokens", 0) > 0:
            return step_token_usage, True
        return self._estimate_token_usage(messages, raw_text), False

    def _record_llm_trace(self, step_token_usage: dict, raw_text: str) -> None:
        """Records the llm_call trace for this turn in the DataEngine."""
        if self.ctx.data_engine:
            current_step_id = getattr(self.ctx.data_engine, "current_step_id", None)
            self.ctx.data_engine.record_trace(
                type="llm_call",
                name="FlashRunner",
                payload={"token_usage": step_token_usage, "response": raw_text},
                step_id=current_step_id,
                status="success",
            )

    def _resolve_tool_calls(self, response, raw_text: str) -> list:
        """Returns native tool calls, falling back to text parsing if absent."""
        tool_calls = response.tool_calls or []

        # Fallback text parsing if tool_calls not parsed natively
        if not tool_calls and "```json" in raw_text:
            parsed = parse_structured(raw_text)
            if isinstance(parsed, ParseFailure):
                logger.warning(
                    "FlashRunner response contained a JSON block that could"
                    f" not be parsed: {parsed.error}"
                )
            elif isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [
                    {
                        "name": parsed["name"],
                        "args": parsed.get("args", {}),
                        "id": str(uuid.uuid4()),
                    }
                ]

        # Local-server envelope repair: mlx_vlm can wrap the model's native
        # `call:name{args}` output into a function literally named "call" whose
        # arguments are the unparseable remainder — langchain then drops it to
        # invalid_tool_calls and the call is silently lost.
        if not tool_calls:
            tool_calls = self._repair_enveloped_tool_calls(
                getattr(response, "invalid_tool_calls", None)
            )

        # Same envelope can leak into text content outright.
        if not tool_calls and raw_text:
            call_match = _GEMMA_CALL_RE.search(raw_text)
            if call_match:
                repaired = _parse_call_syntax(raw_text, match=call_match)
                if repaired:
                    tool_calls = [repaired]
        return tool_calls

    @staticmethod
    def _repair_enveloped_tool_calls(invalid_tool_calls) -> list:
        """Recovers calls mangled into `name="call"` + raw `name{args}` args."""
        repaired = []
        for tc in invalid_tool_calls or []:
            name = tc.get("name")
            raw_args = tc.get("args")
            if name != "call" or not isinstance(raw_args, str):
                continue
            parsed = _parse_call_syntax(raw_args)
            if parsed:
                parsed["id"] = tc.get("id") or parsed["id"]
                repaired.append(parsed)
        return repaired

    # ------------------------------------------------------------------
    # Per-tool-call helpers (act / record / report)
    # ------------------------------------------------------------------

    async def _finalize_task_report(
        self,
        name: str,
        args: dict,
        tc_id: str,
        raw_text: str,
        step_token_usage: dict,
        pre_screenshot_bytes,
        xml_list,
        messages: list[BaseMessage],
        native_text: str | None = None,
    ) -> dict:
        """Records the final step, acknowledges the tool call, and flushes."""
        final_report = args
        self._record_terminal_step(
            args,
            raw_text,
            step_token_usage,
            pre_screenshot_bytes,
            xml_list,
            last_execution_result={"result": "Task completed with final report."},
            native_text=native_text,
        )
        messages.append(
            ToolMessage(
                tool_call_id=tc_id,
                name=name,
                content=json.dumps({"status": "acknowledged"}),
                status="success",
            )
        )
        if self.summarizer:
            await self.summarizer.flush()
        return final_report

    def _record_terminal_step(
        self,
        args: dict,
        raw_text: str,
        step_token_usage: dict,
        pre_screenshot_bytes,
        xml_list,
        last_execution_result: dict,
        native_text: str | None = None,
    ) -> None:
        """Writes a report_task_status step to the data engine (accepted or
        rejected), mirroring the final-step record shape."""
        if not self.ctx.data_engine:
            return
        try:
            if self.ctx.data_engine.current_step_id is None:
                self.ctx.data_engine.allocate_step_id()
            self.ctx.data_engine.record_step(
                pre_screenshot_bytes=pre_screenshot_bytes,
                ui_tree=xml_list,
                action_taken={"action": "report_task_status", "args": args},
                operator_raw_thinking=raw_text,
                operator_native_thinking=native_text or None,
                last_execution_result=last_execution_result,
                extra_metadata={"token_usage": step_token_usage},
            )
        except Exception as step_err:
            logger.warning(f"Error recording terminal step in FlashRunner: {step_err}")

    async def _verify_completion(self, state: State) -> tuple[bool, str]:
        """Terminal-boundary check on a 'completed' report_task_status claim.

        Declared ``goal_invariants`` run first and are authoritative:
        ``unsatisfied`` is a deterministic contradiction and rejects the
        claim regardless of checker gates; all-``satisfied`` releases it
        (the invariants are the declared definition of done); ``unknown``
        falls through to the model final check — or fail-open when that
        check is disabled.

        The final review — final screenshot, element list, and the
        enumerated read-only device probes — then judges the goal from
        device state rather than the agent's narration (observed live: a
        double tap netted zero on the dark-theme switch, then the model
        reported completed). Fail-open on checker errors: a broken verifier
        must never trap the loop.

        Returns ``(allowed, findings)``; ``findings`` lists failed verify
        verdicts and unmet subgoals for feedback on rejection.
        """
        if self._goal_invariants:
            elements = await self._reobserve_elements(state)
            report = evaluate_invariants(self._goal_invariants, elements)
            if report["verdict"] == "unsatisfied":
                logger.info("Completion rejected: declared invariant unsatisfied.")
                return False, describe_failures(report)
            if report["verdict"] == "satisfied":
                logger.info("Completion released: declared invariants satisfied.")
                return True, ""
            # unknown: insufficient evidence — defer to the model check.

        setup = getattr(self.ctx, "execution_setup", None)
        if not (setup and getattr(setup, "final_check_enabled", False)):
            return True, ""
        try:
            from artemis.agents.checker.checker import (
                run_final_check,
                verdicts_allow_release,
            )

            timeout = float(getattr(setup, "checkpoint_timeout", 180.0) or 180.0)
            report = await asyncio.wait_for(
                run_final_check(
                    self.ctx,
                    goal=self.goal,
                    plan_text=self._plan_text or "",
                    ledger=[],
                    check_items=[],
                ),
                timeout=timeout,
            )
        except Exception as e:
            logger.warning(f"Final check errored ({e}); releasing fail-open.")
            return True, ""
        if verdicts_allow_release(report) and not report.unmet_subgoals:
            logger.info("Final check released the completion report.")
            return True, ""
        findings = [
            f"[verify failed] '{v.item_text}': {v.evidence}"
            + (f" Suggestion: {v.suggestion}" if v.suggestion else "")
            for v in report.verdicts
            if v.kind == "verify" and v.status == "failed"
        ]
        findings += [f"[unmet subgoal] '{t}'" for t in report.unmet_subgoals]
        return False, "\n".join(findings) or "Final check did not confirm completion."

    async def _capture_post_screenshot(
        self, exec_result, name: str, action_names, state: State | None = None
    ):
        """Returns the post-action screenshot, capturing a fallback if needed.

        When a device action produced no observation (a failed action, an
        unknown tool, ...) the screen is observed through the very same
        pipeline the executor's success path uses (:func:`observe`), so the
        next observation tail carries the indexed minimal UI list and
        ``state`` gets the same refreshed screenshot path and indexed
        elements. Nothing is captured for non-action tools.
        """
        post_img_bytes = exec_result.screenshot_bytes
        if post_img_bytes or name not in action_names:
            return post_img_bytes
        try:
            obs, screenshot_bytes = await observe(self.ctx, self.controller, settle_ms=0)
        except Exception as shot_err:
            logger.warning(f"Failed to capture fallback screenshot in FlashRunner: {shot_err}")
            return post_img_bytes
        if not obs.ok:
            logger.warning(f"Failed to capture fallback screenshot in FlashRunner: {obs.message}")
            return post_img_bytes
        post_img_bytes = screenshot_bytes
        if not exec_result.ui_elements_text:
            exec_result.ui_elements_text = obs.elements_text
        if not exec_result.screenshot_path and obs.screenshot_path:
            exec_result.screenshot_path = obs.screenshot_path
        if state is not None:
            if obs.hierarchy_ok:
                state.indexed_points = [el["center"] for el in obs.elements]
                state.indexed_elements = obs.elements
            if obs.screenshot_path:
                state.latest_screenshot = obs.screenshot_path
        return post_img_bytes

    async def _refresh_idle_observation(
        self, state: State | None, img_bytes, xml_list
    ):
        """Re-observes the screen after a turn that executed no device action.

        Turns that end without a tool call never reach the post-action
        capture in ``_process_tool_calls``; without this the next turn's tail
        (and the hybrid decision menu built from ``state.indexed_elements``)
        would describe a stale frame — e.g. one captured mid-transition.
        """
        try:
            obs, screenshot_bytes = await observe(self.ctx, self.controller, settle_ms=0)
        except Exception as obs_err:
            logger.warning(f"Idle-turn re-observation failed: {obs_err}")
            return img_bytes, xml_list
        if not obs.ok:
            return img_bytes, xml_list
        if state is not None:
            if obs.hierarchy_ok:
                state.indexed_points = [el["center"] for el in obs.elements]
                state.indexed_elements = obs.elements
            if obs.screenshot_path:
                state.latest_screenshot = obs.screenshot_path
        if obs.elements_text:
            xml_list = obs.elements_text
        return screenshot_bytes or img_bytes, xml_list

    @staticmethod
    def _screens_differ(before: bytes | None, after: bytes | None) -> bool:
        """Pixel-level 'did the screen change' oracle for stale-tree detection.

        A 16x16 dHash comparison: deterministic screencap encoding means a
        same-screen pair scores 0 bits, while even a single appended digit
        scored ~9 bits on the emulator (measured) — a threshold of 3 keeps
        real small changes while staying far above capture noise.
        """
        if not before or not after:
            return False
        dist = hamming_distance_hex(
            dhash_hex(before, hash_size=16), dhash_hex(after, hash_size=16)
        )
        return dist is not None and dist >= 3

    async def _reobserve_elements(self, state: State | None) -> list:
        """Re-observes the screen when a decision needs the settled truth.

        Post-action captures race UI updates — a theme toggle applies
        after the ~0.4 s capture delay, and the accessibility helper's
        event-driven tree can trail a text commit too — so
        ``state.indexed_elements`` can hold the pre-action state for a
        whole turn (observed live: four allowed taps ping-ponged the
        dark-theme switch on lagged ``checked``; a working dial-pad tap
        read 'dead' on a lagged dump). Verification decisions — suppress
        an action, release a completion, call a tap dead — need the
        settled state, and a fresh dump is cheap next to the model call
        that just ran. Falls back to the last captured elements on
        failure.
        """
        elements = getattr(state, "indexed_elements", None) or []
        try:
            obs, _ = await observe(self.ctx, self.controller)
        except Exception as obs_err:
            logger.warning(f"Re-observation failed: {obs_err}")
            return elements
        if obs.ok and obs.hierarchy_ok:
            elements = obs.elements
            if state is not None:
                state.indexed_points = [el["center"] for el in elements]
                state.indexed_elements = elements
        return elements

    def _extract_normalized_coordinates(self, name: str, args: dict, resolved=None):
        """Extracts and enriches coordinate metadata for the recorded action.

        ``resolved`` is the normalized point the executor sent a point action
        to; it is the coordinate of record when the model addressed an element
        by index (the arguments then carry no point at all).
        """
        norm_coords = None
        norm_start = None
        norm_end = None
        if name in ("click", "tap", "long_press", "input_text") and (
            isinstance(resolved, (list, tuple)) and len(resolved) == 2
        ):
            return [int(resolved[0]), int(resolved[1])], None, None
        if name == "swipe":
            kind, target_val, _ = parse_swipe_parameters(args)
            if kind == "coords" and isinstance(target_val, list) and len(target_val) == 4:
                norm_coords = target_val
                norm_start = target_val[:2]
                norm_end = target_val[2:]
            elif kind == "direction" and isinstance(target_val, str):
                g_lower = target_val.lower()
                if "up" in g_lower:
                    norm_coords = [600, 700, 600, 300]
                elif "down" in g_lower:
                    norm_coords = [600, 300, 600, 700]
                elif "left" in g_lower:
                    norm_coords = [750, 500, 250, 500]
                elif "right" in g_lower:
                    norm_coords = [250, 500, 750, 500]
                if norm_coords:
                    norm_start = norm_coords[:2]
                    norm_end = norm_coords[2:]
        elif name in ("click", "tap", "long_press", "input_text"):
            target = args.get("target") or args.get("coordinates")
            if isinstance(target, (list, tuple)) and len(target) == 2:
                norm_coords = list(target)
            elif isinstance(target, str):
                nums = re.findall(r"-?\d+(?:\.\d+)?", target)
                if len(nums) == 2:
                    norm_coords = [int(float(nums[0])), int(float(nums[1]))]
        return norm_coords, norm_start, norm_end

    def _build_action_record(self, name: str, args: dict, exec_result) -> dict:
        """The action as the step record and the visual lens see it.

        Flash records the model's own 0–1000 coordinates verbatim (an index
        target records the point the executor resolved it to); the explicit
        space marker keeps every later normalization pass (agent-friendly
        steps, MCP inspector) a no-op on this record. The tool arguments ride
        along under ``args``, and the target semantics the executor validated
        (observed element fields for an index, the model's own
        ``target_description`` for a point) land top level in the shape Pro
        records.
        """
        metadata = getattr(exec_result, "metadata", None) or {}
        resolved = metadata.get("target_coordinates")
        norm_coords, norm_start, norm_end = self._extract_normalized_coordinates(
            name, args, resolved=resolved
        )
        given = args.get("target") or args.get("coordinates") or args.get("sequence")
        if isinstance(given, (int, float)) and not isinstance(given, bool):
            given = None  # An element index is not a coordinate.
        action_dict = {
            "action": name,
            "coordinates": given or norm_coords,
            COORDINATE_SPACE_KEY: COORDINATE_SPACE_NORMALIZED,
            "args": args,
        }
        if norm_coords:
            action_dict["normalized_coordinates"] = norm_coords
        if norm_start and norm_end:
            action_dict["normalized_start_coordinates"] = norm_start
            action_dict["normalized_end_coordinates"] = norm_end
        action_dict.update(metadata.get("target_semantics") or {})
        return action_dict

    def _record_action_step(
        self,
        name: str,
        args: dict,
        exec_result,
        raw_text: str,
        step_token_usage: dict,
        pre_screenshot_bytes,
        xml_list,
        post_img_bytes,
        injected: str | None = None,
        native_text: str | None = None,
        action_dict: dict | None = None,
    ):
        """Records telemetry / step in DataEngine; returns the step id or None."""
        recorded_step_id = None
        try:
            if self.ctx.data_engine.current_step_id is None:
                self.ctx.data_engine.allocate_step_id()

            if action_dict is None:
                action_dict = self._build_action_record(name, args, exec_result)

            succeeded = exec_result.status == "success"
            last_execution_result = {
                "status": "dispatched" if succeeded else "failed",
                "result": exec_result.text_summary,
            }
            if not succeeded:
                last_execution_result["error"] = exec_result.text_summary

            extra_metadata: dict = {"token_usage": step_token_usage}
            if injected:
                # Stamped verbatim on the step it reached: the chunk ledger
                # keeps it as a never-evicted line at every compression level.
                extra_metadata["injected_instruction"] = injected

            recorded_step_id = self.ctx.data_engine.record_step(
                pre_screenshot_bytes=pre_screenshot_bytes,
                post_screenshot_bytes=post_img_bytes,
                ui_tree=(exec_result.ui_elements_text or xml_list),
                action_taken=action_dict,
                operator_raw_thinking=raw_text,
                operator_native_thinking=native_text or None,
                last_execution_result=last_execution_result,
                extra_metadata=extra_metadata,
            )
            self._notify_history_chunker(recorded_step_id)
        except Exception as step_err:
            logger.warning(f"Error recording step in FlashRunner: {step_err}")
        return recorded_step_id

    def _notify_history_chunker(self, step_id) -> None:
        """Stamps the recorded step for the chunk manager (single segment, no plan)."""
        ledger = getattr(self.ctx, "transcript_ledger", None)
        chunker = getattr(ledger, "chunker", None) if ledger is not None else None
        if chunker is None or step_id is None:
            return
        try:
            chunker.on_step_stamped(str(step_id), None)
        except Exception as e:
            logger.warning(f"History chunker step stamp failed: {e}")

    async def _execute_and_record_action(
        self,
        name: str,
        args: dict,
        tc_id: str,
        state: State,
        messages: list[BaseMessage],
        raw_text: str,
        step_token_usage: dict,
        pre_screenshot_bytes,
        xml_list,
        action_sequence: int,
        turn: _TurnRecord,
        injected: str | None = None,
        native_text: str | None = None,
        index_elements: list[dict] | None = None,
    ):
        """Executes one tool call, records it, and updates loop state.

        The tool message carries the outcome text only; the post-action
        screenshot and UI list become the next turn's observation tail.
        ``index_elements`` is the turn's element snapshot index targets
        resolve against (see :meth:`_process_tool_calls`); ``None`` resolves
        against the state's current list. Returns the (possibly updated)
        pre_screenshot_bytes, xml_list, and action_sequence for the next
        iteration.
        """
        # Dynamic dispatch set: manifest device actions plus any backend
        # extension tools, so extension steps are recorded like actions.
        action_names = self.executor.action_tool_names
        try:
            exec_result = await self.executor.execute(
                name, args, tc_id, state, index_elements=index_elements
            )

            post_img_bytes = await self._capture_post_screenshot(
                exec_result, name, action_names, state
            )

            # One record shape for the DataEngine step and the visual lens.
            action_dict = (
                self._build_action_record(name, args, exec_result) if name in action_names else None
            )

            # Record telemetry / step in DataEngine
            recorded_step_id = None
            if self.ctx.data_engine and name in action_names:
                recorded_step_id = self._record_action_step(
                    name,
                    args,
                    exec_result,
                    raw_text,
                    step_token_usage,
                    pre_screenshot_bytes,
                    xml_list,
                    post_img_bytes,
                    injected=injected,
                    native_text=native_text,
                    action_dict=action_dict,
                )

            if name in action_names:
                turn.step_keys.append(str(recorded_step_id) if recorded_step_id else str(tc_id))
                turn.actions.append((name, exec_result.status, exec_result.text_summary))

            # ⚡ Non-blocking dispatch of objective visual transition summarizer
            if self.summarizer and name in action_names:
                action_sequence += 1
                self.summarizer.dispatch(
                    step_number=action_sequence,
                    action_name=name,
                    # The recorded action minus its verb: the same shape the Pro
                    # summarizer hands the lens, so the phrase renders alike.
                    action_args={k: v for k, v in action_dict.items() if k != "action"},
                    pre_img_bytes=pre_screenshot_bytes,
                    post_img_bytes=post_img_bytes,
                    exec_outcome=exec_result.text_summary,
                    action_key=str(tc_id),
                    data_engine_step_id=recorded_step_id,
                    # The model's own reasoning for this turn steers the lens's
                    # attention; the target it named rides on ``args``.
                    focus=build_focus_context(
                        intent=raw_text,
                        goal=self.goal,
                        injected_instruction=injected,
                    ),
                )

            if post_img_bytes:
                pre_screenshot_bytes = post_img_bytes
            if exec_result.ui_elements_text:
                xml_list = exec_result.ui_elements_text

            # Helper tools may return multimodal blocks (a step screenshot);
            # device actions always report text — their post screenshot is the
            # next observation tail, never a tool-message image. The image
            # carrier follows the model's provider (tool_result_messages).
            raw_blocks = exec_result.raw_result if name not in action_names else None
            content = (
                raw_blocks
                if isinstance(raw_blocks, list) and raw_blocks
                else exec_result.text_summary or f"Action '{name}' dispatched."
            )
            messages.extend(
                tool_result_messages(
                    tc_id, content, name=name, status=exec_result.status, llm=self._llm
                )
            )

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            if name in action_names:
                turn.step_keys.append(str(tc_id))
                turn.actions.append((name, "error", f"Error executing tool {name}: {e}"))
            messages.append(
                ToolMessage(
                    tool_call_id=tc_id,
                    name=name,
                    content=f"Error executing tool {name}: {e}",
                    status="error",
                )
            )
        return pre_screenshot_bytes, xml_list, action_sequence

    async def _invoke_model(self, llm, current_tools: list, messages: list[BaseMessage]):
        """Binds the active tools and invokes the model through the LLM gateway."""
        # Keep only the latest screenshot: strict gateways cap images per
        # request, and earlier frames are stale context anyway.
        self._prune_intermediate_screenshots(messages)
        if os.environ.get("ARTEMIS_DEBUG_PROMPTS"):
            # Per-turn prompt digests: [msg_type, per-block (kind,len,hash)]
            # so prefix-cache divergence points can be localized by diffing.
            import hashlib
            import json as _json

            dump_path = Path(os.environ["ARTEMIS_DEBUG_PROMPTS"])
            digest = []
            for msg in messages:
                blocks = msg.content if isinstance(msg.content, list) else [{"type": "text", "text": msg.content}]
                digest.append([
                    getattr(msg, "type", "?"),
                    [
                        (
                            b.get("type", "?"),
                            len(str(b.get("text") or b.get("image_url") or "")),
                            hashlib.md5(str(b).encode()).hexdigest()[:8],
                        )
                        for b in blocks
                    ],
                ])
            with dump_path.open("a") as fh:
                fh.write(_json.dumps(digest) + "\n")
        # Bind active tools. Local OpenAI-compatible servers (mlx_vlm and
        # friends) may let the model answer with narration only, which the
        # loop reads as a no-progress turn; tool_choice="required" holds the
        # provider to the loop's contract that every turn is a tool call.
        endpoint = getattr(llm, "endpoint", None)
        provider = getattr(endpoint, "provider", None)
        if provider == ModelProvider.CUSTOM:
            bound_llm = llm.bind_tools(current_tools, tool_choice="required")
        else:
            bound_llm = llm.bind_tools(current_tools)

        # Invoke Model. Streaming, live-token UI deltas, classified
        # retries, and pause/resume are all owned by the LLM gateway
        # (acomplete); a typed LLMCallError propagates if it gives up.
        return await invoke_llm_with_timeout_message(
            acomplete(bound_llm, messages),
            timeout_seconds=10,
            hard_timeout=settings.LLM_FLASH_HARD_TIMEOUT_SECONDS,
        )

    async def _process_tool_calls(
        self,
        tool_calls: list,
        state: State,
        messages: list[BaseMessage],
        raw_text: str,
        step_token_usage: dict,
        pre_screenshot_bytes,
        xml_list,
        action_sequence: int,
        turn: _TurnRecord,
        injected: str | None = None,
        native_text: str | None = None,
    ):
        """Dispatches the turn's tool calls.

        Every index target of the turn resolves against the element list the
        model decided on: the list of the observation this turn was built
        from, snapshotted here before the first action. Each action still
        refreshes ``state.indexed_elements`` (that refreshed list is what the
        next observation shows), so without the snapshot ``click(5)`` after
        ``click(3)`` would be resolved against the screen after the first
        click.

        The snapshot is the very list object the observation left on the
        state (not a copy), and observations replace the state's list rather
        than mutating it, so ``ask_explorer`` candidates registered before
        any action land in the snapshot by construction. When a helper tool
        appends candidates after an action has already replaced the state's
        list, the explorer numbered them against that replaced list, so the
        snapshot is re-taken from it: indices the explorer just handed out
        resolve correctly, at the cost of the pre-action numbering (mixing
        the two in one turn is the model's error, not a supported case).

        Returns (final_report, pre_screenshot_bytes, xml_list, action_sequence);
        final_report is non-None only when 'report_task_status' was called.
        """
        action_names = self.executor.action_tool_names
        index_elements = self._turn_index_snapshot(state)
        for tc in tool_calls:
            name = tc["name"].split(":")[-1] if ":" in tc["name"] else tc["name"]
            args = tc.get("args") or {}
            tc_id = tc.get("id") or str(uuid.uuid4())
            logger.info(f"Executing Flash tool: {name}({args})")

            if name == "report_task_status":
                # Terminal-boundary verification: a 'completed' claim is only
                # released when the final check confirms the goal from device
                # state. A rejected claim feeds the verdicts back and the run
                # continues; once the retry budget (final_check_max_attempts)
                # is spent, the report is finalized as failed — a blocked
                # outcome, mirroring the Pro graph.
                if args.get("status") == "completed":
                    allowed, findings = await self._verify_completion(state)
                    if not allowed:
                        self._completion_rejections += 1
                        setup = getattr(self.ctx, "execution_setup", None)
                        max_attempts = int(
                            getattr(setup, "final_check_max_attempts", 3) or 3
                        )
                        if self._completion_rejections < max_attempts:
                            logger.info(
                                "Completion report rejected by final check"
                                f" ({self._completion_rejections}/{max_attempts}):"
                                f" {findings}"
                            )
                            self._record_terminal_step(
                                args,
                                raw_text,
                                step_token_usage,
                                pre_screenshot_bytes,
                                xml_list,
                                last_execution_result={
                                    "result": (
                                        "Completion rejected by final check:"
                                        f" {findings}"
                                    )
                                },
                                native_text=native_text,
                            )
                            messages.append(
                                ToolMessage(
                                    tool_call_id=tc_id,
                                    name=name,
                                    content=json.dumps(
                                        {
                                            "status": "rejected",
                                            "reason": (
                                                "Completion verification"
                                                " failed."
                                            ),
                                            "findings": findings,
                                            "instruction": (
                                                "The final check could not"
                                                " confirm the goal from device"
                                                " state. Continue the task, or"
                                                " report status 'failed'"
                                                " honestly."
                                            ),
                                        }
                                    ),
                                    status="error",
                                )
                            )
                            continue
                        logger.warning(
                            "Final-check retry budget exhausted; finalizing"
                            f" as failed. Last findings: {findings}"
                        )
                        args = dict(args)
                        args["status"] = "failed"
                        args["explanation"] = (
                            f"{args.get('explanation', '')}\n\n"
                            "Completion verification repeatedly failed;"
                            f" blocked outcome. {findings}"
                        ).strip()
                final_report = await self._finalize_task_report(
                    name,
                    args,
                    tc_id,
                    raw_text,
                    step_token_usage,
                    pre_screenshot_bytes,
                    xml_list,
                    messages,
                    native_text=native_text,
                )
                return final_report, pre_screenshot_bytes, xml_list, action_sequence

            elements_before = getattr(state, "indexed_elements", None)
            before_len = len(elements_before) if isinstance(elements_before, list) else 0
            pre_screenshot_bytes, xml_list, action_sequence = await self._execute_and_record_action(
                name,
                args,
                tc_id,
                state,
                messages,
                raw_text,
                step_token_usage,
                pre_screenshot_bytes,
                xml_list,
                action_sequence,
                turn,
                injected=injected,
                native_text=native_text,
                index_elements=index_elements,
            )
            if name not in action_names:
                index_elements = self._extend_index_snapshot(state, index_elements, before_len)
        return None, pre_screenshot_bytes, xml_list, action_sequence

    @staticmethod
    def _turn_index_snapshot(state: State) -> list[dict]:
        """The element list this turn's index targets resolve against.

        The state's list object itself, so in-place appends (explorer
        candidates) before any action join the snapshot; an empty list is
        installed on the state when there is none, for the same reason.
        """
        elements = getattr(state, "indexed_elements", None)
        if not isinstance(elements, list):
            elements = []
            try:
                state.indexed_elements = elements
            except (AttributeError, TypeError, ValueError):
                pass
        return elements

    @staticmethod
    def _extend_index_snapshot(state: State, snapshot: list[dict], before_len: int) -> list[dict]:
        """Re-takes the snapshot after a helper tool appended candidates to a
        list that is no longer the snapshot (an action replaced it first).

        Appends into the snapshot itself need nothing: the explorer numbered
        them against the snapshot. Only growth of a different list object
        moves the snapshot, since that is the list the new numbers refer to.
        """
        current = getattr(state, "indexed_elements", None)
        if not isinstance(current, list) or current is snapshot:
            return snapshot
        if len(current) > before_len:
            return current
        return snapshot

    async def _prepare_conversation(self, state: State, tools_declaration: list):
        """Installs the static prefix and captures the initial device state."""
        ledger = self._build_ledger()
        ledger.set_static_prefix(
            [SystemMessage(content=self._render_system_prompt(tools_declaration))]
        )

        # Capture Initial State (Screenshot + UI Tree)
        shot_path, img_bytes, xml_list = await capture_screenshot_and_parse_ui(
            self.ctx, state, self.controller, skip_settling=False
        )
        state.latest_screenshot = shot_path
        return ledger, img_bytes, xml_list

    @staticmethod
    def _commit_turn(ledger: TranscriptLedger, turn: _TurnRecord | None) -> None:
        """Commits the previous turn once its step ids and outcomes are known."""
        if turn is None:
            return
        ledger.commit_staged(
            step_key=turn.step_keys[0] if turn.step_keys else None,
            validator_result=turn.result(),
            extra_step_keys=turn.step_keys[1:],
        )

    @trace(type="agent", name="FlashRunner")
    async def run(self, state: State) -> dict:
        limit = self.turn_limit
        logger.info(
            f"Starting Artemis Flash reactive loop for goal: {self.goal}"
            f" (turn limit: {limit if limit else 'unlimited'})"
        )

        # 1. Initialize Universal LLM via Service Layer
        llm = self._init_llm()
        self._llm = llm

        # Hybrid loop (decision.flash_loop): VLM describes, a typed decision
        # model picks the action. Both flags must be on; a missing/unreachable
        # decision endpoint escalates per-turn to the standard tool-call path.
        decision_cfg = getattr(getattr(self.ctx, "agent_config", None), "decision", None)
        hybrid = bool(
            decision_cfg
            and getattr(decision_cfg, "enabled", False)
            and getattr(decision_cfg, "flash_loop", False)
        )
        setup = getattr(self.ctx, "execution_setup", None)
        self._goal_invariants = list(getattr(setup, "goal_invariants", None) or [])
        if self._goal_invariants:
            logger.info(
                f"Goal invariants armed: {len(self._goal_invariants)} declared"
                " condition(s) gate completion and may close the run early."
            )
        tools_declaration = self._get_tools()
        report_only_tools = [t for t in tools_declaration if t.name == "report_task_status"]

        ledger, img_bytes, xml_list = await self._prepare_conversation(state, tools_declaration)

        plan_text = None
        if hybrid:
            logger.info("Hybrid Flash loop enabled: VLM describes, decision model acts.")
            plan_text = await self._hybrid_plan(llm, img_bytes, xml_list)
            self._plan_text = plan_text

        turns = 0
        action_sequence = 0
        final_report = None
        current_pre_screenshot_bytes = img_bytes
        current_xml_list = xml_list
        previous_turn: _TurnRecord | None = None
        pending_notices: list[str] = []
        no_tool_call_streak = 0
        # Element taps already executed, by (index, label) identity plus a
        # distinctive-label set that survives re-indexing. The decision model
        # cannot see the transcript, so masked menu entries are the only
        # thing stopping it from re-picking a dead element forever (observed
        # live: three identical taps on an unchanged screen, and the same
        # icon tapped again four turns later once it worked). Meta actions
        # are tracked only as a consecutive-repeat streak — presses back/home
        # can be legitimately repeated during navigation.
        tried_taps: set[tuple[int, str]] = set()
        tried_labels: set[str] = set()
        hybrid_last_key: str | None = None
        hybrid_last_call_sig = ""
        hybrid_repeat_streak = 0
        # Whether the previous turn's actions left the screen unchanged —
        # the pre-execution half of verbatim dead-call suppression.
        prev_turn_unchanged = False
        # Sliding histories for cycle detection: an A,B,A,B ping-pong (enter
        # a page, back out, re-enter — observed live on Network & internet)
        # never repeats a call verbatim and every screen "changes", so it
        # escapes dead-call tracking. The give-away is re-issuing the call
        # from two turns ago on a matching screen.
        call_sig_history: list[str] = []
        pre_sig_history: list[frozenset] = []
        # How many turns in a row action calls were suppressed because the
        # declared goal invariants already held on the live screen. Bounded
        # like the other streaks: after the model ignores the notice a few
        # times, the verified conditions close the run instead of letting a
        # redundant action undo them (observed live: the 'dark theme off'
        # run toggled a satisfied switch back on).
        satisfied_suppression_streak = 0

        while limit is None or turns < limit:
            turns += 1
            logger.info(f"--- Artemis Flash Turn {turns}{f'/{limit}' if limit else ''} ---")

            if self.ctx.data_engine:
                self.ctx.data_engine.allocate_step_id()

            # Commit the previous turn: its step ids and outcomes exist now.
            self._commit_turn(ledger, previous_turn)
            previous_turn = None

            # Check for real-time injected instructions: the verbatim text
            # is stamped on the step, the guidance pair goes to the tail.
            injected, injected_guidance = await self._read_injected_instruction(
                ledger.elapsed_label()
            )

            # Tool restriction on the final turn (bounded loops only)
            is_final = limit is not None and turns == limit
            tail = self._build_tail(
                ledger,
                turns,
                current_pre_screenshot_bytes,
                current_xml_list,
                injected=injected_guidance,
                notices=pending_notices,
                is_final=is_final,
                # The ledger judged the turn it just committed: a bare tool
                # call (or thinking only) earns a one-line reminder, no bounce.
                previous_turn_silent=bool(getattr(ledger, "last_turn_silent", False)),
            )
            pending_notices = []

            # S + F + A + tail: chunk compression and the scrub edge advance here.
            messages = ledger.render([tail])
            turn_base = len(messages) - 1
            current_tools = report_only_tools if is_final else tools_declaration

            # Hybrid turn: describe → decision-model action pick. The final
            # turn stays on the standard path (report_task_status only), and
            # a repeat streak escalates to the full model entirely.
            response = None
            tool_calls = None
            if (
                hybrid
                and not is_final
                and hybrid_repeat_streak < _HYBRID_REPEAT_ESCALATE
            ):
                decided = await self._hybrid_decide(
                    llm,
                    state,
                    tail,
                    plan_text,
                    decision_cfg,
                    tried_taps=tried_taps,
                    tried_labels=tried_labels,
                )
                if decided is not None:
                    response, tool_calls = decided

            if response is None:
                response = await self._invoke_model(llm, current_tools, messages)

            if response is None:
                break

            messages.append(response)

            # Extract thought and tool calls
            raw_text = self._extract_response_text(response)
            native_text = self._extract_response_thinking(response)

            # Extract token usage metadata from response
            step_token_usage, measured = self._resolve_token_usage(response, messages, raw_text)
            # The ledger's compaction thresholds read the runner's own prompt
            # size, never the session-wide meter (which any agent overwrites).
            # The prompt as sent (``messages`` minus the just-appended reply)
            # rides along so the ledger can calibrate its chars-per-token
            # ratio — only from a provider measurement, never from the
            # estimate (same as the Pro operator).
            ledger.record_prompt_tokens(
                step_token_usage.get("prompt_tokens"),
                messages=messages[:-1] if measured else None,
            )

            self._record_llm_trace(step_token_usage, raw_text)

            if tool_calls is None:
                tool_calls = self._resolve_tool_calls(response, raw_text)

            if not tool_calls:
                logger.info(
                    f"FlashRunner received response without tool calls at turn {turns}:"
                    f" {raw_text[:100]}..."
                )
                if is_final:
                    final_report = {"status": "failed", "explanation": raw_text}
                    break
                # Local VLM servers (mlx_vlm) accept tool_choice="required" yet
                # still let the model narrate. Re-ask once, in-turn, with an
                # explicit directive before counting a no-progress turn — the
                # model does emit calls when pushed (observed: it narrates
                # "the toggle is on" instead of calling report_task_status).
                messages.append(
                    HumanMessage(content=[{"type": "text", "text": _NO_TOOL_CALL_NOTICE}])
                )
                retry_response = await self._invoke_model(llm, current_tools, messages)
                if retry_response is not None:
                    retry_text = self._extract_response_text(retry_response)
                    retry_calls = self._resolve_tool_calls(retry_response, retry_text)
                    if retry_calls:
                        response, raw_text, tool_calls = retry_response, retry_text, retry_calls
                        messages.append(response)

            if not tool_calls:
                no_tool_call_streak += 1
                if no_tool_call_streak >= _MAX_CONSECUTIVE_NO_TOOL_CALLS:
                    final_report = {
                        "status": "failed",
                        "explanation": (
                            f"Stopped after {no_tool_call_streak} consecutive responses"
                            " without a tool call; the model kept narrating instead"
                            " of acting."
                        ),
                    }
                    break
                pending_notices.append(_NO_TOOL_CALL_NOTICE)
                # No action ran, so nothing refreshes the observation — but
                # the current frame may be a mid-transition capture (the prior
                # action's post-screenshot raced a screen animation). Re-serve
                # that stale frame next turn and both the describe pass and
                # the hybrid menu keep acting on the wrong screen. Re-observe
                # here so the next tail reflects what is actually displayed.
                (
                    current_pre_screenshot_bytes,
                    current_xml_list,
                ) = await self._refresh_idle_observation(
                    state, current_pre_screenshot_bytes, current_xml_list
                )
                ledger.stage_turn(messages[turn_base:])
                previous_turn = _TurnRecord()
                continue

            no_tool_call_streak = 0

            # Snapshot the pre-action screen for the repeat machinery. A tap
            # only joins the masked sets when it did NOT change the screen —
            # a tap that navigated somewhere is working navigation, and masking
            # its label would eventually erase every goal-relevant element from
            # the menu (observed: the dark-theme run masked 'Use Dark theme'
            # itself, leaving the VLM to toggle the switch blind forever).
            elements_now = getattr(state, "indexed_elements", None) or []
            pre_sig = self._element_signature(elements_now)
            turn_pre_image = current_pre_screenshot_bytes
            label_by_idx = {
                el.get("index"): self._element_label(el)
                for el in elements_now
                if el.get("index") is not None
            }
            turn_keys: list[str] = []
            tap_records: list[tuple[int, str]] = []
            call_sig = "|".join(
                sorted(
                    f"{tc.get('name')}:{json.dumps(tc.get('args') or {}, sort_keys=True, default=str)}"
                    for tc in tool_calls
                )
            )
            for tc in tool_calls:
                key = self._menu_key_for_call(tc.get("name"), tc.get("args"))
                if key is None:
                    continue
                turn_keys.append(key)
                if key.startswith("tap_"):
                    idx = int(key[len("tap_"):])
                    tap_records.append((idx, label_by_idx.get(idx, "")))

            # Declared-invariant short-circuit: when every goal invariant is
            # verified satisfied on the live screen, further actions can only
            # risk undoing the achieved state. Suppress the action calls
            # (a report_task_status in the set still goes through — the
            # completion gate will release it on the same evidence) and tell
            # the model to report; after enough ignored notices the verified
            # conditions close the run themselves.
            if self._goal_invariants:
                has_report_call = any(
                    str(tc.get("name") or "").split(":")[-1] == "report_task_status"
                    for tc in tool_calls
                )
                # A report call defers to the completion gate (which re-
                # observes itself); otherwise re-observe now — the elements
                # snapshot above comes from the post-action capture, which
                # can lag a still-applying transition by a whole turn.
                inv_elements = (
                    elements_now
                    if has_report_call
                    else await self._reobserve_elements(state)
                )
                inv_report = evaluate_invariants(self._goal_invariants, inv_elements)
                if inv_report["verdict"] == "unsatisfied" and not has_report_call:
                    # Evidence-first feedback: the model narrates its own plan
                    # ("I dialed 55501") and can drift from the field's real
                    # contents. The invariant already measured live state —
                    # put the observed-vs-expected values where they cannot
                    # be missed, before the next action proposal.
                    pending_notices.append(
                        "[GOAL CHECK] Declared conditions are not yet met on"
                        " the live screen — act on these observed values, not"
                        " on your plan:\n" + describe_failures(inv_report)
                    )
                if inv_report["verdict"] == "satisfied" and not has_report_call:
                    satisfied_suppression_streak += 1
                    logger.info(
                        "Suppressing actions: declared goal invariants already"
                        f" satisfied ({satisfied_suppression_streak}/3)."
                    )
                    notice = (
                        "[NOTICE] The declared goal conditions are already"
                        " satisfied on the current screen. Do not act further"
                        " — call report_task_status('completed')."
                    )
                    for tc in tool_calls:
                        messages.append(
                            ToolMessage(
                                tool_call_id=tc.get("id") or str(uuid.uuid4()),
                                name=str(tc.get("name") or ""),
                                content=json.dumps(
                                    {
                                        "status": "error",
                                        "error": (
                                            "Goal conditions already satisfied;"
                                            " action not executed. Report"
                                            " task status instead."
                                        ),
                                    }
                                ),
                                status="error",
                            )
                        )
                    if satisfied_suppression_streak >= 3:
                        ledger.stage_turn(messages[turn_base:])
                        if self.summarizer:
                            await self.summarizer.flush()
                        return {
                            "status": "completed",
                            "explanation": (
                                "Declared goal conditions verified satisfied"
                                " on device state: "
                                + ", ".join(
                                    str(r.get("predicate_id") or "invariant")
                                    for r in inv_report["results"]
                                )
                            ),
                        }
                    pending_notices.append(notice)
                    (
                        current_pre_screenshot_bytes,
                        current_xml_list,
                    ) = await self._refresh_idle_observation(
                        state, current_pre_screenshot_bytes, current_xml_list
                    )
                    ledger.stage_turn(messages[turn_base:])
                    previous_turn = _TurnRecord()
                    continue
                satisfied_suppression_streak = 0

            # Dead-verbatim suppression: when this turn's call set is identical
            # to the previous turn's and that turn changed nothing, executing
            # the same actions again can only produce the same dead result —
            # the VLM is looping on an element it keeps mis-picking (observed:
            # five identical click(10)s on a blank row). Answer the tool calls
            # with an error instead so the next turn must choose differently;
            # the notice names navigation alternatives.
            if call_sig and call_sig == hybrid_last_call_sig and prev_turn_unchanged:
                nav = self._navigation_suggestions(elements_now)
                notice = (
                    "[NOTICE] The exact same action was already executed and"
                    " changed nothing. It was not re-run. Pick a different"
                    " action."
                    + (f" Available navigation elements: {nav}." if nav else "")
                )
                logger.info(f"Suppressing verbatim repeat of dead call set: {call_sig}")
                for tc in tool_calls:
                    messages.append(
                        ToolMessage(
                            tool_call_id=tc.get("id") or str(uuid.uuid4()),
                            name=str(tc.get("name") or ""),
                            content=json.dumps(
                                {
                                    "status": "error",
                                    "error": (
                                        "This exact action was already executed"
                                        " and changed nothing; it was not"
                                        " re-run. Choose a different action."
                                    ),
                                }
                            ),
                            status="error",
                        )
                    )
                pending_notices.append(notice)
                hybrid_repeat_streak += 1
                if hybrid_repeat_streak >= 3:
                    # Third verbatim re-issue of a call that was suppressed
                    # with an explicit "not re-run" error — the model cannot
                    # break this loop itself (observed: 30+ turns of the same
                    # suppressed click until the turn cap). Fail honestly
                    # instead of burning the rest of the budget.
                    ledger.stage_turn(messages[turn_base:])
                    if self.summarizer:
                        await self.summarizer.flush()
                    return {
                        "status": "failed",
                        "explanation": (
                            "The agent kept re-issuing an action that was"
                            " already suppressed for having no effect, and"
                            " could not find an alternative way forward."
                        ),
                    }
                (
                    current_pre_screenshot_bytes,
                    current_xml_list,
                ) = await self._refresh_idle_observation(
                    state, current_pre_screenshot_bytes, current_xml_list
                )
                ledger.stage_turn(messages[turn_base:])
                previous_turn = _TurnRecord()
                continue

            # Process tool calls
            turn = _TurnRecord()
            (
                final_report_from_calls,
                current_pre_screenshot_bytes,
                current_xml_list,
                action_sequence,
            ) = await self._process_tool_calls(
                tool_calls,
                state,
                messages,
                raw_text,
                step_token_usage,
                current_pre_screenshot_bytes,
                current_xml_list,
                action_sequence,
                turn,
                injected=injected,
                native_text=native_text,
            )

            # Did anything actually change? Strict set equality — volatile
            # sources are filtered inside the signature, so a single real
            # difference (one digit appended, one switch flipped) counts as
            # changed even on a dense screen. Mask dead taps, count repeats
            # of already-dead keys, and flag a verbatim re-issue of last
            # turn's call set.
            post_sig = self._element_signature(
                getattr(state, "indexed_elements", None) or []
            )
            unchanged = pre_sig == post_sig
            if unchanged and turn.actions:
                # Two stale-evidence guards before a 'dead' verdict. The
                # a11y tree can freeze while the screen still moves —
                # observed live: a frozen helper tree served identical XML
                # across a whole dial-pad run while digits accumulated.
                # Pixels come through a different capture path, so a
                # changed screenshot disproves 'dead' even when the tree
                # lies. Failing that, the post-action dump may just lag a
                # settling UI — re-observe once; a real dead tap stays
                # dead on the second look.
                if self._screens_differ(turn_pre_image, current_pre_screenshot_bytes):
                    unchanged = False
                    logger.info(
                        "Post-action screenshot changed while the element"
                        " tree did not — stale hierarchy, not a dead turn."
                    )
                else:
                    fresh = await self._reobserve_elements(state)
                    post_sig = self._element_signature(fresh)
                    unchanged = pre_sig == post_sig
                    if not unchanged:
                        logger.info(
                            "Re-observation resolved an ambiguous dead read:"
                            " the screen did change after all."
                        )
            repeat_keys: list[str] = []
            for key in turn_keys:
                if key.startswith("tap_"):
                    idx = int(key[len("tap_"):])
                    label = label_by_idx.get(idx, "")
                    ident = (idx, label)
                    if ident in tried_taps or (
                        len(label) >= 4 and label in tried_labels
                    ):
                        repeat_keys.append(key)
                elif key == hybrid_last_key:
                    repeat_keys.append(key)
                hybrid_last_key = key
            if unchanged:
                for idx, label in tap_records:
                    tried_taps.add((idx, label))
                    if len(label) >= 4:
                        tried_labels.add(label)
            verbatim_repeat = (
                unchanged and call_sig and call_sig == hybrid_last_call_sig
            )
            ping_pong = (
                bool(call_sig)
                and len(call_sig_history) >= 2
                and call_sig == call_sig_history[-2]
                and self._signatures_similar(pre_sig, pre_sig_history[-2])
            )
            hybrid_last_call_sig = call_sig
            prev_turn_unchanged = unchanged
            call_sig_history.append(call_sig)
            pre_sig_history.append(pre_sig)
            del call_sig_history[:-12]
            del pre_sig_history[:-12]
            if repeat_keys or verbatim_repeat or ping_pong:
                hybrid_repeat_streak += 1
                noted = repeat_keys or ["the same action"]
                nav = self._navigation_suggestions(elements_now)
                if ping_pong and not repeat_keys and not verbatim_repeat:
                    notice = (
                        "[NOTICE] You are alternating between two screens —"
                        " this action was already tried two steps ago and the"
                        " following action undid it. Do not go back and"
                        " forth: commit to a direction, finish the goal on"
                        " the current screen, or report the task status."
                    )
                else:
                    notice = (
                        f"[NOTICE] {', '.join(sorted(set(noted)))} was already"
                        " attempted and did not move the task forward."
                    )
                if nav:
                    notice += f" Available navigation elements instead: {nav}."
                if hybrid_repeat_streak >= 3:
                    notice += (
                        " This action has now failed several times in a row —"
                        " do not repeat it; use a different element, a system"
                        " key (back/home), or report_task_status('failed')."
                    )
                else:
                    notice += " Pick a different action or report the task status."
                pending_notices.append(notice)
            else:
                hybrid_repeat_streak = 0

            # Feed the decision model a one-line outcome: it cannot see the
            # transcript, so a no-op tap and real progress otherwise look the
            # same on the next pick (observed: three re-picks of a dead row).
            if tool_calls:
                names = ", ".join(str(tc.get("name") or "?") for tc in tool_calls)
                outcome = "no visible change" if unchanged else "screen changed"
                self._last_action_note = f"{names} → {outcome}"
            else:
                self._last_action_note = None

            ledger.stage_turn(messages[turn_base:])
            previous_turn = turn
            if final_report_from_calls is not None:
                return final_report_from_calls

        if self.summarizer:
            await self.summarizer.flush()

        if final_report is None:
            explanation = (
                "Max turns reached without final status report."
                if limit is not None and turns >= limit
                else "The model returned no response; the reactive loop stopped."
            )
            final_report = {"status": "failed", "explanation": explanation}
        return final_report
