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

"""Typed decision-model client — the "System One" layer.

Decision models (Jev from TypeSafe AI, or the open-weight Laya from Convai
Innovations — served locally via Ollama MLX builds) answer *typed* questions
about a distilled text/JSON state in a single non-autoregressive pass:
``choice`` (pick one of labelled options), ``score`` (ordered rubric), or
``noul`` (calibrated yes/no probability). They never emit free text, so
there is nothing to parse and no format to hallucinate.

Wire format (TypeSafe-compatible, which Laya's ``typed-decisions`` checkpoint
also speaks)::

    POST {url}
    {"model": "...", "state": "...",
     "questions": {"gate": {"type": "noul", "instructions": "..."}}}
    ->
    {"answers": {"gate": {"noul": 0.83, "confidence": 0.91}}, ...}

Used to offload cheap boolean/categorical junctions (validator gates,
planner review, routing triage) from the heavyweight multimodal model.
Fails closed: any transport/parse problem yields ``DecisionResult`` with
``verdict=None`` and ``confidence=0`` so callers escalate to the LLM path.
"""

import asyncio
from dataclasses import dataclass, field
import json
import urllib.error
import urllib.request

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS = 5.0


@dataclass
class DecisionResult:
    """One answered question.

    ``verdict`` is the typed answer — bool for ``noul`` (probability >= 0.5),
    the winning key for ``choice``, a float for ``score`` — or ``None`` when
    the decision failed and the caller should escalate to the model path.
    """

    verdict: bool | str | float | None
    confidence: float = 0.0
    probabilities: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.verdict is not None


class DecisionClient:
    """Minimal urllib client for Jev/Laya-compatible decision endpoints."""

    def __init__(
        self,
        url: str,
        model: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self._url = url
        self._model = model
        self._timeout = timeout_seconds

    def decide(self, state: str, questions: dict) -> dict[str, DecisionResult]:
        """One request, all questions answered in a single forward pass."""
        payload: dict = {"state": state, "questions": questions}
        if self._model:
            payload["model"] = self._model
        request = urllib.request.Request(
            self._url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            logger.warning(f"Decision endpoint {self._url} failed: {e}")
            return {name: DecisionResult(verdict=None) for name in questions}
        answers = body.get("answers") or {}
        return {
            name: self._parse_answer(spec, answers.get(name) or {})
            for name, spec in questions.items()
        }

    async def adecide(self, state: str, questions: dict) -> dict[str, DecisionResult]:
        return await asyncio.to_thread(self.decide, state, questions)

    def noul(self, state: str, question: str) -> DecisionResult:
        """Calibrated yes/no — the gate/guardrail primitive."""
        results = self.decide(
            state, {"q": {"type": "noul", "instructions": question}}
        )
        return results["q"]

    async def anoul(self, state: str, question: str) -> DecisionResult:
        return await asyncio.to_thread(self.noul, state, question)

    def choose(
        self, state: str, question: str, criteria: dict[str, str]
    ) -> DecisionResult:
        """Categorical pick over labelled options."""
        results = self.decide(
            state,
            {"q": {"type": "choice", "instructions": question, "criteria": criteria}},
        )
        return results["q"]

    async def achoose(
        self, state: str, question: str, criteria: dict[str, str]
    ) -> DecisionResult:
        return await asyncio.to_thread(self.choose, state, question, criteria)

    def _parse_answer(self, spec: dict, answer: dict) -> DecisionResult:
        qtype = spec.get("type", "noul")
        confidence = answer.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else 0.0
        except (TypeError, ValueError):
            confidence = 0.0
        probabilities = answer.get("probabilities") or {}
        if qtype == "noul":
            prob = answer.get("noul")
            if prob is None:
                return DecisionResult(verdict=None, confidence=confidence, raw=answer)
            try:
                prob = float(prob)
            except (TypeError, ValueError):
                return DecisionResult(verdict=None, confidence=confidence, raw=answer)
            return DecisionResult(
                verdict=prob >= 0.5,
                confidence=confidence,
                probabilities={"yes": prob, "no": 1.0 - prob},
                raw=answer,
            )
        if qtype == "choice":
            verdict = answer.get("choice")
            return DecisionResult(
                verdict=verdict,
                confidence=confidence,
                probabilities=probabilities,
                raw=answer,
            )
        if qtype == "score":
            score = answer.get("score")
            try:
                verdict = float(score) if score is not None else None
            except (TypeError, ValueError):
                verdict = None
            return DecisionResult(
                verdict=verdict,
                confidence=confidence,
                probabilities=probabilities,
                raw=answer,
            )
        return DecisionResult(verdict=None, confidence=confidence, raw=answer)
