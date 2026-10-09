# Running ARTEMIS on a Local Model

ARTEMIS can drive every agent node (planner, operator, checker, flash,
grounding, OCR helpers) from a single local multimodal model served over an
OpenAI-compatible HTTP API — no cloud key required.

Tested reference setup: **Gemma 4 E4B (4-bit, MLX)** on Apple Silicon —
~6 GB resident, ~40 tok/s decode, both `flash` and `pro` profiles verified
end-to-end on a real device.

Any server that speaks `/v1/chat/completions` (text + `image_url` + tool
calls) works the same way: `ondevice-agent-platform`, `mlx_lm.server`,
Ollama, LM Studio, llama.cpp, vLLM, …

## 1. Install a model server

Pick one. The `ondevice-agent-platform` route is the tested path and is what
the readiness probe's remediation hints assume.

**ondevice-agent-platform (reference)**

```bash
# provides the `ondevice-agent-platform` CLI on PATH
ondevice-agent-platform serve --port 8080
```

**Generic alternative — MLX (Apple Silicon)**

```bash
pip install mlx-lm
mlx_lm.server --model mlx-community/gemma-4-E4B-it-4bit --port 8080
```

**Generic alternative — Ollama**

```bash
ollama serve   # listens on :11434; use api_base http://127.0.0.1:11434/v1
```

## 2. Pull the model

```bash
# ondevice-agent-platform
ondevice-agent-platform model pull --alias gemma4-e4b

# Ollama
ollama pull gemma3:4b
```

Whatever name the server reports in `GET /v1/models` is the string the
ARTEMIS config's `"model"` field must match exactly.

```bash
curl http://127.0.0.1:8080/v1/models   # confirm your alias appears
```

## 3. Point ARTEMIS at it

A ready-made config ships with this repo:

```bash
export ARTEMIS_ARTEMIS_JSONC=config/artemis.gemma4.jsonc
# or copy it over your artemis.jsonc
```

To adapt it for a different server, edit the `default` block (every node
inherits it unless overridden):

```jsonc
"default": {
  "provider": "custom",                       // skips cloud credential checks
  "model": "gemma4-e4b",                      // must match /v1/models exactly
  "api_base": "http://127.0.0.1:8080/v1",     // your server
  "timeout": 300,                             // local decode is slower; keep generous
}
```

## 4. Verify

```bash
uv run artemis doctor
```

The **Local Model Endpoint** probe checks the endpoint is reachable and that
every configured alias is actually served:

- ✅ `Serving N model(s)` — ready
- ❌ `model 'X' not installed` → suggests `ondevice-agent-platform model pull --alias X`
- ❌ `endpoint unreachable` → suggests `ondevice-agent-platform serve`

The same probe backs the web console's setup wizard and `mobile_diagnose`.

## 5. Run

```bash
uv run artemis run "Open Settings" --profile flash
# or the heavier planner/operator/checker loop:
uv run artemis run "Open Reminders" --profile pro
```

## Tuning notes for small local models

- **Timeouts**: `LLM_FLASH_HARD_TIMEOUT_SECONDS` (default 180 s) caps Flash
  turns; keep `timeout` ≥ 300 in the config for thinking-style models.
- **Images**: strict servers cap images-per-request — Flash prunes
  intermediate screenshots automatically; no action needed.
- **Structured output**: strict JSON-schema `response_format` is not
  required — ARTEMIS uses function-calling for structured nodes.
- **Termination**: 4B-class models can recognize the goal yet keep
  deliberating instead of ending the turn. Prefer bounded goals
  ("open X", "scroll to Y") over open-ended ones until your deployment adds
  an explicit done affordance.
- **Memory**: budget ~6 GB for a 4-bit 4B vision model + KV cache.
