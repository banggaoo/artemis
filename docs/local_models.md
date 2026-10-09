# Running ARTEMIS on a Local Model

ARTEMIS can drive every agent node (planner, operator, checker, flash,
grounding, OCR helpers) from a single local multimodal model — and it can
pull and serve that model itself, no external serving stack or cloud key
required.

Tested reference setup: **Gemma 4 E4B (4-bit, MLX)** on Apple Silicon —
~6 GB resident, ~40 tok/s decode, both `flash` and `pro` profiles verified
end-to-end on a real device.

## 1. Install the local-model extra (Apple Silicon)

```bash
pip install 'artemis[local]'     # or: uv pip install -e '.[local]'
```

This installs `mlx-vlm` + `huggingface-hub`, the serving stack ARTEMIS
manages for you. (On other platforms, skip to
[BYO server](#byo-any-openai-compatible-server).)

## 2. Pull and serve the model

```bash
artemis model list                  # known aliases
artemis model pull gemma4-e4b       # download weights (~5 GB)
artemis model serve gemma4-e4b      # OpenAI-compatible server on :8080
```

`serve` runs `mlx_vlm.server` as a managed background process, waits for
readiness, and prints the endpoint. Lifecycle:

```bash
artemis model status                # is the managed server up?
artemis model stop                  # shut it down
```

The served aliases appear under `GET /v1/models` — the same names the
ARTEMIS config's `"model"` field must match.

## 3. Point ARTEMIS at it

A ready-made config ships with this repo:

```bash
export ARTEMIS_ARTEMIS_JSONC=config/artemis.gemma4.jsonc
# or copy it over your artemis.jsonc
```

The `default` block (every node inherits it unless overridden):

```jsonc
"default": {
  "provider": "custom",                       // skips cloud credential checks
  "model": "gemma4-e4b",                      // must match /v1/models exactly
  "api_base": "http://127.0.0.1:8080/v1",     // the managed server
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
- ❌ `model 'X' not installed` → action: `artemis model pull X`
- ❌ `endpoint unreachable` → action: `artemis model serve X --port 8080`

The same probe backs the web console's setup wizard and `mobile_diagnose`,
so remediation is one click anywhere it appears.

## 5. Run

```bash
uv run artemis run "Open Settings" --profile flash
# or the heavier planner/operator/checker loop:
uv run artemis run "Open Reminders" --profile pro
```

## BYO: any OpenAI-compatible server

ARTEMIS only needs `/v1/chat/completions` with text + `image_url` + tool
calls. Ollama, LM Studio, llama.cpp, vLLM all work — pull the model there,
set `api_base` + `"model"` to match its `/v1/models`, and `artemis doctor`
verifies the same way (remediation hints fall back to generic pull/serve
guidance for non-catalog aliases).

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
