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

import os
from typing import Any

import httpx
from artemis.config import settings
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None

# Provider selection via ARTEMIS_OCR_PROVIDER:
#   "auto"     (default) - the local platform's vision-hybrid route first
#                          (Apple Vision tier, VLM delegate when needed),
#                          then Google Cloud Vision when an API key exists.
#   "platform"           - vision-hybrid only.
#   "google"             - Google Cloud Vision only.
#   "apple"              - deprecated alias for "platform": the standalone
#                          Apple Vision worker was removed; the platform's
#                          oap-vision-bridge is the Apple Vision path now.
_OCR_PROVIDER_ENV = "ARTEMIS_OCR_PROVIDER"
_OAP_API_BASE_ENV = "OAP_API_BASE"
_OAP_OCR_MODEL_ENV = "OAP_OCR_MODEL"


def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)
    return _HTTP_CLIENT


def _ocr_provider() -> str:
    return os.environ.get(_OCR_PROVIDER_ENV, "auto").strip().lower()


def _google_vision_key() -> str | None:
    """Raw Google Vision API key, or None (placeholders are not filtered)."""
    ocr_secret = settings.get_api_key("ocr")
    return (
        (ocr_secret.get_secret_value() if ocr_secret else None)
        or os.environ.get("OCR_API_KEY")
        or os.environ.get("VISION_API_KEY")
    )


def _google_vision_key_present() -> bool:
    """True when a non-placeholder Google Vision key is configured."""
    for val in (_google_vision_key(),):
        if (
            val
            and val.strip()
            and val.strip()
            not in (
                "API_KEY",
                "your_google_cloud_vision_api_key_here",
            )
        ):
            return True
    return False


def is_ocr_configured() -> bool:
    """True when some OCR provider is usable: the local platform
    vision-hybrid route (fixed loopback endpoint - reachability is a
    runtime matter), or a configured Google Cloud Vision API key."""
    provider = _ocr_provider()
    if provider in ("auto", "platform", "apple"):
        return True
    if provider == "google":
        return _google_vision_key_present()
    return _google_vision_key_present()


def _oap_ocr_endpoint() -> tuple[str, str]:
    base = os.environ.get(
        _OAP_API_BASE_ENV, "http://127.0.0.1:8080/v1"
    ).rstrip("/")
    return f"{base}/chat/completions", os.environ.get(
        _OAP_OCR_MODEL_ENV, "vision-hybrid"
    )


async def _run_platform_ocr(
    screenshot_b64: str,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """OCR via the local platform's vision-hybrid route (Apple Vision
    first, VLM delegate when the image is not plain text extraction).
    Structured observations arrive in the additive ``oap_ocr`` response
    field as ``{text, confidence, position}`` with position already in
    pixel vertices (TL,TR,BR,BL). An escalated VLM answer carries no
    positions and yields ``[]`` - callers treat it like an empty OCR."""
    url, model = _oap_ocr_endpoint()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Extract all text from this image."},
                {"type": "image_url",
                 "image_url": {
                     "url": "data:image/png;base64," + screenshot_b64}},
            ],
        }],
    }
    active_client = client if client is not None else get_http_client()
    response = await active_client.post(url, json=payload, timeout=30.0)
    response.raise_for_status()
    results = []
    for obs in response.json().get("oap_ocr") or []:
        text = obs.get("text")
        position = obs.get("position")
        if (
            isinstance(text, str)
            and text.strip()
            and isinstance(position, list)
        ):
            results.append({"text": text, "position": position})
    return results


async def perform_ocr(
    screenshot_b64: str,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Runs text recognition on an image with the selected provider.

    Provider order under "auto": the local platform vision-hybrid route
    first, then Google Cloud Vision when an API key is configured. An
    empty platform result is authoritative - Google is only consulted
    when the platform call itself fails (daemon absent/unreachable).

    Args:
        screenshot_b64: Base64 encoded screenshot image.
        client: Optional persistent httpx.AsyncClient.

    Returns:
        A list of dictionaries containing detected text and position vertices.
    """
    provider = _ocr_provider()
    if provider == "apple":
        logger.warning(
            "ARTEMIS_OCR_PROVIDER=apple is deprecated - the standalone "
            "worker was removed; using the platform vision-hybrid route"
        )
    if provider in ("auto", "platform", "apple"):
        try:
            return await _run_platform_ocr(screenshot_b64, client)
        except Exception as e:
            if provider in ("platform", "apple"):
                logger.warning(f"platform vision-hybrid OCR failed: {e}")
                return []
            logger.debug(f"platform vision-hybrid OCR unreachable: {e}")

    api_key = _google_vision_key()
    if not api_key:
        return []

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "requests": [
            {
                "image": {"content": screenshot_b64},
                "features": [{"type": "TEXT_DETECTION"}],
            }
        ]
    }

    active_client = client if client is not None else get_http_client()
    response = await active_client.post(url, json=data, headers=headers, timeout=30.0)
    return _parse_ocr_response(response)


def _parse_ocr_response(response: httpx.Response) -> list[dict[str, Any]]:
    if response.status_code == 200:
        res_json = response.json()
        responses = res_json.get("responses", [])
        if responses and "textAnnotations" in responses[0]:
            annotations = responses[0]["textAnnotations"]
            results = []
            # Skip the first annotation (index 0) as it is the full-screen combined text
            for ann in annotations[1:]:
                desc = ann.get("description", "")
                vertices = ann.get("boundingPoly", {}).get("vertices", [])
                results.append({"text": desc, "position": vertices})
            return results
        else:
            return []
    else:
        raise Exception(f"Vision API returned status {response.status_code}: {response.text}")
