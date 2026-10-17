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

"""Screen observation: capture, fuse, index, and persist -- without LangGraph State.

This is the act-then-observe half that used to be welded into every executor action
(``capture_screenshot_and_parse_ui``). Split out so that actuation and observation are
separate calls: the Flash adapter observes after each action, while the
Validator -- which runs its own UI-change polling -- simply never calls it.
"""

import asyncio
import base64
import time

from artemis.config import get_temp_dir
from artemis.context import ArtemisContext
from artemis.controllers.unified_controller import UnifiedMobileController
from artemis.mcp.action_types import ActionCode, ObserveResult
from artemis.utils.ocr_xml_fusion import fuse_ocr_with_xml
from artemis.utils.visualization import (
    format_minimal_list_with_elements,
    get_center_coordinates,
    parse_bounds,
)
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["observe"]


_CONTROL_CLASSES = ("switch", "checkbox", "radiobutton", "togglebutton")
_STATE_ATTRS = ("checked", "enabled", "selected", "clickable", "scrollable", "package")


def _control_only_elements(fused_xml: list[dict], elements: list[dict]) -> list[dict]:
    """Metadata for stateful controls the compact list never renders.

    Checkable/toggle nodes carry verified device state but usually no text,
    so ``format_minimal_list_with_elements`` skips them — a bare Settings
    switch is invisible to code consumers as well as to the model. Appended
    AFTER the labeled sequence: their ``index`` values continue the
    sequence so positional index resolution for listed elements is
    unchanged, while signatures and goal invariants can see the state.
    No text line is produced — the model-facing list is unchanged.
    """
    seen = {
        (el.get("resource_id"), el.get("class"), tuple(el.get("bounds") or ())) for el in elements
    }
    controls: list[dict] = []
    index = len(elements) + 1
    for node in fused_xml:
        if node.get("ocr_elements"):
            continue
        cls = str(node.get("class") or "")
        if not (node.get("checkable") == "true" or cls.lower().endswith(_CONTROL_CLASSES)):
            continue
        bounds = parse_bounds(node.get("bounds"))
        key = (node.get("resource-id"), node.get("class"), bounds)
        if not bounds or key in seen:
            continue
        seen.add(key)
        left, top, right, bottom = bounds
        cx, cy = get_center_coordinates(left, top, right, bottom)
        element = {
            "index": index,
            "center": [cx, cy],
            "text": str(node.get("text") or node.get("content-desc") or "").strip(),
            "bounds": [left, top, right, bottom],
            "class": node.get("class"),
            "resource_id": node.get("resource-id"),
            "is_ocr": False,
            "is_control_only": True,
        }
        for attr in _STATE_ATTRS:
            if node.get(attr) is not None:
                element[attr] = node.get(attr)
        controls.append(element)
        index += 1
    return controls


async def observe(
    ctx: ArtemisContext,
    controller: UnifiedMobileController,
    settle_ms: int = 400,
    ocr_results: list | None = None,
) -> tuple[ObserveResult, bytes | None]:
    """Captures and indexes the current screen.

    Returns the structured observation plus the raw screenshot bytes (an in-process
    convenience; the serialized ``ObserveResult`` carries only the persisted path).

    Mirrors the historical ``capture_screenshot_and_parse_ui`` behavior exactly:
    a settling delay, screen capture, XML fusion and element indexing, persistence via
    the data engine with a temp-file fallback -- and the same failure semantics: a
    failed capture yields ``ok=False``; a failed hierarchy parse still returns the
    screenshot with ``hierarchy_ok=False``.
    """
    try:
        if settle_ms > 0:
            logger.info(f"Delaying {settle_ms / 1000:.1f}s before capturing screen state...")
            await asyncio.sleep(settle_ms / 1000.0)
        else:
            logger.info("Retrieving screen data directly (skip settling)...")
        device_data = await controller.get_screen_data()
        screenshot_bytes = base64.b64decode(device_data.base64)
    except Exception as e:
        logger.error(f"Failed to capture screen state: {e}")
        return (
            ObserveResult(
                ok=False,
                code=ActionCode.DEVICE_ERROR,
                message=f"Failed to capture screen state: {e}",
                hierarchy_ok=False,
            ),
            None,
        )

    xml_hierarchy = None
    ocr_results = ocr_results or []
    elements: list[dict] = []
    hierarchy_ok = True
    width = height = None
    try:
        xml_hierarchy = device_data.elements
        width = device_data.width or 1080
        height = device_data.height or 2400
        if ctx.device:
            ctx.device.device_width = width
            ctx.device.device_height = height

        fused_xml = fuse_ocr_with_xml(xml_hierarchy, ocr_results)
        minimal_list_str, elements, _ = format_minimal_list_with_elements(fused_xml, width, height)
        elements = elements + _control_only_elements(fused_xml, elements)
    except Exception as e:
        logger.error(f"Failed to fetch or fuse UI hierarchy: {e}")
        minimal_list_str = "Not available due to hierarchy error."
        hierarchy_ok = False

    screenshot_path = None
    if ctx.data_engine:
        try:
            image_name = ctx.data_engine.get_or_create_image(
                screenshot_bytes,
                ui_tree=xml_hierarchy,
                ocr_result=ocr_results,
            )
            screenshot_path = str(ctx.data_engine.get_image_path(image_name))
        except Exception as e:
            logger.error(f"Failed to save image in DataEngine: {e}")

    if not screenshot_path:
        try:
            temp_dir = get_temp_dir("screenshots")
            temp_file = temp_dir / f"screenshot_{int(time.time())}.jpg"
            temp_file.write_bytes(screenshot_bytes)
            screenshot_path = str(temp_file)
        except Exception as e:
            logger.error(f"Failed to save fallback screenshot: {e}")

    from pathlib import Path

    return (
        ObserveResult(
            ok=True,
            message="Screen observed.",
            screenshot_path=screenshot_path,
            image_name=Path(screenshot_path).stem if screenshot_path else None,
            width=width,
            height=height,
            elements_text=minimal_list_str,
            elements=elements,
            hierarchy_ok=hierarchy_ok,
        ),
        screenshot_bytes,
    )
