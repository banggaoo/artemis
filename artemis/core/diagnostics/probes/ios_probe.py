# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""iOS Simulator Readiness Probe.

Informational only (``is_blocker = False``): iOS support is opt-in per task,
so a missing Xcode toolchain must never block the Android default path. The
probe reports whether this host could run an iOS Simulator task today —
macOS, Xcode 27+, and at least one available simulator.
"""

import asyncio
import re

from artemis.core.diagnostics.probes.base import BaseProbe
from artemis.core.diagnostics.schema import (
    ProbeAction,
    ProbeCategory,
    ProbeResult,
    ProbeStatus,
)
from artemis.drivers.ios.discovery import (
    list_ios_simulators,
    run_xcrun,
    simctl_available,
)


class IosSimulatorProbe(BaseProbe):
    """Reports iOS Simulator automation capability (Xcode 27+ on macOS)."""

    @property
    def probe_id(self) -> str:
        return "ios_simulators"

    @property
    def category(self) -> ProbeCategory:
        return ProbeCategory.DEVICE

    @property
    def is_blocker(self) -> bool:
        return False

    async def probe(self) -> ProbeResult:
        if not simctl_available():
            return ProbeResult(
                id=self.probe_id,
                category=self.category,
                title="iOS Simulator",
                status=ProbeStatus.SKIPPED,
                is_blocker=self.is_blocker,
                summary="Unavailable",
                description=(
                    "iOS Simulator automation requires macOS with Xcode 27+; "
                    "this host cannot enumerate simulators."
                ),
                metadata={"platform_supported": False},
            )

        xcode_ok = False
        xcode_version: str | None = None
        try:
            raw = await asyncio.wait_for(run_xcrun("xcodebuild", "-version"), timeout=15.0)
            match = re.search(rb"Xcode\s+(\d+(?:\.\d+)*)", raw)
            if match:
                xcode_version = match.group(1).decode()
                xcode_ok = int(xcode_version.split(".")[0]) >= 27
        except (OSError, RuntimeError, TimeoutError):
            xcode_version = None

        simulators = await list_ios_simulators()
        booted = [d for d in (simulators or []) if d.get("state") == "Booted"]
        metadata = {
            "platform_supported": True,
            "xcode_version": xcode_version,
            "xcode_27_or_newer": xcode_ok,
            "simulator_count": len(simulators or []),
            "booted_udids": [d.get("udid") for d in booted],
            "simulators": simulators or [],
        }

        if not xcode_ok:
            return ProbeResult(
                id=self.probe_id,
                category=self.category,
                title="iOS Simulator",
                status=ProbeStatus.WARN,
                is_blocker=self.is_blocker,
                summary="Xcode 27+ Required",
                description=(
                    "Native iOS automation requires Xcode 27 or later"
                    + (f" (found {xcode_version})." if xcode_version else "; none detected.")
                ),
                metadata=metadata,
                actions=[
                    ProbeAction(
                        action_type="hint",
                        label="Select Xcode 27",
                        payload="sudo xcode-select -s /Applications/Xcode.app/Contents/Developer",
                    ),
                ],
            )

        if not simulators:
            return ProbeResult(
                id=self.probe_id,
                category=self.category,
                title="iOS Simulator",
                status=ProbeStatus.WARN,
                is_blocker=self.is_blocker,
                summary="No Simulators",
                description=(
                    "Xcode is installed but no available iOS simulators were found. "
                    "Create one in Xcode > Settings > Platforms or via simctl."
                ),
                metadata=metadata,
            )

        summary = f"{len(booted)} Booted" if booted else f"{len(simulators)} Available"
        return ProbeResult(
            id=self.probe_id,
            category=self.category,
            title="iOS Simulator",
            status=ProbeStatus.PASS,
            is_blocker=self.is_blocker,
            summary=summary,
            description=(
                f"Xcode {xcode_version} with {len(simulators)} available iOS "
                f"simulator(s){' (' + str(len(booted)) + ' booted)' if booted else ''}. "
                "Submit a task with platform='ios' and a simulator UDID to use them."
            ),
            metadata=metadata,
        )
