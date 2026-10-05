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

"""Physical iOS device recording via polled ``devicectl`` screenshots.

``simctl io recordVideo`` is simulator-only, so physical capture polls
``xcrun devicectl device capture screenshot`` into timestamped PNG frames and
assembles each contiguous run of equal-sized frames into a CFR MP4 with the
ffconcat demuxer, preserving real per-frame timing. Segments roll on rotation
(dimension change), max duration, or capture failures. Physical capture has no
audio and a lower frame rate than simulator recording (~1-3 fps bounded by
devicectl round-trip time).
"""

import asyncio
from io import BytesIO
from pathlib import Path
import tempfile
import time
from typing import Any
from uuid import uuid4

from PIL import Image

from artemis.config.paths import get_temp_dir
from artemis.drivers.ios.discovery import run_xcrun
from artemis.drivers.ios.recording import IosRecordingSession
from artemis.utils.video import get_ffmpeg_path, probe_video_segment, write_recording_manifest
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

CAPTURE_TIMEOUT_SECONDS = 15.0
WATCHDOG_INTERVAL_SECONDS = 0.5
MAX_CONSECUTIVE_FAILURES = 5


async def _run_ffmpeg(arguments: list[str]) -> tuple[int, bytes]:
    """Seam for segment assembly so tests can intercept the ffmpeg process."""
    process = await asyncio.create_subprocess_exec(
        get_ffmpeg_path(),
        *arguments,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _stdout, stderr = await process.communicate()
    return process.returncode, stderr


class IosPhysicalRecordingSession(IosRecordingSession):
    """Recording session for polled physical-device frame capture."""

    poll_task: asyncio.Task | None = None
    frames_dir: Path | None = None
    frames: list[dict[str, Any]] = []
    frame_index: int = 0


class PhysicalIosRecorder:
    """Owns one screenshot-poll recording lifecycle for one physical UDID."""

    def __init__(self, device_id: str):
        self._device_id = device_id
        self._session: IosPhysicalRecordingSession | None = None
        self._output_dir: Path | None = None
        self._lock = asyncio.Lock()
        self._max_duration_seconds = 900
        self._consecutive_failures = 0

    @property
    def session(self) -> IosPhysicalRecordingSession | None:
        """The latest session, kept after stop/failure for error reporting."""
        return self._session

    def _segment_output_path(self, session: IosPhysicalRecordingSession, index: int) -> Path:
        assert self._output_dir is not None
        return self._output_dir / f"segment_{index:04d}.mp4"

    async def _capture_frame(self, session: IosPhysicalRecordingSession) -> None:
        """Append one timestamped screenshot frame to the open segment."""
        assert session.frames_dir is not None
        session.frame_index += 1
        path = session.frames_dir / f"frame_{session.frame_index:06d}.png"
        await run_xcrun(
            "devicectl",
            "device",
            "capture",
            "screenshot",
            "--device",
            self._device_id,
            "--destination",
            str(path),
            timeout=CAPTURE_TIMEOUT_SECONDS,
        )
        captured_at = time.monotonic()
        if not path.exists() or path.stat().st_size == 0:
            raise RuntimeError("devicectl produced no screenshot frame")
        with Image.open(path) as image:
            width, height = image.size
        if session.capture_width is None:
            session.capture_width, session.capture_height = width, height
            session.segment_started_monotonic = captured_at
            session.segment_started_at = time.time()
            if session.anchor_monotonic is None:
                session.anchor_monotonic = captured_at
                session.start_time = session.segment_started_at
        session.frames.append({"path": path, "at": captured_at})
        if (width, height) != (session.capture_width, session.capture_height):
            await self._roll(session, captured_at, "rotation")

    async def _poll_loop(self, session: IosPhysicalRecordingSession) -> None:
        """Capture frames back-to-back; devicectl's own latency paces the loop."""
        try:
            while session.is_active:
                failed = False
                async with self._lock:
                    if not session.is_active:
                        return
                    try:
                        await self._capture_frame(session)
                        self._consecutive_failures = 0
                    except asyncio.CancelledError:
                        raise
                    except (OSError, RuntimeError, TimeoutError) as exc:
                        failed = True
                        self._consecutive_failures += 1
                        session.errors.append(f"Frame capture failed: {exc}")
                        if self._consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            session.errors.append(
                                "Frame capture failure limit reached; session stopped"
                            )
                            session.is_active = False
                            return
                # Yield every iteration: callers and the watchdog share this loop.
                await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS if failed else 0)
        except asyncio.CancelledError:
            return

    async def _start_segment(self, session: IosPhysicalRecordingSession) -> None:
        assert self._output_dir is not None
        session.frames = []
        session.frame_index = 0
        session.capture_width = session.capture_height = None
        session.segment_started_monotonic = None
        session.frames_dir = self._output_dir / f"frames_{session.segment_index:04d}"
        session.frames_dir.mkdir(parents=True, exist_ok=True)

    async def _poll_first_frame(self, session: IosPhysicalRecordingSession) -> None:
        """Block until the first frame anchors the recording timeline."""
        deadline = time.monotonic() + CAPTURE_TIMEOUT_SECONDS * 2
        while session.anchor_monotonic is None:
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "Timed out waiting for the first devicectl screenshot frame."
                )
            if not session.is_active:
                raise RuntimeError(
                    session.errors[-1]
                    if session.errors
                    else "Frame capture stopped before the first frame."
                )
            await asyncio.sleep(0.05)

    def _seal_current_segment(
        self, session: IosPhysicalRecordingSession, end_monotonic: float
    ) -> dict[str, Any] | None:
        frames = [f for f in session.frames if f["at"] < end_monotonic]
        session.frames = [f for f in session.frames if f["at"] >= end_monotonic]
        if not frames:
            return None
        record = {
            "frames": frames,
            "output_path": self._segment_output_path(session, session.segment_index),
            "start": max(0.0, frames[0]["at"] - (session.anchor_monotonic or 0.0)),
            "end": max(0.0, end_monotonic - (session.anchor_monotonic or 0.0)),
            "width": session.capture_width,
            "height": session.capture_height,
            "generation": session.generation,
            "conversion_done": False,
            "conversion_error": None,
        }
        session.segments.append(record)
        session.sealed_until = max(session.sealed_until, record["end"])
        session.generation += 1
        session.segment_index += 1
        return record

    def _write_concat(self, session: IosPhysicalRecordingSession, record: dict[str, Any]) -> Path:
        """Write an ffconcat list preserving real capture timestamps."""
        frames = record["frames"]
        end_monotonic = record["end"] + (session.anchor_monotonic or 0.0)
        lines = ["ffconcat version 1.0"]
        for index, frame in enumerate(frames):
            lines.append(f"file '{Path(frame['path']).as_posix()}'")
            following = frames[index + 1]["at"] if index + 1 < len(frames) else end_monotonic
            lines.append(f"duration {max(0.001, following - frame['at']):.3f}")
        # The concat demuxer ignores the final frame's duration unless the file
        # line repeats, so list the last frame twice to keep its tail visible.
        lines.append(f"file '{Path(frames[-1]['path']).as_posix()}'")
        concat_path = Path(frames[0]["path"]).parent / "frames.txt"
        concat_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return concat_path

    async def _convert_record(self, session: IosPhysicalRecordingSession, record: dict[str, Any]) -> None:
        output = Path(record["output_path"])
        width = max(2, int(record["width"] or 1080)) // 2 * 2
        height = max(2, int(record["height"] or 1920)) // 2 * 2
        temporary = output.with_name(f"{output.stem}.part.mp4")
        if temporary.exists():
            temporary.unlink()
        try:
            concat = self._write_concat(session, record)
            returncode, stderr = await _run_ffmpeg(
                [
                    "-y",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(concat),
                    "-vf",
                    f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
                    f"force_divisible_by=2,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:"
                    "color=black,setsar=1",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
            )
            metadata = await probe_video_segment(temporary)
            valid = (
                returncode == 0
                and temporary.exists()
                and metadata.get("duration", 0) > 0
                and metadata.get("width", 0) > 0
                and metadata.get("height", 0) > 0
            )
            if valid:
                temporary.replace(output)
                record["conversion_done"] = True
                return
            record["conversion_error"] = "finalization produced no valid MP4"
            logger.error(
                f"Physical iOS recording finalization failed (code {returncode}): "
                f"{stderr.decode(errors='replace')[-2000:]}"
            )
        except OSError as exc:
            record["conversion_error"] = str(exc)
            logger.error(f"Physical iOS recording finalization failed: {exc}")
        if temporary.exists():
            temporary.unlink()
        session.errors.append(
            f"Segment {record['output_path']} failed finalization; frames kept"
        )

    async def _roll(
        self, session: IosPhysicalRecordingSession, end_monotonic: float, reason: str
    ) -> None:
        """Seal the current frames and open the next segment directory."""
        record = self._seal_current_segment(session, end_monotonic)
        leftover = session.frames
        if record is not None:
            logger.info(f"Rolling physical iOS recording segment after {reason}")
            session.conversion_tasks.append(
                asyncio.create_task(self._convert_record(session, record))
            )
        if session.is_active:
            await self._start_segment(session)
            for frame in leftover:
                target = session.frames_dir / Path(frame["path"]).name
                Path(frame["path"]).rename(target)
                frame["path"] = target
            session.frames = leftover
            if leftover:
                with Image.open(leftover[0]["path"]) as image:
                    session.capture_width, session.capture_height = image.size
                session.segment_started_monotonic = leftover[0]["at"]

    async def _watchdog(self, session: IosPhysicalRecordingSession) -> None:
        try:
            while True:
                await asyncio.sleep(WATCHDOG_INTERVAL_SECONDS)
                async with self._lock:
                    if not session.is_active:
                        return
                    if (
                        session.anchor_monotonic is not None
                        and time.monotonic() - session.anchor_monotonic
                        >= self._max_duration_seconds
                    ):
                        session.errors.append("Recording reached its duration limit")
                        session.is_active = False
                        return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.error(f"Physical iOS recording supervisor failed: {exc}")
            session.errors.append(f"recording supervisor failed: {exc}")

    async def start(
        self,
        output_dir: Path | None = None,
        max_duration_seconds: int = 900,
    ) -> IosPhysicalRecordingSession:
        async with self._lock:
            if self._session is not None and self._session.is_active:
                raise RuntimeError(
                    f"iOS recording is already active on {self._device_id}"
                )
            output = (
                Path(output_dir)
                if output_dir is not None
                else Path(
                    tempfile.mkdtemp(prefix="ios_recording_", dir=get_temp_dir("recordings"))
                )
            )
            output.mkdir(parents=True, exist_ok=True)
            self._output_dir = output
            self._max_duration_seconds = max_duration_seconds
            self._consecutive_failures = 0
            session = IosPhysicalRecordingSession(
                video_id=uuid4(),
                device_id=self._device_id,
                start_time=time.time(),
                is_active=True,
            )
            self._session = session
            await self._start_segment(session)
            session.poll_task = asyncio.create_task(self._poll_loop(session))
        # The first-frame wait must run outside the lock: the poll loop takes
        # the same lock to append captured frames.
        try:
            await self._poll_first_frame(session)
        except BaseException:
            session.is_active = False
            if session.poll_task and not session.poll_task.done():
                session.poll_task.cancel()
            raise
        session.watchdog_task = asyncio.create_task(self._watchdog(session))
        return session

    async def seal(self, through_time: float | None = None) -> None:
        """Seal the current segment so its final MP4 can be read safely."""
        async with self._lock:
            session = self._session
            if session is None or not session.is_active:
                return
            if session.anchor_monotonic is None:
                return
            if through_time is not None and session.sealed_until >= through_time:
                return
            end_monotonic = (
                session.anchor_monotonic + through_time
                if through_time is not None
                else time.monotonic()
            )
            end_monotonic = min(end_monotonic, time.monotonic())
            await self._roll(session, end_monotonic, "seal")

    async def _finalize(self, session: IosPhysicalRecordingSession) -> Path | None:
        session.is_active = False
        for task in (session.poll_task, session.watchdog_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    logger.debug(f"Physical iOS recording task ended with an error: {exc}")
        async with self._lock:
            record = self._seal_current_segment(session, time.monotonic())
            if record is not None:
                session.conversion_tasks.append(
                    asyncio.create_task(self._convert_record(session, record))
                )
        if session.conversion_tasks:
            await asyncio.gather(*session.conversion_tasks, return_exceptions=True)
        mp4_paths = [
            Path(record["output_path"])
            for record in session.segments
            if record.get("conversion_done")
            and Path(record["output_path"]).exists()
            and Path(record["output_path"]).stat().st_size > 0
        ]
        if not mp4_paths:
            session.errors.append("No finalized physical iOS recording segments")
            return None
        shift = 0.0
        if session.data_engine_start_time is not None:
            shift = session.start_time - session.data_engine_start_time
        offsets = {
            Path(record["output_path"]): max(0.0, float(record["start"]) + shift)
            for record in session.segments
            if record.get("conversion_done") and Path(record["output_path"]).exists()
        }
        output_dir = mp4_paths[0].parent
        manifest = await write_recording_manifest(output_dir, mp4_paths, offsets)
        if manifest is None:
            session.errors.append("Recording manifest has no valid segments")
            return None
        return mp4_paths[0]

    async def stop(self) -> Path | None:
        """Finalize all owned segments and write the recording manifest."""
        session = self._session
        if session is None:
            return None
        return await self._finalize(session)
