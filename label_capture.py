"""Save microphone and motion measurements around manual experiment labels."""

import csv
import json
import tempfile
import threading
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SECONDS_BEFORE = 2.0
SECONDS_AFTER = 2.0
# Keep some overlap for labels arriving after the screen or disk was busy.
HISTORY_SECONDS = SECONDS_BEFORE + 1.0
# A block can reach the application after its last ADC sample was acquired.
DELIVERY_GRACE = 0.5


@dataclass
class _Window:
    label: dict
    start: float
    end: float
    audio: list
    motion: list


class LabelCapture:
    def __init__(self, directory, sample_rate):
        self.directory = Path(directory) / "captures"
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.audio = deque()
        self.motion = deque()
        self.pending = {}
        self.lock = threading.Lock()
        self.writer_lock = threading.Lock()

    def add_audio(self, samples, start):
        """Accept a mono block and monotonic acquisition time of its first sample."""
        samples = np.array(samples, dtype=np.float32, copy=True)
        if samples.ndim != 1:
            raise ValueError("audio must be mono")
        if samples.size == 0:
            return
        end = start + len(samples) / self.sample_rate
        block = (start, end, samples)
        with self.lock:
            self.audio.append(block)
            while self.audio and self.audio[0][1] <= end - HISTORY_SECONDS:
                self.audio.popleft()
            for window in self.pending.values():
                if start < window.end and end > window.start:
                    window.audio.append(block)

    def add_imu(self, stamp, values):
        """Accept the six parsed Arduino axes, timestamped on receipt."""
        row = (stamp, *values)
        with self.lock:
            self.motion.append(row)
            while self.motion and self.motion[0][0] < stamp - HISTORY_SECONDS:
                self.motion.popleft()
            for window in self.pending.values():
                if window.start <= stamp < window.end:
                    window.motion.append(row)

    def mark(self, label):
        """Register the same label ID and monotonic time used by the experiment."""
        start = label["time"] - SECONDS_BEFORE
        end = label["time"] + SECONDS_AFTER
        with self.lock:
            self.pending[label["id"]] = _Window(
                dict(label), start, end,
                [block for block in self.audio if block[0] < end and block[1] > start],
                [row for row in self.motion if start <= row[0] < end],
            )

    def flush(self, now):
        self._save_ready(now, closing=False)

    def finish(self, now):
        """Call after stopping both inputs to save even the last partial capture."""
        self._save_ready(now, closing=True)

    def _save_ready(self, now, closing):
        now = float(now)
        with self.writer_lock:
            with self.lock:
                ready = []
                for window in self.pending.values():
                    if closing or now >= window.end + DELIVERY_GRACE:
                        ready.append(_Window(
                            window.label, window.start, window.end,
                            window.audio.copy(), window.motion.copy(),
                        ))
            for window in ready:
                self._save(window, now)
                with self.lock:
                    del self.pending[window.label["id"]]

    def _save(self, window, now):
        label = window.label
        mark = label["time"]
        frame_count = round((SECONDS_BEFORE + SECONDS_AFTER) * self.sample_rate)
        stop = min(frame_count, max(0, round((now - window.start) * self.sample_rate)))
        audio = np.zeros(frame_count, dtype=np.float32)
        present = np.zeros(frame_count, dtype=bool)
        for start, _, samples in window.audio:
            first = round((start - window.start) * self.sample_rate)
            left = max(0, first)
            right = min(stop, first + len(samples))
            if left < right:
                audio[left:right] = samples[left - first:right - first]
                present[left:right] = True

        # Keep the true offsets. Fill only internal gaps with zeros in the WAV.
        indices = np.flatnonzero(present)
        left = int(indices[0]) if indices.size else 0
        right = int(indices[-1]) + 1 if indices.size else 0
        missing_edges = np.flatnonzero(np.diff(present, prepend=True, append=True))
        missing_ranges = [
            [int(a) / self.sample_rate - SECONDS_BEFORE,
             int(b) / self.sample_rate - SECONDS_BEFORE]
            for a, b in missing_edges.reshape(-1, 2)
        ]
        motion = [row for row in window.motion if row[0] <= now]
        metadata = {
            "label_id": label["id"],
            "category": label["category"],
            "label_timestamp": label["timestamp"],
            "label_monotonic": mark,
            "requested_seconds_before": SECONDS_BEFORE,
            "requested_seconds_after": SECONDS_AFTER,
            "closed_before_window_end": bool(now < window.end),
            "sample_rate_hz": self.sample_rate,
            "audio_recorded_frames": int(indices.size),
            "wav_frames": right - left,
            "wav_start_seconds_from_mark": left / self.sample_rate - SECONDS_BEFORE if indices.size else None,
            "audio_missing_intervals_seconds_from_mark": missing_ranges,
            "motion_samples": len(motion),
            "first_motion_seconds_from_mark": motion[0][0] - mark if motion else None,
            "last_motion_seconds_from_mark": motion[-1][0] - mark if motion else None,
            "max_motion_interval_seconds": max(
                (b[0] - a[0] for a, b in zip(motion, motion[1:])), default=None,
            ),
        }

        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"mark_{label['id']:04d}"
        with tempfile.TemporaryDirectory(dir=self.directory) as temp:
            folder = Path(temp) / target.name
            folder.mkdir()
            with wave.open(str(folder / "audio.wav"), "wb") as wav:
                wav.setparams((1, 4, self.sample_rate, 0, "NONE", "not compressed"))
                scaled = np.rint(audio[left:right].astype(np.float64) * (2 ** 31))
                wav.writeframes(np.clip(scaled, -(2 ** 31), 2 ** 31 - 1).astype("<i4").tobytes())
            with (folder / "motion.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    "seconds_from_mark", "monotonic", "tilt_x", "tilt_y", "tilt_z",
                    "turn_x", "turn_y", "turn_z",
                ])
                for stamp, *values in motion:
                    writer.writerow([f"{stamp - mark:.6f}", f"{stamp:.6f}", *values])
            (folder / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
            )
            folder.rename(target)
