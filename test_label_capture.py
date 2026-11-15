import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from label_capture import LabelCapture


def label(number, moment):
    return {"id": number, "category": "head_movement", "time": moment,
            "timestamp": "2026-11-15T01:35:14.000-08:00"}


def read_clip(directory, number):
    folder = Path(directory) / "captures" / f"mark_{number:04d}"
    meta = json.loads((folder / "metadata.json").read_text())
    with wave.open(str(folder / "audio.wav"), "rb") as wav:
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i4") / 2**31
    with (folder / "motion.csv").open(newline="") as file:
        motion = list(csv.DictReader(file))
    return meta, data, motion


class LabelCaptureTests(unittest.TestCase):
    def test_overlapping_marks_save_complete_signals_without_detections(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = LabelCapture(directory, sample_rate=100)
            signal = np.arange(600, dtype=np.float32) / 1000000
            for n in range(6):
                capture.add_audio(signal[n * 100:(n + 1) * 100], start=n)
                capture.add_imu(n, (n, 1, 2, 3, 4, 5))
                if n == 2:
                    capture.mark(label(1, 2.5))
                if n == 3:
                    capture.mark(label(2, 3.5))
                capture.flush(n)
            capture.finish(6)
            for number, start in [(1, 50), (2, 150)]:
                meta, audio, motion = read_clip(directory, number)
                np.testing.assert_allclose(audio, signal[start:start + 400], atol=3e-10)
                self.assertEqual(meta["audio_missing_intervals_seconds_from_mark"], [])
                self.assertEqual(meta["wav_start_seconds_from_mark"], -2)
                self.assertEqual([float(row["seconds_from_mark"]) for row in motion],
                                 [-1.5, -0.5, 0.5, 1.5])
                self.assertTrue(all(row["turn_z"] == "5" for row in motion))

    def test_close_early_preserves_internal_gaps_and_trims_missing_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = LabelCapture(directory, sample_rate=10)
            capture.add_audio(np.full(5, 0.5), 9.0)
            capture.mark(label(1, 10.0))
            capture.add_audio(np.full(10, -0.25), 10.0)
            capture.finish(10.8)
            meta, audio, motion = read_clip(directory, 1)
            self.assertTrue(meta["closed_before_window_end"])
            self.assertEqual(meta["audio_recorded_frames"], 13)
            self.assertEqual(meta["wav_start_seconds_from_mark"], -1.0)
            np.testing.assert_allclose(meta["audio_missing_intervals_seconds_from_mark"],
                                       [[-2, -1], [-0.5, 0], [0.8, 2]])
            np.testing.assert_array_equal(audio, [.5] * 5 + [0] * 5 + [-.25] * 8)
            self.assertEqual(motion, [])
            self.assertIsNone(meta["first_motion_seconds_from_mark"])

    def test_silent_and_missing_channels_remain_distinguishable(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = LabelCapture(directory, sample_rate=8)
            capture.mark(label(1, 3.0))
            capture.add_imu(4, (0, 0, 0, 0, 0, 0))
            capture.flush(5.01)
            # The last block may reach the application after the window end.
            self.assertFalse((Path(directory) / "captures/mark_0001").exists())
            capture.add_audio(np.zeros(8), 4.0)
            capture.flush(6)
            capture.mark(label(2, 10.0))
            capture.finish(13)
            first, data, motion = read_clip(directory, 1)
            self.assertEqual(first["audio_recorded_frames"], 8)
            self.assertTrue(np.all(data == 0))
            self.assertEqual(len(motion), 1)
            missing, data, _ = read_clip(directory, 2)
            self.assertEqual(data.size, 0)
            self.assertIsNone(missing["wav_start_seconds_from_mark"])
            self.assertEqual(missing["audio_missing_intervals_seconds_from_mark"], [[-2, 2]])


if __name__ == "__main__":
    unittest.main()
