"""Exercise the training/holdout boundary with the recorder's on-disk formats."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_model import AudioModel
from label_capture import LabelCapture
from session_audio import SessionAudioRecorder
from train_audio_model import train_sessions, compare_session
from validation import ExperimentLog


class TrainAudioModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model_path = self.root / "model.json"
        self.train_dir = self.root / "train"
        self.rate = 1000
        log = ExperimentLog(self.train_dir)
        log.started = 100
        log.started_at = "2026-01-01T10:00:00.000+00:00"
        captures = LabelCapture(self.train_dir, self.rate)
        for when, category in ((103, "swallowing_saliva"), (109, "rest")):
            mark = log.add_label(category, when)
            captures.mark(mark)
            samples = np.zeros(4 * self.rate, dtype=np.float32)
            if category == "swallowing_saliva":
                samples[self.rate:2 * self.rate] = np.sin(np.arange(self.rate) * 0.3)
            captures.add_audio(samples, when - 2)
            captures.flush(when + 3)
        log.finish([], ended=115)

    def train_model(self):
        result = train_sessions([self.train_dir], self.model_path)
        self.assertEqual(result["examples_by_category"], {"swallowing_saliva": 1, "rest": 1})
        self.assertFalse(result["skipped"])
        return AudioModel.load(self.model_path)

    def test_requires_holdout_even_if_session_renamed(self):
        model = self.train_model()
        same_copy = self.root / "new_path"
        self.train_dir.rename(same_copy)
        with self.assertRaisesRegex(ValueError, "participó en el entrenamiento"):
            compare_session(model, same_copy)

    def test_rejects_incomplete_training_window(self):
        folder = self.train_dir / "captures" / "mark_0002"
        (folder / "audio.wav").unlink()
        with self.assertRaisesRegex(ValueError, "Faltan capturas"):
            train_sessions([self.train_dir], self.model_path)

    def test_holdout_only_scores_shared_timeline(self):
        model = self.train_model()
        session = self.root / "holdout"
        log = ExperimentLog(session, tolerance=1.5)
        log.started_at = "2026-01-01T11:00:00.000+00:00"
        log.started = 100
        for when in (102, 106, 109):
            log.add_label("swallowing_saliva", when)
        events = [{"event_id": i, "monotonic": when, "timestamp": "", "rms": 0, "imu_activity": 0}
                  for i, when in enumerate((102, 106, 111), 1)]
        log.finish(events, ended=112)
        recorder = SessionAudioRecorder(session, self.rate)
        quiet = np.zeros(5 * self.rate)
        recorder.add_audio(quiet, 100)
        recorder.add_audio(quiet, 107)
        recorder.close()
        result = compare_session(model, session)
        self.assertEqual(result["scored_seconds"], 8)
        self.assertEqual(result["excluded_labels"], 1)
        self.assertEqual(result["labels_by_category"], {"swallowing_saliva": 2})
        self.assertEqual(result["detector_actual"]["true_positives"], 1)
        self.assertEqual(result["detector_actual"]["false_negatives"], 1)
        self.assertEqual(result["detector_actual"]["false_positives_per_minute"], 7.5)
        self.assertEqual(result["modelo_audio"]["true_positives"], 0)
        self.assertEqual(result["modelo_audio"]["false_negatives"], 2)


if __name__ == "__main__":
    unittest.main()
