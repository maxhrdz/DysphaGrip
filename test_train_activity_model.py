import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from activity_model import ActivityModel, extract_activity_features, extract_voice_features
from label_capture import LabelCapture
from train_activity_model import (
    classification_summary, evaluate_activity_session, read_examples, train_activity_sessions,
)
from validation import ExperimentLog


class ActivityTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_session(self, day, categories, times=None):
        folder = self.root / f"take_{day}"
        log = ExperimentLog(folder)
        log.started_at = f"2026-01-{day:02d}T12:00:00+00:00"
        log.started = 0
        capture = LabelCapture(folder, 1000)
        marks = times or [5 * i + 2 for i in range(len(categories))]
        for when, category in zip(marks, categories):
            label = log.add_label(category, when)
            capture.mark(label)
            signal = np.sin(np.arange(1000) * (0.2 if category == "speaking" else 0.4)) / 3
            capture.add_audio(signal, when - 1)
            for offset in (-0.9, -0.6, -0.3):
                capture.add_imu(when + offset, [1, 2, 3, offset, 5, 6])
            capture.flush(when + 3)
        log.finish([], marks[-1] + 3)
        return folder

    def test_file_features_match_real_time_and_missing_imu_skipped(self):
        session = self.make_session(1, ["swallowing_water", "speaking"])
        motion_file = session / "captures" / "mark_0002" / "motion.csv"
        motion_file.write_text(motion_file.read_text().splitlines()[0] + "\n")
        examples, skipped = read_examples(session)
        self.assertEqual([e.activity for e in examples], ["swallowing"])
        self.assertEqual(skipped[0]["mark"], 2)
        signal = np.sin(np.arange(1000) * 0.4) / 3
        rows = [[t, 1, 2, 3, t, 5, 6] for t in (-0.9, -0.6, -0.3)]
        expected = extract_activity_features(signal.astype(np.float32), 1000, rows, 0)
        np.testing.assert_allclose(examples[0].features, expected, atol=1e-5)
        np.testing.assert_allclose(examples[0].voice, extract_voice_features(signal.astype(np.float32), 1000),
                                   atol=1e-5)

    def test_uncertain_included_in_accuracy_and_missing_classes(self):
        rows = [{"expected": "rest", "predicted": "uncertain"},
                {"expected": "speaking", "predicted": "speaking"}]
        result = classification_summary(rows)
        self.assertEqual(result["accuracy_including_uncertain"], 0.5)
        self.assertEqual(result["coverage"], 0.5)
        self.assertEqual(result["per_class"]["rest"]["confusion"]["uncertain"], 1)
        self.assertIsNone(result["per_class"]["swallowing"]["recall"])

    def test_full_session_folds_chronology_and_future_evaluation(self):
        old = self.make_session(1, ["swallowing_saliva", "speaking"])
        latest = self.make_session(2, ["speaking", "swallowing_water", "rest", "swallowing_saliva", "rest", "rest"],
                                   times=[2, 7, 12, 17, 20, 25])
        model_path = self.root / "models" / "pilot.json"
        report_path = self.root / "reports" / "validation.json"
        report = train_activity_sessions([old, latest], model_path, report_path)
        self.assertEqual(report, json.loads(report_path.read_text()))
        for fold in report["folds"]:
            self.assertNotIn(fold["test_session"], fold["training_sessions"])
        self.assertEqual(report["folds"][1]["test_classes_absent_from_training"], ["rest"])
        self.assertEqual(report["out_of_session"]["examples"], 8)
        chronological = report["within_latest_session_chronological"]
        self.assertEqual(chronological["training_marks_in_latest"], [1, 2, 3, 4])
        self.assertEqual(chronological["excluded_near_split"], 1)
        self.assertEqual([d["mark"] for d in chronological["decisions"]], [6])
        model = ActivityModel.load(model_path)
        with self.assertRaisesRegex(ValueError, "participó"):
            evaluate_activity_session(model, old)
        fresh = self.make_session(3, ["rest", "speaking"])
        self.assertEqual(evaluate_activity_session(model, fresh)["metrics"]["examples"], 2)

    def test_specialist_preserves_primary_and_keeps_both_holdouts_separate(self):
        old = self.make_session(1, ["swallowing_saliva", "speaking", "coughing", "rest"])
        latest = self.make_session(2, ["speaking", "rest", "coughing", "speaking", "coughing", "rest"])
        source = self.root / "pilot.json"
        train_activity_sessions([old], source, self.root / "old-report.json")
        original = source.read_bytes()
        path = self.root / "voice.json"
        result = train_activity_sessions([old, latest], path, self.root / "new-report.json", source)
        self.assertEqual(source.read_bytes(), original)
        pilot, trained = ActivityModel.load(source), ActivityModel.load(path)
        np.testing.assert_array_equal(trained.weights, pilot.weights)
        self.assertEqual(trained.primary_training_sessions, pilot.training_sessions)
        self.assertEqual(len(trained.voice_training_sessions), 2)
        entire = result["published_vs_candidate_latest_session"]
        self.assertNotIn(entire["test_session"], entire["training_sessions"])
        self.assertEqual(entire["candidate"]["metrics"]["examples"], 6)
        tail = result["within_latest_session_chronological"]
        self.assertEqual(tail["training_marks_in_latest"], [1, 2, 3, 4])
        self.assertEqual([d["mark"] for d in tail["candidate"]["decisions"]], [5, 6])
        self.assertEqual(tail["candidate"]["metrics"]["examples"], tail["baseline"]["metrics"]["examples"])
        self.assertEqual(result["paired_baseline_out_of_session"]["examples"], result["out_of_session"]["examples"])


if __name__ == "__main__":
    unittest.main()
