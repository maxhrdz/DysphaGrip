import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from motion_cough import FEATURE_NAMES, MotionCoughSpecialist
from train_motion_cough import Window, compare_windows, train_and_compare


class MotionCoughTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "primary.json"
        self.source.write_text("frozen")
        self.model = self.root / "specialist.json"
        self.report = self.root / "report.json"

    def make_source(self, day, rows):
        session = self.root / f"take{day}"
        session.mkdir()
        identity = f"2026-01-{day:02d}T12:00:00+00:00"
        with (session / "summary.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["started_at", "tolerance_seconds"])
            writer.writerow([identity, 1.5])
        windows = []
        for i, (when, kind) in enumerate(rows, 1):
            scores = {"head_movement": .7, "coughing": .2, "speaking": .1}
            x = np.full(len(FEATURE_NAMES), 1.0 if kind == "coughing" else -1.0)
            windows.append(Window(i, when, kind, kind, scores, "head_movement", .7, x))
        return session, identity, windows

    def test_reservations_no_leakage_and_serialized_final_counts(self):
        before, before_id, older = self.make_source(1, [(1, "coughing"), (8, "head_movement")])
        latest, latest_id, fresh = self.make_source(2, [
            (3, "coughing"), (6, "head_movement"), (8, "coughing"),
            (11, "head_movement"), (16, "coughing"), (20, "speaking")])
        future, _, _ = self.make_source(3, [])
        with patch("train_motion_cough.ActivityModel.load") as load, patch("train_motion_cough.read_windows") as read:
            load.return_value.training_sessions = [before_id]
            read.side_effect = lambda _, session: ({before: older, latest: fresh}[session], [])
            result = train_and_compare(self.source, latest, 4, self.model, self.report)
            self.assertNotIn(future, [c.args[1] for c in read.call_args_list])
        self.assertEqual(result["latest_whole_session"]["training_sessions"], [before_id])
        self.assertEqual(result["latest_whole_session"]["test_marks"], [1, 2, 3, 4, 5, 6])
        block = result["latest_chronological_block"]
        self.assertEqual(block["training_marks_in_latest"], [1, 2])
        self.assertEqual(block["excluded_near_split"], [3])
        self.assertEqual(block["test_marks"], [4, 5, 6])
        self.assertEqual(block["training_examples"], {"coughing": 2, "head_movement": 2})
        self.assertEqual(result["final_model"]["examples"], {"coughing": 4, "head_movement": 3})
        self.assertEqual(set(MotionCoughSpecialist.load(self.model).training_sessions), {before_id, latest_id})
        self.assertEqual(self.source.read_text(), "frozen")

    def test_fallback_rows_kept_and_seen_session_rejected(self):
        latest, identity, windows = self.make_source(1, [(0, "head_movement"), (5, "coughing")])
        model = MotionCoughSpecialist.fit([w.features for w in windows],
                                          [w.activity for w in windows], sessions=[identity])
        windows[0].features = None
        report = compare_windows(model, [windows[0]])
        self.assertEqual(report["baseline"]["correct"], report["candidate"]["correct"])
        self.assertEqual(report["candidate"]["examples"], 1)
        with patch("train_motion_cough.ActivityModel.load") as load:
            load.return_value.training_sessions = [identity]
            with self.assertRaisesRegex(ValueError, "primario"):
                train_and_compare(self.source, latest, 2, self.model, self.report)


if __name__ == "__main__":
    unittest.main()
