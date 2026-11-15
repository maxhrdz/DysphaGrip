import csv
import tempfile
import unittest
from pathlib import Path

from validation import ExperimentLog, match_events, score_matches


def prediction(event_id, time):
    return {"id": event_id, "time": time}


def label(label_id, time, category):
    return {"id": label_id, "time": time, "category": category}


class MatchingTests(unittest.TestCase):
    def test_closest_pair_within_tolerance_is_a_true_positive(self):
        matched = match_events(
            [prediction(1, 10.2)],
            [label(1, 10.0, "swallowing_water")],
            tolerance=1.0,
        )
        self.assertEqual(len(matched["true_positives"]), 1)
        self.assertAlmostEqual(matched["true_positives"][0]["latency"], 0.2)
        self.assertEqual(matched["false_positives"], [])
        self.assertEqual(matched["false_negatives"], [])

    def test_gap_beyond_tolerance_is_not_a_match(self):
        matched = match_events(
            [prediction(1, 0.0)],
            [label(1, 5.0, "swallowing_saliva")],
            tolerance=1.0,
        )
        self.assertEqual(matched["true_positives"], [])
        self.assertEqual(len(matched["false_positives"]), 1)
        self.assertEqual(len(matched["false_negatives"]), 1)

    def test_one_label_accepts_only_one_prediction(self):
        matched = match_events(
            [prediction(1, 10.0), prediction(2, 10.2)],
            [label(1, 10.1, "swallowing_saliva")],
            tolerance=1.0,
        )
        self.assertEqual(len(matched["true_positives"]), 1)
        self.assertEqual(len(matched["false_positives"]), 1)
        self.assertEqual(matched["true_positives"][0]["prediction_id"], 1)

    def test_one_prediction_accepts_only_one_label(self):
        matched = match_events(
            [prediction(1, 2.1)],
            [
                label(1, 0.0, "swallowing_water"),
                label(2, 2.0, "swallowing_water"),
            ],
            tolerance=1.5,
        )
        self.assertEqual(len(matched["true_positives"]), 1)
        self.assertEqual(matched["true_positives"][0]["label_id"], 2)
        self.assertEqual(len(matched["false_negatives"]), 1)

    def test_detection_aligned_with_a_control_is_a_false_positive(self):
        matched = match_events(
            [prediction(1, 0.9)],
            [
                label(1, 0.0, "swallowing_saliva"),
                label(2, 1.0, "coughing"),
            ],
            tolerance=1.5,
        )
        self.assertEqual(matched["true_positives"], [])
        self.assertEqual(len(matched["false_positives"]), 1)
        self.assertEqual(matched["false_positives"][0]["kind"], "false_positive_control")
        self.assertEqual(len(matched["false_negatives"]), 1)

    def test_unmatched_control_is_not_a_false_negative(self):
        matched = match_events(
            [],
            [label(1, 3.0, "speaking"), label(2, 4.0, "head_movement")],
            tolerance=1.0,
        )
        self.assertEqual(matched["false_negatives"], [])
        self.assertEqual(len(matched["controls_without_detection"]), 2)


class MetricTests(unittest.TestCase):
    def test_precision_recall_f1_and_rate(self):
        matched = match_events(
            [prediction(1, 1.0), prediction(2, 2.0), prediction(3, 9.0)],
            [
                label(1, 1.1, "swallowing_saliva"),
                label(2, 2.2, "swallowing_water"),
                label(3, 5.0, "swallowing_water"),
            ],
            tolerance=0.5,
        )
        scored = score_matches(matched, duration_seconds=120.0)
        self.assertEqual(scored["true_positives"], 2)
        self.assertEqual(scored["false_positives"], 1)
        self.assertEqual(scored["false_negatives"], 1)
        self.assertAlmostEqual(scored["precision"], 2 / 3)
        self.assertAlmostEqual(scored["recall"], 2 / 3)
        self.assertAlmostEqual(scored["f1"], 2 / 3)
        self.assertAlmostEqual(scored["false_positives_per_minute"], 0.5)
        self.assertAlmostEqual(scored["mean_latency_seconds"], (-0.1 + -0.2) / 2)

    def test_undefined_scores_stay_blank(self):
        scored = score_matches(
            match_events([], [], tolerance=1.0),
            duration_seconds=30.0,
        )
        self.assertIsNone(scored["precision"])
        self.assertIsNone(scored["recall"])
        self.assertIsNone(scored["f1"])
        self.assertIsNone(scored["mean_latency_seconds"])
        self.assertEqual(scored["false_positives_per_minute"], 0.0)

    def test_zero_duration_does_not_divide(self):
        matched = match_events(
            [prediction(1, 1.0)],
            [],
            tolerance=1.0,
        )
        scored = score_matches(matched, duration_seconds=0.0)
        self.assertIsNone(scored["false_positives_per_minute"])
        self.assertEqual(scored["false_positives"], 1)


class ReportTests(unittest.TestCase):
    def test_finish_writes_real_rows_only(self):
        folder = Path(tempfile.mkdtemp())
        log = ExperimentLog(folder, tolerance=1.0)
        log.add_label("swallowing_water", when=log.started + 2.0)
        log.add_label("coughing", when=log.started + 4.0)
        summary = log.finish(
            [{
                "event_id": 7,
                "monotonic": log.started + 2.2,
                "timestamp": "2026-11-15T00:00:02.200-08:00",
                "rms": 0.02,
                "imu_activity": 1.2,
            }],
            ended=log.started + 60.0,
        )
        self.assertEqual(summary["true_positives"], 1)
        self.assertEqual(summary["false_positives"], 0)
        self.assertEqual(summary["false_negatives"], 0)
        self.assertEqual(summary["controls_without_detection"], 1)
        with (folder / "predictions.csv").open(newline="", encoding="utf-8") as file:
            predictions = list(csv.DictReader(file))
        self.assertEqual(predictions, [{
            "event_id": "7",
            "timestamp": "2026-11-15T00:00:02.200-08:00",
            "monotonic": f"{log.started + 2.2:.6f}",
            "audio_rms": "0.02",
            "imu_activity": "1.2",
        }])
        with (folder / "summary.csv").open(newline="", encoding="utf-8") as file:
            row = next(csv.DictReader(file))
        self.assertEqual(row["true_positives"], "1")
        self.assertIn("not an aspiration test", row["disclaimer"])


if __name__ == "__main__":
    unittest.main()
