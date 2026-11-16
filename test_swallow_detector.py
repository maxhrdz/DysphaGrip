import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from event_detector_live import SwallowEventDetector


def make_detector(**kwargs):
    folder = tempfile.mkdtemp()
    path = Path(folder) / "events.csv"
    params = {
        "calibration_seconds": 0.0,
        "coincidence_seconds": 0.65,
        "cooldown_seconds": 1.5,
        "audio_multiplier": 2.5,
        "imu_multiplier": 3.0,
        "csv_path": path,
    }
    params.update(kwargs)
    detector = SwallowEventDetector(**params)
    return detector, path


def quiet_audio():
    return np.full(128, 0.001, dtype=np.float32)


def loud_audio():
    return np.full(128, 0.05, dtype=np.float32)


def calibrate(detector, samples=25):
    """Feed a still, quiet baseline and return the timestamp when it is ready."""
    moment = detector.start_time + detector.calibration_seconds
    tilt = (0.0, 0.0, 1.0)
    turn = (0.2, 0.0, 0.0)
    for _ in range(samples):
        detector.on_audio(quiet_audio(), now=moment)
        detector.on_imu(tilt, turn, now=moment)
    return moment


class CalibrationTests(unittest.TestCase):
    def test_calibration_waits_for_the_full_window(self):
        detector, _path = make_detector(calibration_seconds=5.0)
        early = detector.start_time + 1.0
        for _ in range(30):
            detector.on_audio(quiet_audio(), now=early)
            detector.on_imu((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=early)

        self.assertFalse(detector.ready)
        self.assertIsNone(detector.audio_threshold)

        ready_at = detector.start_time + 5.0
        detector.on_audio(quiet_audio(), now=ready_at)
        detector.on_imu((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=ready_at)
        self.assertTrue(detector.ready)
        self.assertGreaterEqual(
            detector.audio_threshold,
            0.001 * detector.audio_multiplier,
        )

    def test_baseline_noise_stays_below_the_threshold(self):
        detector, _path = make_detector()
        calibrate(detector)
        self.assertGreater(detector.audio_threshold, 0.001)
        self.assertGreater(detector.turn_threshold, 0.2)


class DetectionTests(unittest.TestCase):
    def test_restart_preserves_previous_events_and_appends_to_same_csv(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            path.touch()
            for _ in range(2):
                previous_rows = path.read_bytes()
                detector = SwallowEventDetector(calibration_seconds=0, csv_path=path)
                when = calibrate(detector) + 0.2
                detector.on_audio(loud_audio(), now=when)
                detector.on_imu((1, 0, 1), (4, 0, 0), now=when)
                self.assertTrue(path.read_bytes().startswith(previous_rows))

            with path.open(newline="", encoding="utf-8") as file:
                self.assertEqual(
                    [row["event_type"] for row in csv.DictReader(file)],
                    ["possible_swallow_event", "possible_swallow_event"],
                )

    def test_first_event_accepts_overlapping_signals_with_different_onsets(self):
        detector, _path = make_detector()
        when = calibrate(detector)
        for step in range(11):
            detector.on_audio(loud_audio(), now=when + step * 0.1)
        detector.on_imu((1, 0, 1), (4, 0, 0), now=when + 1)
        self.assertEqual(detector.event_count, 1)

    def test_coincident_audio_and_motion_records_one_event(self):
        detector, path = make_detector()
        moment = calibrate(detector)
        when = moment + 0.2
        detector.on_audio(loud_audio(), now=when)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=when)

        self.assertEqual(detector.event_count, 1)
        self.assertEqual(len(detector.events), 1)
        self.assertEqual(detector.events[0]["event_id"], 1)
        with path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_type"], "possible_swallow_event")
        self.assertEqual(rows[0]["event_id"], "1")
        self.assertIn("timestamp", rows[0])
        self.assertGreater(float(rows[0]["audio_rms"]), detector.audio_threshold)
        self.assertGreater(float(rows[0]["time_difference_seconds"]), -0.001)

    def test_cooldown_suppresses_a_second_event(self):
        detector, path = make_detector(cooldown_seconds=1.5)
        moment = calibrate(detector)
        first = moment + 0.2
        second = first + 0.4
        detector.on_audio(loud_audio(), now=first)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=first)
        detector.on_audio(quiet_audio(), now=first + 0.1)
        detector.on_imu((1.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=first + 0.1)
        detector.on_audio(loud_audio(), now=second)
        detector.on_imu((1.2, 0.2, 1.0), (5.0, 0.0, 0.0), now=second)

        self.assertEqual(detector.event_count, 1)
        with path.open(newline="", encoding="utf-8") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 1)

    def test_event_after_cooldown_is_counted(self):
        detector, _path = make_detector(cooldown_seconds=1.5)
        moment = calibrate(detector)
        first = moment + 0.2
        later = first + 1.6
        detector.on_audio(loud_audio(), now=first)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=first)
        detector.on_audio(quiet_audio(), now=first + 0.1)
        detector.on_imu((1.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=first + 0.1)
        detector.on_audio(loud_audio(), now=later)
        detector.on_imu((1.4, 0.0, 1.0), (6.0, 0.0, 0.0), now=later)
        self.assertEqual(detector.event_count, 2)


class FalsePositiveTests(unittest.TestCase):
    def test_continuous_activity_counts_once_across_cooldowns(self):
        detector, _path = make_detector()
        moment = calibrate(detector)
        for step in range(41):
            when = moment + step * 0.1
            detector.on_audio(loud_audio(), now=when)
            detector.on_imu((0.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=when)

        self.assertEqual(detector.event_count, 1)

    def test_audio_without_motion_is_not_an_event(self):
        detector, _path = make_detector()
        moment = calibrate(detector)
        when = moment + 0.2
        detector.on_audio(loud_audio(), now=when)
        detector.on_imu((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=when)
        self.assertEqual(detector.event_count, 0)

    def test_motion_without_audio_is_not_an_event(self):
        detector, _path = make_detector()
        moment = calibrate(detector)
        when = moment + 0.2
        detector.on_audio(quiet_audio(), now=when)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=when)
        self.assertEqual(detector.event_count, 0)

    def test_hits_outside_the_time_window_are_not_an_event(self):
        detector, _path = make_detector(coincidence_seconds=0.65)
        moment = calibrate(detector)
        detector.on_audio(loud_audio(), now=moment + 0.2)
        detector.on_imu(
            (1.0, 0.0, 1.0),
            (4.0, 0.0, 0.0),
            now=moment + 0.2 + 0.8,
        )
        self.assertEqual(detector.event_count, 0)

    def test_invalid_readings_are_ignored(self):
        detector, _path = make_detector()
        before_audio = len(detector.audio_calibration)
        before_tilt = len(detector.tilt_calibration)
        detector.on_audio(np.array([np.nan, 0.1], dtype=np.float64))
        detector.on_audio(np.array([], dtype=np.float32))
        detector.on_imu((0.0, 0.0), (0.0, 0.0, 0.0))
        detector.on_imu((np.nan, 0.0, 1.0), (0.0, 0.0, 0.0))
        self.assertEqual(len(detector.audio_calibration), before_audio)
        self.assertEqual(len(detector.tilt_calibration), before_tilt)
        self.assertEqual(detector.event_count, 0)


if __name__ == "__main__":
    unittest.main()
