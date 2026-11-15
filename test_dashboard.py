import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

import sensor_fusion_live as dashboard
from event_detector_live import SwallowEventDetector


def ready_detector(csv_path):
    detector = SwallowEventDetector(
        calibration_seconds=0.0,
        csv_path=csv_path,
    )
    moment = detector.start_time
    quiet = np.full(64, 0.001, dtype=np.float32)
    for _ in range(25):
        detector.on_audio(quiet, now=moment)
        detector.on_imu((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=moment)
    return detector, moment


class DashboardStatusTests(unittest.TestCase):
    def test_history_shows_detector_events_and_connection(self):
        folder = tempfile.mkdtemp()
        detector, moment = ready_detector(Path(folder) / "events.csv")
        when = moment + 0.2
        detector.on_audio(np.full(64, 0.05, dtype=np.float32), now=when)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=when)

        view = dashboard.detector_view(detector)
        lines = "\n".join(dashboard.status_lines(
            when, view, last_audio_at=when, last_imu_at=when,
        ))
        self.assertEqual(view["event_count"], 1)
        self.assertIn("Microphone    Connected", lines)
        self.assertIn("MPU6050       Connected", lines)
        self.assertIn("Detector      Detection active", lines)
        self.assertIn("Possible Swallow Event Detected", lines)
        self.assertNotIn("Safe Swallow", lines)
        self.assertNotIn("Aspiration", lines)

    def test_waiting_sensors_and_calibration_are_labeled(self):
        view = {
            "events": [],
            "ready": False,
            "event_count": 0,
            "started": 100.0,
            "calibration_seconds": 5.0,
        }
        lines = "\n".join(dashboard.status_lines(
            101.0, view, last_audio_at=0.0, last_imu_at=0.0,
        ))
        self.assertIn("Microphone    Waiting", lines)
        self.assertIn("MPU6050       Waiting", lines)
        self.assertIn("Calibrating (4s)", lines)
        self.assertIn("No candidate events yet.", lines)

    def test_redraw_marks_a_real_detector_event(self):
        folder = tempfile.mkdtemp()
        detector, moment = ready_detector(Path(folder) / "events.csv")
        when = moment + 0.2
        detector.on_audio(np.full(64, 0.05, dtype=np.float32), now=when)
        detector.on_imu((1.0, 0.0, 1.0), (4.0, 0.0, 0.0), now=when)

        samples = np.sin(np.linspace(0, 40, 48000, endpoint=False)).astype(np.float32) * 0.2
        imu_rows = [
            (when - 1.0 + index * 0.05, 0.1, 0.2, 0.9, 1.0, -0.2, 0.4)
            for index in range(40)
        ]
        board = dashboard.create_dashboard()
        dashboard.redraw(
            board, samples, imu_rows, dashboard.detector_view(detector),
            last_audio_at=when, last_imu_at=when, now=when + 0.5,
        )
        self.assertEqual(board["counter_text"].get_text(), "1")
        self.assertGreaterEqual(len(board["event_artists"]), 3)
        image = Path("/tmp/dysphagrip_layout.png")
        board["fig"].savefig(image, dpi=110, facecolor=board["fig"].get_facecolor())
        self.assertGreater(image.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
