import csv
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.backend_bases import KeyEvent, MouseEvent

import numpy as np

import sensor_fusion_live as dashboard
from event_detector_live import SwallowEventDetector


def ready_detector(csv_path, **kwargs):
    detector = SwallowEventDetector(
        calibration_seconds=0.0,
        csv_path=csv_path,
        **kwargs,
    )
    moment = detector.start_time
    quiet = np.full(64, 0.001, dtype=np.float32)
    for _ in range(25):
        detector.on_audio(quiet, now=moment)
        detector.on_imu((0.0, 0.0, 1.0), (0.2, 0.0, 0.0), now=moment)
    return detector, moment


class DashboardStatusTests(unittest.TestCase):
    def test_manual_label_saves_both_sensor_streams_with_zero_detections(self):
        board = dashboard.create_dashboard(experiment_mode=True)
        canvas = board["fig"].canvas
        reader_started = threading.Event()
        mic = {"default_samplerate": 8, "max_input_channels": 2, "name": "test"}
        port = SimpleNamespace(device="test-port", description="Arduino")

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            env = {"DYSPHAGRIP_EVENTS": str(folder / "events.csv"), "DYSPHAGRIP_SNAPSHOT": ""}

            def simulate_session():
                self.assertTrue(reader_started.wait(timeout=2))
                receive_imu = reader.call_args.args[2]
                receive_audio = stream.call_args.kwargs["callback"]
                clock.return_value = 102.75
                receive_imu((0.0, 0.0, 1.0, 0.2, 0.0, 0.0))
                clock.return_value = 103.0
                # Two 2-second stereo blocks; only the contact input is recorded.
                audio = np.tile([0.001, 0.9], (16, 1)).astype(np.float32)
                receive_audio(audio, 16, SimpleNamespace(currentTime=90, inputBufferAdcTime=88), None)
                canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, key="4"))
                clock.return_value = 103.25
                receive_imu((0.1, 0.0, 1.0, 0.2, 0.0, 0.0))
                clock.return_value = 105.125
                audio[:, 0] = -0.001
                receive_audio(audio, 16, SimpleNamespace(currentTime=92.125, inputBufferAdcTime=90), None)

            with (
                patch.dict(dashboard.os.environ, env),
                patch.object(dashboard, "find_contact_mic", return_value=(0, mic)),
                patch.object(dashboard.list_ports, "comports", return_value=[port]),
                patch.object(dashboard, "create_dashboard", return_value=board),
                patch.object(dashboard, "experiment_directory", return_value=folder),
                patch.object(dashboard, "SwallowEventDetector", side_effect=lambda **kw: ready_detector(**kw)[0]),
                patch.object(dashboard.sd, "InputStream") as stream,
                patch.object(dashboard, "read_arduino", side_effect=lambda *_: reader_started.set()) as reader,
                patch.object(dashboard.time, "monotonic", return_value=100.0) as clock,
                patch.object(dashboard.plt, "show", side_effect=simulate_session),
            ):
                dashboard.main(["--experiment"])
            with (folder / "summary.csv").open() as file:
                summary = next(csv.DictReader(file))
            self.assertEqual((summary["label_count"], summary["prediction_count"]), ("1", "0"))
            mark = folder / "captures" / "mark_0001"
            wav_path = mark / "audio.wav"
            with wave.open(str(wav_path)) as wav:
                self.assertEqual((wav.getframerate(), wav.getnchannels(), wav.getnframes()), (8, 1, 32))
                pcm = wav.readframes(32)
                width = wav.getsampwidth()
                samples = [int.from_bytes(pcm[i:i + width], "little", signed=True)
                           for i in range(0, len(pcm), width)]
            self.assertTrue(all(value > 0 for value in samples[:16]))
            self.assertTrue(all(value < 0 for value in samples[16:]))
            with (mark / "motion.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([float(row["seconds_from_mark"]) for row in rows], [-0.25, 0.25])
            self.assertEqual([float(row["tilt_x"]) for row in rows], [0.0, 0.1])

    def test_experiment_keys_and_click_save_and_display_manual_marks(self):
        board = dashboard.create_dashboard(experiment_mode=True)
        canvas = board["fig"].canvas
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mic = {"default_samplerate": 48000, "max_input_channels": 1, "name": "test"}
            port = SimpleNamespace(device="test-port", description="Arduino")

            def press_controls():
                for key, name in (("4", "Movimiento"), ("1", "Saliva")):
                    event = KeyEvent("key_press_event", canvas, key=key)
                    canvas.callbacks.process(event.name, event)
                    self.assertIn(name, board["label_status"].get_text())
                    self.assertIn("guardada", board["label_status"].get_text())
                canvas.draw()
                x, y = board["label_buttons"]["4"].ax.transAxes.transform((0.5, 0.5))
                for kind in ("button_press_event", "button_release_event"):
                    canvas.callbacks.process(kind, MouseEvent(kind, canvas, x, y, button=1))
                self.assertIn("Marca #3 guardada · Movimiento", board["label_status"].get_text())
                self.assertEqual(board["counter_text"].get_text(), "0")
                self.assertTrue(any(
                    text.get_text() == "#3 Movimiento" for text in board["audio_axis"].texts
                ))
                with (folder / "ground_truth.csv").open() as file:
                    self.assertEqual([row["category"] for row in csv.DictReader(file)],
                                     ["head_movement", "swallowing_saliva", "head_movement"])

            with (
                patch.dict(dashboard.os.environ, {"DYSPHAGRIP_EVENTS": str(folder / "events.csv"),
                                                 "DYSPHAGRIP_SNAPSHOT": ""}),
                patch.object(dashboard, "find_contact_mic", return_value=(0, mic)),
                patch.object(dashboard.list_ports, "comports", return_value=[port]),
                patch.object(dashboard.sd, "InputStream"),
                patch.object(dashboard, "read_arduino"),
                patch.object(dashboard, "create_dashboard", return_value=board),
                patch.object(dashboard, "experiment_directory", return_value=folder),
                patch.object(dashboard.plt, "show", side_effect=press_controls),
            ):
                dashboard.main(["--experiment"])

    def test_sessions_started_in_same_second_get_distinct_directories(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(dashboard, "datetime") as clock:
            clock.now.return_value.strftime.return_value = "20261115_010000"
            root = Path(tmp) / "experiments"
            first = dashboard.experiment_directory(root)
            second = dashboard.experiment_directory(root)
            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir() and second.is_dir())

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

    def test_stale_imu_pauses_active_status(self):
        lines = dashboard.status_lines(
            110.0, {"ready": True, "events": []},
            last_audio_at=110.0, last_imu_at=107.0,
        )
        self.assertIn("MPU6050       No signal (3s)", lines)
        self.assertIn("Detector      Waiting for signal", lines)

    def test_arduino_disconnect_finishes_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            env = {
                "DYSPHAGRIP_EVENTS": str(folder / "events.csv"),
                "DYSPHAGRIP_SNAPSHOT": str(folder / "snapshot.png"),
            }
            mic = {"default_samplerate": 48000, "max_input_channels": 1, "name": "test"}
            port = SimpleNamespace(device="test-port", description="test")
            with (
                patch.dict(dashboard.os.environ, env),
                patch.object(dashboard, "find_contact_mic", return_value=(0, mic)),
                patch.object(dashboard.list_ports, "comports", return_value=[port]),
                patch("builtins.input", return_value="0"),
                patch.object(dashboard.sd, "InputStream"),
                patch.object(dashboard, "read_arduino", side_effect=OSError("disconnected")),
                patch.object(dashboard, "experiment_directory", return_value=folder),
                self.assertRaisesRegex(SystemExit, "disconnected"),
            ):
                dashboard.main(["--experiment"])
            self.assertTrue((folder / "summary.csv").is_file())
            self.assertFalse((folder / "snapshot.png").exists())

    def test_full_experiment_export_exceeds_dashboard_history(self):
        quiet = np.full(64, 0.001, dtype=np.float32)
        loud = np.full(64, 0.05, dtype=np.float32)

        def populate_detector(**kwargs):
            detector, start = ready_detector(**kwargs)
            for index in range(205):
                when = start + 2.0 * (index + 1)
                detector.on_audio(quiet, now=when - 0.1)
                detector.on_imu((0, 0, 1), (0.2, 0, 0), now=when - 0.1)
                detector.on_audio(loud, now=when)
                detector.on_imu((0, 0, 1), (4, 0, 0), now=when)
            view = dashboard.detector_view(detector)
            self.assertEqual(view["event_count"], 205)
            self.assertEqual(len(view["events"]), 200)
            return detector

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            env = {"DYSPHAGRIP_EVENTS": str(folder / "events.csv"), "DYSPHAGRIP_SNAPSHOT": ""}
            mic = {"default_samplerate": 48000, "max_input_channels": 1, "name": "test"}
            port = SimpleNamespace(device="test-port", description="Arduino")
            with (
                patch.dict(dashboard.os.environ, env),
                patch.object(dashboard, "find_contact_mic", return_value=(0, mic)),
                patch.object(dashboard.list_ports, "comports", return_value=[port]),
                patch("builtins.input", return_value="0"),
                patch("builtins.print"),
                patch.object(dashboard.sd, "InputStream"),
                patch.object(dashboard, "read_arduino"),
                patch.object(dashboard.plt, "show"),
                patch.object(dashboard, "SwallowEventDetector", side_effect=populate_detector),
                patch.object(dashboard, "experiment_directory", return_value=folder),
            ):
                dashboard.main(["--experiment"])
            with (folder / "predictions.csv").open(newline="") as file:
                self.assertEqual(
                    [int(row["event_id"]) for row in csv.DictReader(file)],
                    list(range(1, 206)),
                )
            with (folder / "summary.csv").open(newline="") as file:
                summary = next(csv.DictReader(file))
            self.assertEqual(summary["prediction_count"], "205")
            self.assertEqual(summary["false_positives"], "205")

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
        image = Path(folder) / "dysphagrip_layout.png"
        board["fig"].savefig(image, dpi=110, facecolor=board["fig"].get_facecolor())
        self.assertGreater(image.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
