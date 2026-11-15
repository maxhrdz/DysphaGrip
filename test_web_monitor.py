import csv
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from event_detector_live import SwallowEventDetector
from session_audio import iter_segments
from web_monitor import Monitor, Packet, make_server, select_motion_port, usb_motion_candidates


def wait_for(condition, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = condition()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError("timeout waiting for monitor state")


class WebMonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_motion_discovery_distinguishes_missing_multiple_and_configured_ports(self):
        def port(name, maker=None, vid=None):
            return SimpleNamespace(device="/dev/cu." + name, description=None, manufacturer=maker, vid=vid)
        host = [port("Bluetooth-Incoming-Port"), port("debug-console")]
        bridge = port("SLAB_USBtoUART", vid=0x10c4)
        uno = port("usbmodem1180261702", "Arduino", 0x2341)
        self.assertEqual(usb_motion_candidates(host), [])
        with self.assertRaisesRegex(RuntimeError, "No Arduino USB serial"):
            select_motion_port(host)
        self.assertEqual(select_motion_port([*host, bridge]), bridge.device)
        self.assertEqual(select_motion_port([*host, bridge, uno]), uno.device)
        with self.assertRaisesRegex(RuntimeError, "Multiple.*usbmodem"):
            select_motion_port([uno, port("usbmodem21201", "Arduino")])
        with self.assertRaisesRegex(RuntimeError, "Configured.*unavailable"):
            select_motion_port([uno], "/dev/cu.usbmodemOld")
        self.assertEqual(select_motion_port([uno, bridge], bridge.device), bridge.device)

    def test_device_presence_observes_hotplug_without_opening_inputs(self):
        probe = Mock(side_effect=[{"audio": True, "imu": False}, {"audio": True, "imu": True}])
        source = Mock(side_effect=AssertionError("status must not open sensor streams"))
        monitor = Monitor(self.root, source=source, probe=probe)
        with patch("web_monitor.time.monotonic", side_effect=[100., 101., 103.]):
            self.assertFalse(monitor.snapshot()["devices"]["imu"])
            self.assertFalse(monitor.snapshot()["devices"]["imu"])
            latest = monitor.snapshot()
        self.assertTrue(latest["devices"]["imu"])
        self.assertEqual(latest["state"], "idle")
        self.assertFalse(any(latest["hardware"].values()))
        self.assertEqual(probe.call_count, 2)
        source.assert_not_called()

    def test_worker_exports_comparisons_only_with_evaluable_material(self):
        from concurrent.futures import ThreadPoolExecutor
        from activity_model import ActivityModel
        from motion_cough import MotionCoughSpecialist
        from sensor_fusion_live import DEFAULT_ACTIVITY_MODEL, DEFAULT_MOTION_COUGH
        from web_monitor import Recording

        primary = ActivityModel.load(DEFAULT_ACTIVITY_MODEL)
        specialist = MotionCoughSpecialist.load(DEFAULT_MOTION_COUGH)
        with ThreadPoolExecutor(max_workers=1) as worker:
            recording = Recording(self.root, 8000, primary, specialist)
            start = recording.log.started
            recording.mark("rest", start + 2)
            sound = np.zeros(4 * recording.rate, dtype="f4")
            recording.capture.add_audio(sound, start)
            recording.recorder.add_audio(sound, start)
            for dt in np.arange(0., 4., .2):
                recording.capture.add_imu(start + dt, (0, 0, 1, 0, 0, 0))
            with patch("matplotlib.pyplot.subplots", side_effect=AssertionError("GUI on worker")):
                worker.submit(recording.finish, start + 4.1, True).result()
            activity = json.loads((recording.path / "activity_report.json").read_text())
            self.assertEqual(activity["metrics"]["per_class"]["rest"]["examples"], 1)
            audio = json.loads((recording.path / "audio_comparison.json").read_text())
            self.assertEqual(audio["scored_labels"], 1)
            self.assertTrue((recording.path / "audio_comparison.png").read_bytes().startswith(b"\x89PNG"))

            short = Recording(self.root, 8000, primary, specialist)
            short.mark("rest", short.log.started + .1)
            worker.submit(short.finish, short.log.started + .2, True).result()
            self.assertTrue((short.path / "web_recording.json").is_file())
            self.assertTrue((short.path / "summary.csv").is_file())
            self.assertFalse((short.path / "activity_report.json").exists())
            self.assertFalse((short.path / "audio_comparison.json").exists())

    def test_only_genuine_packet_pair_opens_session(self):
        def no_devices(stop):
            raise RuntimeError("Falta el Arduino")
            yield
        monitor = Monitor(self.root, no_devices, probe=lambda: None)
        self.assertEqual(monitor.start()["state"], "starting")
        wait_for(lambda: monitor.snapshot()["state"] == "error")
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertEqual(monitor.snapshot()["error"], "Falta el Arduino")

        def audio_only(stop):
            while not stop.wait(.01):
                yield Packet("audio", time.monotonic(), np.zeros(100), 1000)
        monitor.source = audio_only
        monitor.start()
        wait_for(lambda: monitor.snapshot()["hardware"]["audio"])
        self.assertEqual(monitor.stop()["state"], "idle")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_stop_mark_and_replay_with_isolated_inputs(self):
        # Simulated devices are only injected inside this test's temporary directory.
        def packets(stop):
            rate = 8000
            start = time.monotonic()
            for i in range(1000):
                stamp = start + i * .02
                if stop.wait(max(0., stamp + .02 - time.monotonic())):
                    return
                yield Packet("imu", stamp, (0., .1, 1., 0., 2., 1.))
                idx = np.arange(160) + i * 160
                yield Packet("audio", stamp, (.004 * np.sin(2*np.pi*180*idx/rate)).astype("f4"), rate)

        monitor = Monitor(self.root, packets, probe=lambda: {"audio": True, "imu": True})
        server = make_server(0, monitor)
        self.addCleanup(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.addCleanup(monitor.stop)
        base = f"http://127.0.0.1:{server.server_port}/api/monitor/"

        def request(path, body=None, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            data = json.dumps(body).encode() if body is not None else None
            with urlopen(Request(base + path, data, headers), timeout=5) as resp:
                return json.load(resp)

        self.assertEqual(request("status")["state"], "idle")
        with self.assertRaises(HTTPError) as denied:
            request("start", {}, "https://remote.invalid")
        self.assertEqual(denied.exception.code, 403)
        self.assertEqual(list(self.root.iterdir()), [])
        with patch("web_monitor.SwallowEventDetector",
                   side_effect=lambda **kwargs: SwallowEventDetector(calibration_seconds=.05, **kwargs)):
            self.assertEqual(request("start", {}, "http://localhost:5173")["state"], "starting")
            with self.assertRaises(HTTPError) as duplicate:
                request("start", {})
            self.assertEqual(duplicate.exception.code, 409)
            wait_for(lambda: request("status")["activity"])
            live = request("status")
            self.assertTrue(live["calibrated"])
            self.assertTrue(all(live["hardware"].values()))
            self.assertTrue(live["motion"]["time"] and live["audio_waveform"])
            self.assertIn(live["activity"]["label"], {"swallowing","coughing","speaking","head_movement","rest","uncertain"})
            self.assertEqual(request("mark", {"category":"rest"})["last_mark"], {"category":"rest","id":1})
            stopped = request("stop", {})
        self.assertEqual(stopped["state"], "idle")
        folder = self.root / stopped["session"]
        summary = json.loads((folder / "web_recording.json").read_text())
        self.assertEqual((summary["marks"], summary["counts"], summary["session_id"]),
                         (1, {"rest":1}, folder.name))
        self.assertGreater(sum(len(samples) for _, samples, _ in iter_segments(folder)), 0)
        self.assertTrue((folder / "captures/mark_0001/motion.csv").exists())
        self.assertTrue((folder / "summary.csv").exists())
        with (folder / "activity_at_marks.csv").open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row["manual_activity"], "rest")
        self.assertNotEqual(row["displayed_activity"], "waiting")
        self.assertEqual(request("stop", {})["session"], stopped["session"])


if __name__ == "__main__":
    unittest.main()
