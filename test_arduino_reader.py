import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

import event_detector_live as live


FRAME = b"Tilt X: -0.99 Y: 0.00 Z: 0.20 | Turn X: 1.00 Y: 2.00 Z: 3.00"


class ArduinoReaderTests(unittest.TestCase):
    def test_dashboard_reads_six_axes_with_board_timing_suffix(self):
        self.assertEqual(
            live.parse_imu_line(FRAME.decode() + " seq=245 t_ms=10032"),
            (-0.99, 0, 0.2, 1, 2, 3),
        )

    def test_skips_corrupt_frames_and_delivers_next_complete_sample(self):
        serial = MagicMock()
        port = serial.Serial.return_value.__enter__.return_value
        port.readline.side_effect = [
            b"booting\n",
            FRAME.replace(b"-0.99", b"--1") + b"\n",
            FRAME.replace(b"0.20", b"0.\xff20") + b"\n",
            FRAME.replace(b"1.00", b"1e999") + b"\n",
            FRAME + b".0\n",
            FRAME,  # readline timed out before the line terminator
            FRAME + b"\r\n",
        ]
        stop = threading.Event()
        samples = []

        def receive(values):
            samples.append(values)
            stop.set()

        with patch.dict(sys.modules, {"serial": serial}):
            live.read_arduino("mock-port", stop, receive)
        self.assertEqual(port.readline.call_count, 7)
        self.assertEqual(samples, [(-0.99, 0, 0.2, 1, 2, 3)])

    def test_serial_failure_terminates_live_session(self):
        serial = MagicMock()
        serial.Serial.side_effect = OSError("USB disconnected")
        modules = {"sounddevice": MagicMock(), "serial": serial, "serial.tools": serial.tools}
        port = MagicMock(device="usbmodem", description="Arduino")
        serial.tools.list_ports.comports.return_value = [port]
        info = {"name": "Mock mic", "default_samplerate": 48000, "max_input_channels": 2}
        with (
            patch.dict(sys.modules, modules),
            patch.object(live, "find_contact_mic", return_value=(0, info)),
            patch.object(live, "SwallowEventDetector"),
            patch.object(threading, "Thread") as thread,
            patch.object(threading.Event, "wait", autospec=True) as wait,
        ):
            thread.return_value.start.side_effect = lambda: thread.call_args.kwargs["target"]()
            wait.side_effect = lambda stop: self.assertTrue(stop.is_set())
            with self.assertRaisesRegex(SystemExit, "USB disconnected"):
                live.main()
        modules["sounddevice"].InputStream.return_value.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
