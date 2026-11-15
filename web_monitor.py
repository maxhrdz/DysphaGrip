"""Local browser monitor for the real contact microphone and Arduino. Starts idle."""

import argparse
import csv
import json
import logging
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import NamedTuple
from urllib.parse import urlsplit

import numpy as np

from audio_model import AudioModel
from activity_model import ActivityModel, InsufficientMotion, extract_activity_features, extract_voice_features
from event_detector_live import SwallowEventDetector, find_contact_mic, parse_imu_line
from label_capture import LabelCapture
from motion_cough import MotionCoughSpecialist, extract_motion_cough_features
from sensor_fusion_live import (
    DEFAULT_ACTIVITY_MODEL, DEFAULT_MOTION_COUGH, displayed_activity, experiment_directory,
    save_activity_report, save_audio_comparison,
)
from session_audio import SessionAudioRecorder, TIMESTAMP_TOLERANCE
from validation import CATEGORIES, ExperimentLog


ROOT = Path(__file__).resolve().parent
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
CATEGORIES_BY_NAME = {category for category, _ in CATEGORIES.values()}


class Packet(NamedTuple):
    kind: str
    stamp: float
    values: object
    rate: int = 0


def usb_motion_candidates(ports):
    """Prefer Arduino descriptors, then common USB serial bridges. Skip host consoles."""
    boards, adapters = [], []
    for port in ports:
        device = port.device.lower()
        if "bluetooth" in device or "debug-console" in device:
            continue
        description = " ".join(str(getattr(port, name, "") or "").lower()
                               for name in ("manufacturer", "product", "description"))
        vid = getattr(port, "vid", None)
        if vid in {0x2341, 0x2A03} or "arduino" in description or "uno q" in description:
            boards.append(port.device)
        elif (vid in {0x0403, 0x10C4, 0x1A86} or
              any(name in device for name in ("usbmodem", "usbserial", "usbtouart", "ttyacm", "ttyusb"))):
            adapters.append(port.device)
    return sorted(set(boards or adapters))


def select_motion_port(ports, selected=None):
    ports = list(ports)
    if selected:
        if any(p.device == selected and not any(k in p.device.lower() for k in ("bluetooth", "debug-console"))
               for p in ports):
            return selected
        raise RuntimeError(f"Configured Arduino port {selected} is unavailable. Reconnect the board or update DYSPHAGRIP_ARDUINO_PORT.")
    candidates = usb_motion_candidates(ports)
    if not candidates:
        raise RuntimeError("No Arduino USB serial port is visible. Connect the UNO Q with a USB data cable, wait for it to start, then retry recording.")
    if len(candidates) > 1:
        raise RuntimeError("Multiple USB serial ports are available: " + ", ".join(candidates)
                           + ". Set DYSPHAGRIP_ARDUINO_PORT to the board's port and restart the monitor.")
    return candidates[0]


def device_presence():
    """Enumerate hardware without opening either device, a serial port or an audio stream."""
    from serial.tools import list_ports
    if sys.platform == "darwin":
        bus = subprocess.run(["/usr/sbin/ioreg", "-p", "IOUSB", "-w", "0"], capture_output=True,
                             text=True, timeout=1, check=True).stdout
        audio = bool(re.search(r"\+-o [^\n]*AIR[ \t_-]*192[ \t|_-]*6[^\n]*@", bus, re.I))
    else:
        import sounddevice as sd
        audio = any(int(d["max_input_channels"]) > 0 and ("air 192" in d["name"].lower()
                    or "m-audio" in d["name"].lower()) for d in sd.query_devices())
    ports = list(list_ports.comports())
    choice = os.environ.get("DYSPHAGRIP_ARDUINO_PORT")
    imu = bool(usb_motion_candidates(ports) or (choice and any(
        p.device == choice and not any(k in p.device.lower() for k in ("bluetooth", "debug-console")) for p in ports)))
    return {"audio": audio, "imu": imu}


def hardware_packets(stop):
    """Open the USB devices after a start request. Yield verified samples or a heartbeat."""
    import serial
    import sounddevice as sd
    from serial.tools import list_ports

    try:
        device, info = find_contact_mic()
    except SystemExit as error:
        raise RuntimeError("M-Audio AIR 192|6 input not found. Connect it by USB and plug the contact pickup into input 1.") from error
    port = select_motion_port(list_ports.comports(), os.environ.get("DYSPHAGRIP_ARDUINO_PORT"))
    # The desktop monitor also uses this port. Avoid sharing its serial stream.
    if shutil.which("lsof"):
        busy = subprocess.run(["lsof", "-t", "--", port], capture_output=True, timeout=4)
        if busy.stdout.strip():
            raise RuntimeError(f"Serial port {port} is in use. Close the desktop monitor or Arduino Serial Monitor.")
    rate = int(info["default_samplerate"])
    events = queue.Queue(maxsize=512)
    failures = []

    def offer(item):
        try:
            events.put_nowait(item)
        except queue.Full:
            if not failures:
                failures.append("Sensor capture fell behind and lost samples.")

    offset = None

    def audio_callback(indata, frames, timing, status):
        nonlocal offset
        if stop.is_set() or failures:
            return
        now = time.monotonic()
        adc, current = timing.inputBufferAdcTime, timing.currentTime
        if adc > 0 and adc <= current and math.isfinite(current):
            if offset is None:
                offset = now - current
            start = adc + offset
        else:
            start = now - frames / rate
        if status.input_overflow:
            failures.append("The audio interface lost samples. Start a new recording.")
            return
        offer(Packet("audio", start, indata[:, 0].copy(), rate))

    with serial.Serial(port, 115200, timeout=.15, exclusive=True) as arduino:
        def read_motion():
            try:
                while not stop.is_set() and not failures:
                    raw = arduino.readline()
                    if raw.endswith(b"\n"):
                        values = parse_imu_line(raw.decode("utf-8", errors="replace").strip())
                        if values is not None:
                            offer(Packet("imu", time.monotonic(), values))
            except (OSError, serial.SerialException) as error:
                failures.append(f"Arduino connection interrupted: {error}")

        with sd.InputStream(device=device, channels=int(info["max_input_channels"]),
                            samplerate=rate, blocksize=1024, dtype="float32", callback=audio_callback) as stream:
            reader = threading.Thread(target=read_motion, name="web-monitor-imu", daemon=True)
            reader.start()
            try:
                while not stop.is_set():
                    if failures or not stream.active:
                        raise RuntimeError(failures[0] if failures else "The audio input stopped.")
                    try:
                        yield events.get(timeout=.15)
                    except queue.Empty:
                        yield None
            finally:
                stop.set()
                reader.join(timeout=1)


def waveform(samples, buckets=180):
    if len(samples) < buckets:
        return samples.tolist()
    frames = samples[:len(samples) // buckets * buckets].reshape(buckets, -1)
    return np.column_stack((frames.min(axis=1), frames.max(axis=1))).ravel().tolist()


class Recording:
    """One hardware session. Worker processes samples, HTTP mark requests use Monitor.lock."""

    def __init__(self, root, rate, activity_model, motion_model):
        self.path = experiment_directory(root)
        (self.path / "web_recording.pending").touch()
        self.rate = rate
        self.activity_model = activity_model
        self.motion_model = motion_model
        self.log = ExperimentLog(self.path)
        self.detector = SwallowEventDetector(csv_path=self.path / "dysphagrip_events.csv", keep_session_events=True)
        self.capture = LabelCapture(self.path, rate)
        self.recorder = SessionAudioRecorder(self.path, rate)
        self.audio = deque()
        self.motion = deque()
        self.activity_history = deque(maxlen=2)
        self.predictions = []
        self.mark_estimates = []
        self.activity = None
        self.audio_end = self.last_inference = self.first_active = None
        self.valid_samples = 0

    def consume(self, packet):
        stamp, values = packet.stamp, packet.values
        if packet.kind == "imu":
            self.motion.append((stamp, *values))
            while self.motion and self.motion[0][0] < stamp - 5:
                self.motion.popleft()
            self.capture.add_imu(stamp, values)
            self.detector.on_imu(values[:3], values[3:], now=stamp)
            return
        if self.audio_end is None or abs(stamp - self.audio_end) > TIMESTAMP_TOLERANCE:
            self.valid_samples = 0
            self.audio.clear()
        self.audio_end = stamp + len(values) / self.rate
        self.valid_samples = min(self.rate, self.valid_samples + len(values))
        self.audio.append((stamp, values))
        while self.audio and self.audio[0][0] < self.audio_end - 5:
            self.audio.popleft()
        self.capture.add_audio(values, stamp)
        if self.detector.ready and self.motion and abs(self.audio_end - self.motion[-1][0]) < 1.2:
            self.recorder.add_audio(values, stamp)
            if self.first_active is None:
                self.first_active = stamp
        self.detector.on_audio(values, now=self.audio_end)

    def update(self, now, hardware):
        audio = np.concatenate([block for _, block in self.audio]) if self.audio else np.empty(0)
        rows = [row for row in self.motion if row[0] >= now - 5]
        calibrated = self.detector.ready and all(hardware.values())
        if not calibrated or self.valid_samples < self.rate:
            self.activity = None
            self.activity_history.clear()
        elif self.last_inference is None or self.audio_end - self.last_inference >= .20:
            self.last_inference = self.audio_end
            window = audio[-self.rate:]
            raw, score = "uncertain", None
            try:
                features = extract_activity_features(window, self.rate, rows, self.audio_end)
                voice = extract_voice_features(window, self.rate) if self.activity_model.has_voice_specialist else None
                scores = self.activity_model.scores(features, voice)
                detail = extract_motion_cough_features(window, self.rate, rows, self.audio_end)
                raw, score = self.motion_model.predict(scores, detail)
            except InsufficientMotion:
                self.activity_history.clear()
            shown = displayed_activity(raw, self.audio_end, self.activity_history)
            self.activity_history.append((self.audio_end, raw))
            self.activity = {"label": shown, "raw_label": raw, "score": score,
                             "time": max(0., self.audio_end - self.log.started)}
            self.predictions.append((self.audio_end, raw, score, shown))
        axes = np.array(rows).reshape(-1, 7)
        self.capture.flush(now)
        return {
            "session": self.path.name, "calibrated": calibrated, "event_count": self.detector.event_count,
            "elapsed_seconds": max(0., now - self.log.started), "activity": self.activity,
            "audio_waveform": waveform(audio), "audio_span_seconds": len(audio) / self.rate,
            "audio_end_offset_seconds": self.audio_end - now if self.audio_end is not None else 0.,
            "motion": {"time": (axes[:, 0] - now).tolist(), "accel": np.linalg.norm(axes[:, 1:4], axis=1).tolist(),
                       "gyro": np.linalg.norm(axes[:, 4:], axis=1).tolist()},
        }

    def mark(self, category, when):
        label = self.log.add_label(category, when=when)
        self.capture.mark(label)
        info = self.activity or {}
        self.mark_estimates.append((label["id"], when, category, self.last_inference if info else None,
                                    info.get("raw_label", "waiting"), info.get("label", "waiting")))
        return {"id": label["id"], "category": category}

    def finish(self, ended, complete):
        try:
            self.capture.finish(ended)
        finally:
            self.recorder.close(active_from=self.first_active)
        summary = self.log.finish(list(self.detector.events), ended)
        for name, columns, rows in (
            ("activity_predictions.csv", ["window_end_monotonic", "estimated_activity", "score", "displayed_activity"],
             self.predictions),
            ("activity_at_marks.csv", ["label_id", "mark_monotonic", "manual_activity",
                                       "window_end_monotonic", "estimated_activity", "displayed_activity"],
             self.mark_estimates),
        ):
            with (self.path / name).open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(columns)
                writer.writerows(rows)
        if complete:
            result = {"session_id": self.path.name, "started_at": summary["started_at"],
                      "completed_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                      "duration_seconds": summary["duration_seconds"], "event_count": self.detector.event_count,
                      "marks": len(self.log.labels), "counts": dict(Counter(l["category"] for l in self.log.labels))}
            tmp = self.path / "web_recording.json.tmp"
            tmp.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(self.path / "web_recording.json")
            (self.path / "web_recording.pending").unlink()
            if self.log.labels:
                # Reports have independent coverage requirements. The finalized
                # recording stays available if either comparison has insufficient data.
                try:
                    save_activity_report(self.path, self.activity_model, self.motion_model)
                except (ValueError, OSError) as error:
                    logging.info("Activity report unavailable for %s: %s", self.path.name, error)
                try:
                    audio_model = AudioModel.load(ROOT / "models" / "audio_pilot.json")
                    save_audio_comparison(self.path, audio_model)
                except (ValueError, OSError) as error:
                    logging.info("Audio comparison unavailable for %s: %s", self.path.name, error)
        return summary["duration_seconds"]


def empty_status():
    return {"state": "idle", "error": None, "session": None, "elapsed_seconds": 0.,
            "hardware": {"audio": False, "imu": False}, "calibrated": False,
            "audio_waveform": [], "audio_span_seconds": 0., "audio_end_offset_seconds": 0.,
            "motion": {"time": [], "accel": [], "gyro": []}, "activity": None,
            "marks": 0, "last_mark": None, "event_count": 0}


class Monitor:
    def __init__(self, root=ROOT / "experiments", source=hardware_packets, probe=device_presence):
        self.root, self.source = Path(root), source
        self.probe = probe
        self.devices = None
        self.next_probe = 0.
        self.lock = threading.RLock()
        self.status = empty_status()
        self.seen = {}
        self.stop_event = threading.Event()
        self.worker = self.recording = None

    def snapshot(self):
        now = time.monotonic()
        with self.lock:
            refresh = now >= self.next_probe
            if refresh:
                self.next_probe = now + 2.
        if refresh:
            try:
                presence = self.probe()
            except (ImportError, OSError, RuntimeError, subprocess.SubprocessError):
                presence = None
            with self.lock:
                self.devices = presence
        with self.lock:
            return dict(self.status, devices=self.devices)

    def start(self):
        self.snapshot()
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise RuntimeError("A session is already in progress.")
            self.status = empty_status()
            self.status["state"] = "starting"
            self.seen = {}
            self.stop_event = threading.Event()
            self.worker = threading.Thread(target=self._run, name="web-monitor-session", daemon=True)
            self.worker.start()
            return dict(self.status, devices=self.devices)

    def stop(self):
        with self.lock:
            worker = self.worker
            if not worker or not worker.is_alive():
                return dict(self.status, devices=self.devices)
            self.status["state"] = "stopping"
            self.stop_event.set()
        worker.join(timeout=8)
        return self.snapshot()

    def mark(self, category):
        if category not in CATEGORIES_BY_NAME:
            raise ValueError("Unknown activity label.")
        with self.lock:
            now = time.monotonic()
            if (self.status["state"] != "recording" or not self.recording or not self.recording.detector.ready
                    or any(now - self.seen.get(k, 0.) >= 1.2 for k in ("imu", "audio"))):
                raise RuntimeError("Wait for calibration and live signals from both sensors.")
            self.status["last_mark"] = self.recording.mark(category, now)
            self.status["marks"] += 1
            return dict(self.status, devices=self.devices)

    def _run(self):
        error = None
        capture = None
        try:
            activity_model = ActivityModel.load(DEFAULT_ACTIVITY_MODEL)
            motion_model = MotionCoughSpecialist.load(DEFAULT_MOTION_COUGH)
            opened = time.monotonic()
            recent = {}
            last_update = 0.
            packets = self.source(self.stop_event)
            try:
                for packet in packets:
                    if self.stop_event.is_set():
                        break
                    now = time.monotonic()
                    if packet is not None:
                        self.seen[packet.kind] = now
                        recent[packet.kind] = packet
                    hardware = {kind: now - self.seen.get(kind, -math.inf) < 1.2 for kind in ("audio", "imu")}
                    if capture is None and all(hardware.values()):
                        capture = Recording(self.root, recent["audio"].rate, activity_model, motion_model)
                        with self.lock:
                            self.recording = capture
                            if not self.stop_event.is_set():
                                self.status.update(state="recording", session=capture.path.name)
                        for first in sorted(recent.values(), key=lambda p: p.stamp):
                            capture.consume(first)
                    elif capture is not None and packet is not None:
                        capture.consume(packet)
                    if now - last_update >= .10:
                        update = capture.update(now, hardware) if capture else {}
                        with self.lock:
                            self.status.update(hardware=hardware, **update)
                        last_update = now
                    if now - self.seen.get("imu", opened) > 8:
                        raise RuntimeError("The serial port opened, but no valid MPU6050 samples arrived. Check the UNO Q sketch, I²C connection and Tilt/Turn output at 115200 baud.")
                    if now - self.seen.get("audio", opened) > 8:
                        raise RuntimeError("The M-Audio input opened, but no audio samples arrived. Check its USB connection and microphone permission.")
            finally:
                packets.close()
        except (Exception, SystemExit) as exc:
            error = str(exc) or "Sensor capture failed."
        finally:
            with self.lock:
                self.recording = None
                self.status["state"] = "stopping"
            if capture:
                try:
                    duration = capture.finish(time.monotonic(), complete=error is None)
                    with self.lock:
                        self.status.update(elapsed_seconds=duration, event_count=capture.detector.event_count)
                except Exception:
                    logging.exception("Could not finalize session %s", capture.path)
                    error = "Could not finish saving the session. Check available disk space and folder permissions."
            with self.lock:
                self.status.update(state="error" if error else "idle", error=error,
                                   hardware={"audio": False, "imu": False}, calibrated=False, activity=None)


def make_server(port, monitor):
    class Handler(BaseHTTPRequestHandler):
        def local_origin(self):
            origin = self.headers.get("Origin")
            if not origin:
                return True
            parsed = urlsplit(origin)
            return parsed.scheme in {"http", "https"} and parsed.hostname in LOCAL_HOSTS

        def reply(self, status, result):
            data = json.dumps(result, allow_nan=False).encode("utf-8")
            self.send_response(status)
            if self.local_origin() and self.headers.get("Origin"):
                self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
                self.send_header("Vary", "Origin")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            if not self.local_origin() or urlsplit("//" + self.headers.get("Host", "")).hostname not in LOCAL_HOSTS:
                self.reply(403, {"error": "The sensor monitor only accepts local requests."})
                return False
            return True

        def do_OPTIONS(self):
            if not self.allowed():
                return
            self.send_response(204)
            if self.headers.get("Origin"):
                self.send_header("Access-Control-Allow-Origin", self.headers["Origin"])
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            if self.allowed():
                if urlsplit(self.path).path == "/api/monitor/status":
                    self.reply(200, monitor.snapshot())
                else:
                    self.reply(404, {"error": "Unknown monitor endpoint."})

        def do_POST(self):
            if not self.allowed():
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 4096 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Send a JSON object of up to 4096 bytes.")
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object.")
                route = urlsplit(self.path).path
                if route == "/api/monitor/start":
                    result = monitor.start()
                elif route == "/api/monitor/stop":
                    result = monitor.stop()
                elif route == "/api/monitor/mark":
                    result = monitor.mark(body.get("category"))
                else:
                    self.reply(404, {"error": "Unknown monitor endpoint."})
                    return
            except (ValueError, TypeError) as exc:
                self.reply(422, {"error": str(exc)})
            except RuntimeError as exc:
                self.reply(409, {"error": str(exc)})
            else:
                self.reply(200, result)

        def log_message(self, fmt, *args):
            pass  # Polling is frequent; status already exposes failures.

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    monitor = Monitor()
    server = make_server(args.port, monitor)
    # BaseServer.shutdown must run in a different thread than serve_forever.
    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    print(f"Local sensor monitor at http://127.0.0.1:{server.server_port}/api/monitor/status", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        monitor.stop()


if __name__ == "__main__":
    main()
