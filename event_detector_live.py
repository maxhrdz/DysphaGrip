import csv
import re
import time
import threading
from pathlib import Path
from datetime import datetime

import numpy as np


class SwallowEventDetector:
    """
    Experimental audio + IMU swallow-event detector.

    This is NOT a medical diagnostic device.
    It does not detect aspiration or determine swallow safety.
    A detection is only a candidate: "Possible Swallow Event".
    """

    def __init__(
        self,
        calibration_seconds=5.0,
        coincidence_seconds=0.65,
        cooldown_seconds=1.5,
        audio_multiplier=2.5,
        imu_multiplier=3.0,
        csv_path="dysphagrip_events.csv",
    ):
        self.calibration_seconds = calibration_seconds
        self.coincidence_seconds = coincidence_seconds
        self.cooldown_seconds = cooldown_seconds
        self.audio_multiplier = audio_multiplier
        self.imu_multiplier = imu_multiplier

        self.lock = threading.Lock()
        self.start_time = time.monotonic()
        self.last_event_time = -float("inf")

        self.audio_calibration = []
        self.tilt_calibration = []
        self.turn_calibration = []

        self.audio_threshold = None
        self.tilt_threshold = None
        self.turn_threshold = None

        self.previous_tilt = None
        self.last_audio_hit = None
        self.last_imu_hit = None

        self.ready = False
        self.event_count = 0
        self.events = []
        self.last_calibration_warning = 0.0

        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        with self.csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "elapsed_seconds",
                "event_id",
                "event_type",
                "audio_rms",
                "audio_peak",
                "audio_threshold",
                "imu_activity",
                "tilt_delta",
                "turn_magnitude",
                "time_difference_seconds",
            ])

        print("[DysphaGrip] Detector initialized")
        print("[DysphaGrip] Not a medical assessment of swallow safety.")
        print(
            "[Calibration] Remain still and silent "
            f"for {self.calibration_seconds:.0f} seconds."
        )

    @staticmethod
    def calculate_threshold(values, multiplier, floor):
        """Threshold above the quiet-period noise, not inside it."""
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            return floor

        median = float(np.median(data))
        mad = float(np.median(np.abs(data - median)))
        noise_ceiling = float(np.percentile(data, 95))
        return max(
            noise_ceiling * multiplier,
            median + 6.0 * mad,
            floor,
        )

    def try_finish_calibration(self, now):
        if self.ready:
            return

        if now - self.start_time < self.calibration_seconds:
            return

        if (
            len(self.audio_calibration) < 20
            or len(self.tilt_calibration) < 20
        ):
            if now - self.last_calibration_warning > 3.0:
                missing = []
                if len(self.audio_calibration) < 20:
                    missing.append(
                        f"audio {len(self.audio_calibration)}/20"
                    )
                if len(self.tilt_calibration) < 20:
                    missing.append(
                        f"IMU {len(self.tilt_calibration)}/20"
                    )
                print("[Calibration] Waiting for " + ", ".join(missing))
                self.last_calibration_warning = now
            return

        self.audio_threshold = self.calculate_threshold(
            self.audio_calibration,
            self.audio_multiplier,
            floor=0.0001,
        )
        self.tilt_threshold = self.calculate_threshold(
            self.tilt_calibration,
            self.imu_multiplier,
            floor=1e-6,
        )
        self.turn_threshold = self.calculate_threshold(
            self.turn_calibration,
            self.imu_multiplier,
            floor=1e-6,
        )
        self.ready = True

        print(f"Audio threshold: {self.audio_threshold:.6f}")
        print(f"Tilt threshold: {self.tilt_threshold:.6f}")
        print(f"Turn threshold: {self.turn_threshold:.6f}")
        print("[DysphaGrip] Calibration complete")
        print("[DysphaGrip] Detection active")

    def _audio_features(self, audio_samples):
        raw = np.asarray(audio_samples)
        if raw.size == 0:
            return None, None

        if np.issubdtype(raw.dtype, np.integer):
            scale = max(
                abs(np.iinfo(raw.dtype).min),
                np.iinfo(raw.dtype).max,
            )
            samples = raw.astype(np.float64) / scale
        else:
            samples = raw.astype(np.float64)

        if not np.all(np.isfinite(samples)):
            return None, None

        if samples.ndim == 2:
            samples = samples[:, 0]
        elif samples.ndim != 1:
            return None, None

        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        return rms, peak

    def on_audio(self, audio_samples, now=None):
        """Feed one audio chunk from the Step 9 callback. Channel 0 is the pickup."""
        now = time.monotonic() if now is None else float(now)
        rms, peak = self._audio_features(audio_samples)
        if rms is None:
            return

        event = None
        with self.lock:
            if not self.ready:
                self.audio_calibration.append(rms)
                self.try_finish_calibration(now)
                return

            if now - self.last_event_time < self.cooldown_seconds:
                return

            if rms >= self.audio_threshold:
                self.last_audio_hit = {
                    "time": now,
                    "rms": rms,
                    "peak": peak,
                }
                event = self.try_detect(now)

        if event:
            self._record(event)

    def on_imu(self, tilt_xyz, turn_xyz, now=None):
        """Feed one MPU6050 sample: tilt (accel) and turn (gyro), three values each."""
        now = time.monotonic() if now is None else float(now)
        tilt = np.asarray(tilt_xyz, dtype=float)
        turn = np.asarray(turn_xyz, dtype=float)

        if tilt.shape != (3,) or turn.shape != (3,):
            return
        if not (np.all(np.isfinite(tilt)) and np.all(np.isfinite(turn))):
            return

        event = None
        with self.lock:
            if self.previous_tilt is None:
                self.previous_tilt = tilt.copy()
                return

            tilt_delta = float(np.linalg.norm(tilt - self.previous_tilt))
            turn_magnitude = float(np.linalg.norm(turn))
            self.previous_tilt = tilt.copy()

            if not self.ready:
                self.tilt_calibration.append(tilt_delta)
                self.turn_calibration.append(turn_magnitude)
                self.try_finish_calibration(now)
                return

            if now - self.last_event_time < self.cooldown_seconds:
                return

            movement_detected = (
                tilt_delta >= self.tilt_threshold
                or turn_magnitude >= self.turn_threshold
            )
            if movement_detected:
                self.last_imu_hit = {
                    "time": now,
                    "tilt_delta": tilt_delta,
                    "turn_magnitude": turn_magnitude,
                }
                event = self.try_detect(now)

        if event:
            self._record(event)

    def try_detect(self, now):
        if self.last_audio_hit is None or self.last_imu_hit is None:
            return None

        audio = self.last_audio_hit
        imu = self.last_imu_hit
        time_difference = abs(audio["time"] - imu["time"])

        if time_difference > self.coincidence_seconds:
            return None
        if now - min(audio["time"], imu["time"]) > self.coincidence_seconds:
            return None
        if now - self.last_event_time < self.cooldown_seconds:
            self.last_audio_hit = None
            self.last_imu_hit = None
            return None

        self.last_event_time = now
        self.event_count += 1
        imu_activity = max(imu["tilt_delta"], imu["turn_magnitude"])
        event = {
            "event_id": self.event_count,
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            ),
            "elapsed": now - self.start_time,
            "rms": audio["rms"],
            "peak": audio["peak"],
            "tilt_delta": imu["tilt_delta"],
            "turn_magnitude": imu["turn_magnitude"],
            "imu_activity": imu_activity,
            "time_difference": time_difference,
            "audio_threshold": self.audio_threshold,
        }
        self.last_audio_hit = None
        self.last_imu_hit = None
        return event

    def _record(self, event):
        recorded = dict(event)
        recorded["monotonic"] = self.start_time + float(event["elapsed"])
        with self.lock:
            self.events.append(recorded)
            if len(self.events) > 200:
                del self.events[:-200]

        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                event["timestamp"],
                round(event["elapsed"], 3),
                event["event_id"],
                "possible_swallow_event",
                round(event["rms"], 6),
                round(event["peak"], 6),
                round(event["audio_threshold"], 6),
                round(event["imu_activity"], 6),
                round(event["tilt_delta"], 6),
                round(event["turn_magnitude"], 6),
                round(event["time_difference"], 3),
            ])
            file.flush()

        print(f"\n[EVENT {event['event_id']}] Possible Swallow Event Detected")
        print(f"Audio RMS: {event['rms']:.6f}")
        print(f"IMU Activity: {event['imu_activity']:.3f}")
        print(f"Time Difference: {event['time_difference']:.3f}s")
        print(f"Total Events: {event['event_id']}")


IMU_PATTERN = re.compile(
    r"Tilt X:\s*([-\d.]+)\s+"
    r"Y:\s*([-\d.]+)\s+"
    r"Z:\s*([-\d.]+)\s*\|\s*"
    r"Turn X:\s*([-\d.]+)\s+"
    r"Y:\s*([-\d.]+)\s+"
    r"Z:\s*([-\d.]+)"
)


def find_contact_mic():
    """El índice de PortAudio cambia al conectar monitores u otros dispositivos."""
    import sounddevice as sd

    for index, dev in enumerate(sd.query_devices()):
        name = str(dev["name"]).lower()
        if int(dev["max_input_channels"]) < 1:
            continue
        if "air 192" in name or "m-audio" in name:
            return index, dev
    raise SystemExit(
        "No se encontró el M-Audio AIR 192|6. "
        "Conéctalo por USB y vuelve a correr el script."
    )


def arduino_port_score(port):
    text = f"{port.device} {port.description or ''}".lower()
    if any(token in text for token in ("usbmodem", "uno", "arduino")):
        return 0
    if any(token in text for token in ("bluetooth", "debug-console")):
        return 2
    return 1


def choose_arduino_port(ports):
    candidates = [port for port in ports if arduino_port_score(port) == 0]
    if len(candidates) == 1:
        port = candidates[0]
        print(f"Arduino: {port.device} - {port.description}")
        return port.device

    shown = candidates or list(ports)
    print("Puertos del Arduino:")
    for index, port in enumerate(shown):
        print(f"{index}: {port.device} - {port.description}")
    return shown[int(input("Arduino port index: "))].device


def main():
    import sounddevice as sd
    import serial
    from serial.tools import list_ports

    device, info = find_contact_mic()
    sample_rate = int(info["default_samplerate"])
    channels = int(info["max_input_channels"])
    print(
        f"Audio: {info['name']} "
        f"(device {device}, {channels} ch @ {sample_rate} Hz)"
    )

    ports = list(list_ports.comports())
    if not ports:
        raise SystemExit(
            "No hay puertos serie. Revisa el USB del Arduino."
        )

    port_name = choose_arduino_port(ports)
    detector = SwallowEventDetector()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        detector.on_audio(np.array(indata[:, 0], copy=True))

    def read_arduino():
        try:
            with serial.Serial(port_name, 115200, timeout=0.2) as arduino:
                print(f"Puerto serie abierto: {port_name}")
                opened = time.monotonic()
                warned = False
                last_line = ""
                while not stop_event.is_set():
                    line = arduino.readline().decode(
                        "utf-8", errors="ignore"
                    ).strip()
                    if line:
                        last_line = line
                    match = IMU_PATTERN.search(line)
                    if not match:
                        if not warned and time.monotonic() - opened > 2.0:
                            print(
                                "Sin tramas IMU en "
                                f"{port_name}. Última línea: {last_line!r}"
                            )
                            warned = True
                        continue
                    warned = True
                    values = [float(value) for value in match.groups()]
                    detector.on_imu(values[:3], values[3:])
        except Exception as error:
            print("Arduino error:", error)

    reader = threading.Thread(target=read_arduino, daemon=True)
    reader.start()
    print("Ctrl+C para detener.")

    try:
        with sd.InputStream(
            device=device,
            channels=channels,
            samplerate=sample_rate,
            blocksize=1024,
            dtype="float32",
            callback=audio_callback,
        ):
            while not stop_event.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nDetenido.")
    finally:
        stop_event.set()
        reader.join(timeout=1)


if __name__ == "__main__":
    main()
