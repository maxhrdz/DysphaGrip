import argparse
import os
import re
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import serial
from serial.tools import list_ports

if os.environ.get("DYSPHAGRIP_SNAPSHOT"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from event_detector_live import SwallowEventDetector
from validation import CATEGORIES, ExperimentLog, format_summary

WINDOW_SECONDS = 5
INK = "#e7f4f4"
MUTED = "#8eaeae"
TEAL = "#1aa6a6"
AMBER = "#e2a63b"
BG = "#0e1c22"
PANEL = "#143039"
AXIS_COLORS = ("#5ec8c8", "#8eb6ff", "#e0b07a")

# Parse the existing Arduino output. The firmware is unchanged.
IMU_LINE = re.compile(
    r"Tilt X:\s*([-\d.]+)\s+"
    r"Y:\s*([-\d.]+)\s+"
    r"Z:\s*([-\d.]+)\s*\|\s*"
    r"Turn X:\s*([-\d.]+)\s+"
    r"Y:\s*([-\d.]+)\s+"
    r"Z:\s*([-\d.]+)"
)


def find_contact_mic():
    """El índice de PortAudio cambia al conectar monitores u otros dispositivos."""
    for index, dev in enumerate(sd.query_devices()):
        name = str(dev["name"]).lower()
        if int(dev["max_input_channels"]) < 1:
            continue
        if "air 192" in name or "m-audio" in name:
            return index, dev
    raise SystemExit(
        "No se encontró el M-Audio AIR 192|6. Conéctalo por USB y vuelve a correr el script."
    )


def signal_status(now, last_seen):
    if last_seen <= 0:
        return "Waiting"
    age = now - last_seen
    if age < 1.2:
        return "Connected"
    return f"No signal ({age:.0f}s)"


def status_lines(now, detector_view, last_audio_at, last_imu_at, experiment_note=None):
    if detector_view["ready"]:
        calibration = "Detection active"
    else:
        remaining = max(
            0.0,
            detector_view["calibration_seconds"] - (now - detector_view["started"]),
        )
        calibration = f"Calibrating ({remaining:.0f}s)"

    lines = [
        f"Microphone    {signal_status(now, last_audio_at)}",
        f"MPU6050       {signal_status(now, last_imu_at)}",
        f"Detector      {calibration}",
    ]
    if experiment_note:
        lines.append(experiment_note)
    lines.extend(["", "Event history"])
    events = detector_view["events"]
    if not events:
        lines.append("No candidate events yet.")
        return lines

    for event in reversed(events[-4:]):
        clock = str(event["timestamp"]).split("T")[-1][:12]
        lines.append(f"#{event['event_id']}  {clock}")
        lines.append(
            f"   RMS {event['rms']:.4f}   IMU {event['imu_activity']:.2f}"
        )
        lines.append("   Possible Swallow Event Detected")
    return lines


def create_dashboard():
    fig = plt.figure(figsize=(14.5, 8.2), facecolor=BG)
    grid = fig.add_gridspec(
        3, 4, width_ratios=(1.15, 1.15, 1.15, 0.95),
        left=0.05, right=0.98, top=0.86, bottom=0.07, hspace=0.42, wspace=0.28,
    )
    signal_axes = [fig.add_subplot(grid[row, :3]) for row in range(3)]
    history_ax = fig.add_subplot(grid[:, 3])
    audio_axis, tilt_axis, turn_axis = signal_axes

    fig.text(
        0.05, 0.955, "DysphaGrip",
        color=INK, fontsize=18, fontweight="medium", ha="left", va="top",
    )
    fig.text(
        0.05, 0.915,
        "Experimental monitor. Candidate events only — not a medical assessment.",
        color=MUTED, fontsize=9, ha="left", va="top",
    )

    audio_line, = audio_axis.plot([], [], color=TEAL, lw=1.1)
    tilt_lines = [
        tilt_axis.plot([], [], color=color, lw=1.05, label=axis)[0]
        for axis, color in zip(("X", "Y", "Z"), AXIS_COLORS)
    ]
    turn_lines = [
        turn_axis.plot([], [], color=color, lw=1.05, label=axis)[0]
        for axis, color in zip(("X", "Y", "Z"), AXIS_COLORS)
    ]
    for axis, title in (
        (audio_axis, "Contact microphone"),
        (tilt_axis, "MPU6050 accelerometer"),
        (turn_axis, "MPU6050 gyroscope"),
    ):
        axis.set_facecolor(PANEL)
        axis.set_title(title, color=INK, loc="left", fontsize=11, pad=8)
        axis.set_xlim(-WINDOW_SECONDS, 0)
        axis.grid(True, color="#ffffff", alpha=0.08)
        axis.tick_params(colors=MUTED, labelsize=8)
        axis.yaxis.label.set_color(MUTED)
        for spine in axis.spines.values():
            spine.set_color("#24555c")
        axis.set_ylabel("Signal")

    tilt_axis.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=8)
    turn_axis.legend(loc="upper right", frameon=False, labelcolor=INK, fontsize=8)
    turn_axis.set_xlabel("Seconds before now", color=MUTED)

    history_ax.set_facecolor(PANEL)
    history_ax.set_xticks([])
    history_ax.set_yticks([])
    for spine in history_ax.spines.values():
        spine.set_color("#24555c")
    counter_text = history_ax.text(
        0.06, 0.96, "0", color=TEAL, fontsize=32, fontweight="medium",
        va="top", ha="left", transform=history_ax.transAxes,
    )
    history_ax.text(
        0.06, 0.86, "Possible events", color=MUTED, fontsize=10,
        va="top", ha="left", transform=history_ax.transAxes,
    )
    history_text = history_ax.text(
        0.06, 0.78, "", color=INK, fontsize=9, va="top", ha="left",
        transform=history_ax.transAxes, linespacing=1.4,
    )
    return {
        "fig": fig,
        "signal_axes": signal_axes,
        "audio_axis": audio_axis,
        "tilt_axis": tilt_axis,
        "turn_axis": turn_axis,
        "audio_line": audio_line,
        "tilt_lines": tilt_lines,
        "turn_lines": turn_lines,
        "counter_text": counter_text,
        "history_text": history_text,
        "event_artists": [],
    }


def redraw(
    dashboard, audio, imu_rows, detector_view, last_audio_at, last_imu_at,
    now=None, experiment_note=None,
):
    now = time.monotonic() if now is None else float(now)
    for artist in dashboard["event_artists"]:
        artist.remove()
    dashboard["event_artists"].clear()

    step = 40
    audio_plot = np.asarray(audio)[::step]
    dashboard["audio_line"].set_data(
        np.linspace(-WINDOW_SECONDS, 0, len(audio_plot)),
        audio_plot,
    )
    peak = max(0.005, float(np.max(np.abs(audio))) * 1.2) if len(audio) else 0.005
    dashboard["audio_axis"].set_ylim(-peak, peak)

    recent = [row for row in imu_rows if now - row[0] <= WINDOW_SECONDS]
    if recent:
        arr = np.asarray(recent, dtype=float)
        relative = arr[:, 0] - now
        for index in range(3):
            dashboard["tilt_lines"][index].set_data(relative, arr[:, index + 1])
            dashboard["turn_lines"][index].set_data(relative, arr[:, index + 4])
        for axis, columns in (
            (dashboard["tilt_axis"], arr[:, 1:4]),
            (dashboard["turn_axis"], arr[:, 4:7]),
        ):
            low = float(np.min(columns))
            high = float(np.max(columns))
            margin = max(0.1, (high - low) * 0.2)
            axis.set_ylim(low - margin, high + margin)

    visible = []
    for event in detector_view["events"]:
        position = float(event["monotonic"]) - now
        if -WINDOW_SECONDS <= position <= 0:
            visible.append(position)
    for position in visible:
        for axis in dashboard["signal_axes"]:
            dashboard["event_artists"].append(
                axis.axvline(position, color=AMBER, lw=1.3, alpha=0.95)
            )
    if visible:
        dashboard["event_artists"].append(dashboard["audio_axis"].text(
            0.02, 0.90, "Possible Swallow Event Detected",
            color=AMBER, fontsize=8, ha="left", va="top",
            transform=dashboard["audio_axis"].transAxes,
        ))

    dashboard["counter_text"].set_text(str(detector_view["event_count"]))
    dashboard["history_text"].set_text("\n".join(status_lines(
        now, detector_view, last_audio_at, last_imu_at,
        experiment_note=experiment_note,
    )))


def detector_view(detector):
    with detector.lock:
        return {
            "events": list(detector.events),
            "ready": detector.ready,
            "event_count": detector.event_count,
            "started": detector.start_time,
            "calibration_seconds": detector.calibration_seconds,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="DysphaGrip live monitor. Add --experiment to label a session.",
    )
    parser.add_argument(
        "--experiment", action="store_true",
        help="record manual labels and score the detector when the window closes",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1.5,
        help="seconds allowed between a label and a detection (default 1.5)",
    )
    args = parser.parse_args(argv)

    audio_device, audio_dev = find_contact_mic()
    sample_rate = int(audio_dev["default_samplerate"])
    audio_channels = int(audio_dev["max_input_channels"])
    print(
        f"Audio: {audio_dev['name']} "
        f"(device {audio_device}, {audio_channels} ch @ {sample_rate} Hz)"
    )

    audio_buffer = np.zeros(sample_rate * WINDOW_SECONDS, dtype=np.float32)
    imu_data = deque(maxlen=1000)
    audio_lock = threading.Lock()
    imu_lock = threading.Lock()
    stop_event = threading.Event()
    seen = {"audio": 0.0, "imu": 0.0}

    ports = list(list_ports.comports())
    for index, port in enumerate(ports):
        print(f"{index}: {port.device} - {port.description}")
    port_name = ports[int(input("Arduino port index: "))].device
    detector = SwallowEventDetector(
        csv_path=os.environ.get("DYSPHAGRIP_EVENTS", "dysphagrip_events.csv")
    )
    experiment = None
    if args.experiment:
        session = experiment_directory()
        experiment = ExperimentLog(session, tolerance=args.tolerance)
        print("Experimental testing. Not a dysphagia diagnosis.")
        print("After the 5 second calibration, do one action and press its key:")
        print("  1 swallowing saliva")
        print("  2 swallowing water")
        print("  3 speaking")
        print("  4 head movement")
        print("  5 coughing")
        print("Leave at least 3 seconds between actions. Close the window to score.")
        print(f"Match tolerance: {args.tolerance:.1f} s. Report: {session}")

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status)
        chunk = np.array(indata[:, 0], copy=True)
        seen["audio"] = time.monotonic()
        with audio_lock:
            if frames >= len(audio_buffer):
                audio_buffer[:] = indata[-len(audio_buffer):, 0]
            else:
                audio_buffer[:-frames] = audio_buffer[frames:]
                audio_buffer[-frames:] = indata[:, 0]
        try:
            detector.on_audio(chunk)
        except Exception as error:
            print("Detector audio error:", error)

    def read_arduino():
        try:
            with serial.Serial(port_name, 115200, timeout=0.2) as arduino:
                print("Arduino connected.")
                while not stop_event.is_set():
                    line = arduino.readline().decode("utf-8", errors="ignore").strip()
                    match = IMU_LINE.search(line)
                    if not match:
                        continue
                    values = [float(value) for value in match.groups()]
                    stamp = time.monotonic()
                    seen["imu"] = stamp
                    with imu_lock:
                        imu_data.append((stamp, *values))
                    try:
                        detector.on_imu(values[:3], values[3:])
                    except Exception as error:
                        print("Detector IMU error:", error)
        except Exception as error:
            print("Arduino error:", error)

    reader = threading.Thread(target=read_arduino, daemon=True)
    reader.start()
    dashboard = create_dashboard()

    def update(_frame):
        with audio_lock:
            audio = audio_buffer.copy()
        with imu_lock:
            rows = list(imu_data)
        redraw(
            dashboard, audio, rows, detector_view(detector),
            seen["audio"], seen["imu"],
            experiment_note=None if experiment is None else experiment.note(),
        )
        return [dashboard["audio_line"], dashboard["history_text"]]

    if experiment is not None:
        def on_key(event):
            chosen = CATEGORIES.get(event.key)
            if chosen is None:
                return
            category, title = chosen
            label = experiment.add_label(category)
            print(f"Label {label['id']}: {title}")

        dashboard["fig"].canvas.mpl_connect("key_press_event", on_key)

    animation = FuncAnimation(
        dashboard["fig"], update, interval=100, cache_frame_data=False,
    )
    snapshot_path = os.environ.get("DYSPHAGRIP_SNAPSHOT")
    try:
        with sd.InputStream(
            device=audio_device,
            channels=audio_channels,
            samplerate=sample_rate,
            blocksize=1024,
            dtype="float32",
            callback=audio_callback,
        ):
            if snapshot_path:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not stop_event.is_set():
                    update(0)
                    dashboard["fig"].canvas.draw()
                    time.sleep(0.1)
                dashboard["fig"].savefig(
                    snapshot_path, dpi=130,
                    facecolor=dashboard["fig"].get_facecolor(),
                )
                print(f"Snapshot: {snapshot_path}")
            else:
                plt.show()
    finally:
        stop_event.set()
        reader.join(timeout=1)
        if experiment is not None:
            summary = experiment.finish(
                detector_view(detector)["events"],
                ended=time.monotonic(),
            )
            print(format_summary(summary))
    del animation
    print("DysphaGrip stopped.")


def experiment_directory():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("experiments") / stamp


if __name__ == "__main__":
    main()
