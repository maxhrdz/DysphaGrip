import argparse
import os
import tempfile
import time
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from serial.tools import list_ports

if os.environ.get("DYSPHAGRIP_SNAPSHOT"):
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from event_detector_live import (
    RECENT_EVENT_LIMIT, SwallowEventDetector, choose_arduino_port, read_arduino,
)
from label_capture import LabelCapture
from validation import CATEGORIES, ExperimentLog, format_summary

WINDOW_SECONDS = 5
INK = "#e7f4f4"
MUTED = "#8eaeae"
TEAL = "#1aa6a6"
AMBER = "#e2a63b"
BG = "#0e1c22"
PANEL = "#143039"
AXIS_COLORS = ("#5ec8c8", "#8eb6ff", "#e0b07a")
LABEL_COLOR = "#b7a1ff"
LABEL_NAMES = {
    "1": "Saliva", "2": "Agua", "3": "Hablar", "4": "Movimiento", "5": "Toser",
}


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
    audio_status = signal_status(now, last_audio_at)
    imu_status = signal_status(now, last_imu_at)
    if detector_view["ready"]:
        if audio_status == imu_status == "Connected":
            calibration = "Detection active"
        else:
            calibration = "Waiting for signal"
    else:
        remaining = max(
            0.0,
            detector_view["calibration_seconds"] - (now - detector_view["started"]),
        )
        calibration = f"Calibrating ({remaining:.0f}s)"

    lines = [
        f"Microphone    {audio_status}",
        f"MPU6050       {imu_status}",
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


def create_dashboard(experiment_mode=False):
    fig = plt.figure(figsize=(14.5, 8.2), facecolor=BG)
    grid = fig.add_gridspec(
        3, 4, width_ratios=(1.15, 1.15, 1.15, 0.95),
        left=0.05, right=0.98, top=0.83, bottom=0.07, hspace=0.42, wspace=0.28,
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
    label_status = fig.text(
        0.49, 0.955,
        ("Etiqueta con tecla o botón · Audio y movimiento ±2 s"
         if experiment_mode else "Modo monitor · Para etiquetar, inicia con --experiment"),
        color=LABEL_COLOR if experiment_mode else MUTED,
        fontsize=11, va="top",
    )
    buttons = {}
    if experiment_mode:
        for index, (key, name) in enumerate(LABEL_NAMES.items()):
            area = fig.add_axes([0.49 + index * 0.099, 0.873, 0.093, 0.042])
            button = Button(area, f"{key} · {name}", color=PANEL, hovercolor="#345558")
            button.label.set_color(INK)
            button.label.set_fontsize(10)
            buttons[key] = button

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
        "label_status": label_status,
        "label_buttons": buttons,
        "event_artists": [],
    }


def redraw(
    dashboard, audio, imu_rows, detector_view, last_audio_at, last_imu_at,
    now=None, experiment_note=None, manual_marks=(),
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
    for label, key in manual_marks:
        position = label["time"] - now
        if -WINDOW_SECONDS <= position <= 0:
            for axis in dashboard["signal_axes"]:
                dashboard["event_artists"].append(axis.axvline(
                    position, color=LABEL_COLOR, lw=1.5, linestyle="--",
                ))
            dashboard["event_artists"].append(dashboard["audio_axis"].annotate(
                f"#{label['id']} {LABEL_NAMES[key]}", xy=(position, 0.04),
                xycoords=("data", "axes fraction"), xytext=(-3, 0),
                textcoords="offset points", ha="right", color=LABEL_COLOR,
                fontsize=8, clip_on=True,
            ))

    dashboard["counter_text"].set_text(str(detector_view["event_count"]))
    dashboard["history_text"].set_text("\n".join(status_lines(
        now, detector_view, last_audio_at, last_imu_at,
        experiment_note=experiment_note,
    )))


def detector_view(detector):
    with detector.lock:
        return {
            "events": detector.events[-RECENT_EVENT_LIMIT:],
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
    if not np.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("--tolerance debe ser un número finito mayor o igual a cero.")

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

    port_name = choose_arduino_port(list(list_ports.comports()))
    detector = SwallowEventDetector(
        csv_path=os.environ.get("DYSPHAGRIP_EVENTS", "dysphagrip_events.csv"),
        keep_session_events=args.experiment,
    )
    experiment = None
    capture = None
    audio_clock_offset = None
    if args.experiment:
        session = experiment_directory()
        experiment = ExperimentLog(session, tolerance=args.tolerance)
        capture = LabelCapture(session, sample_rate)
        print("Experimental testing. Not a dysphagia diagnosis.")
        print("After the 5 second calibration, do one action and press its key:")
        print("  1 swallowing saliva")
        print("  2 swallowing water")
        print("  3 speaking")
        print("  4 head movement")
        print("  5 coughing")
        print("Cada marca guarda 2 segundos de audio y movimiento antes y después.")
        print("Deja al menos 5 segundos entre acciones y espera 2 segundos tras la última marca.")
        print(f"Match tolerance: {args.tolerance:.1f} s. Report: {session}")

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_clock_offset
        if stop_event.is_set():
            return
        stamp = time.monotonic()
        if status:
            print(status)
        chunk = np.array(indata[:, 0], copy=True)
        seen["audio"] = stamp
        if capture is not None:
            adc = getattr(time_info, "inputBufferAdcTime", 0.0)
            current = getattr(time_info, "currentTime", 0.0)
            if 0 < adc <= current and np.isfinite(current):
                # Anchor the stream clock once so callback scheduling jitter does
                # not shift otherwise contiguous blocks on the label timeline.
                if audio_clock_offset is None:
                    audio_clock_offset = stamp - current
                block_start = adc + audio_clock_offset
            else:
                block_start = stamp - frames / sample_rate
            capture.add_audio(chunk, block_start)
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

    def on_imu(values):
        if stop_event.is_set():
            return
        stamp = time.monotonic()
        if capture is not None:
            capture.add_imu(stamp, values)
        with imu_lock:
            seen["imu"] = stamp
            imu_data.append((stamp, *values))
        detector.on_imu(values[:3], values[3:], now=stamp)

    reader_error = None

    def monitor_arduino():
        nonlocal reader_error
        try:
            read_arduino(port_name, stop_event, on_imu)
        except Exception as error:
            reader_error = error
            stop_event.set()

    dashboard = create_dashboard(experiment_mode=experiment is not None)
    recent_marks = deque()

    def update():
        if stop_event.is_set():
            plt.close(dashboard["fig"])
            return False
        with audio_lock:
            audio = audio_buffer.copy()
        with imu_lock:
            rows = list(imu_data)
        now = time.monotonic()
        if capture is not None:
            capture.flush(now)
        while recent_marks and recent_marks[0][0]["time"] < now - WINDOW_SECONDS:
            recent_marks.popleft()
        redraw(
            dashboard, audio, rows, detector_view(detector),
            seen["audio"], seen["imu"],
            now=now, manual_marks=recent_marks,
        )
        dashboard["fig"].canvas.draw_idle()

    if experiment is not None:
        def mark_action(key):
            chosen = CATEGORIES.get(key)
            if chosen is None or stop_event.is_set():
                return
            category, title = chosen
            label = experiment.add_label(category)
            capture.mark(label)
            recent_marks.append((label, key))
            clock = label["timestamp"].split("T")[1][:8]
            dashboard["label_status"].set_text(
                f"Marca #{label['id']} guardada · {LABEL_NAMES[key]} · {clock}"
            )
            update()
            print(f"Label {label['id']}: {title}")

        dashboard["fig"].canvas.mpl_connect(
            "key_press_event", lambda event: mark_action(event.key)
        )
        for key, button in dashboard["label_buttons"].items():
            button.on_clicked(lambda _event, key=key: mark_action(key))

    timer = dashboard["fig"].canvas.new_timer(interval=100)
    timer.add_callback(update)
    snapshot_path = os.environ.get("DYSPHAGRIP_SNAPSHOT")
    reader = threading.Thread(target=monitor_arduino, daemon=True)
    reader.start()
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
                    update()
                    time.sleep(0.1)
                if not stop_event.is_set():
                    dashboard["fig"].savefig(
                        snapshot_path, dpi=130,
                        facecolor=dashboard["fig"].get_facecolor(),
                    )
                    print(f"Snapshot: {snapshot_path}")
            elif not stop_event.is_set():
                timer.start()
                plt.show()
    finally:
        stop_event.set()
        timer.stop()
        reader.join(timeout=1)
        plt.close(dashboard["fig"])
        if experiment is not None:
            with detector.lock:
                session_events = detector.events.copy()
            ended = time.monotonic()
            try:
                capture.finish(ended)
            finally:
                summary = experiment.finish(session_events, ended=ended)
                print(format_summary(summary))
    if reader_error is not None:
        raise SystemExit(f"Arduino: {reader_error}") from reader_error
    print("DysphaGrip stopped.")


def experiment_directory(root=Path("experiments")):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{stamp}_", dir=root))


if __name__ == "__main__":
    main()
