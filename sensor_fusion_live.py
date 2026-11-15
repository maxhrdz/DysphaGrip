import argparse
import csv
import json
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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from activity_model import ActivityModel, InsufficientMotion, extract_activity_features, extract_voice_features
from motion_cough import MotionCoughSpecialist, extract_motion_cough_features
from audio_model import AudioModel, AudioStreamDetector
from session_audio import SessionAudioRecorder
from event_detector_live import (
    RECENT_EVENT_LIMIT, SwallowEventDetector, choose_arduino_port, read_arduino,
)
from label_capture import LabelCapture
from train_audio_model import compare_session
from validation import CATEGORIES, POSITIVE_CATEGORIES, ExperimentLog, format_summary

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
    "1": "Saliva", "2": "Agua", "3": "Hablar", "4": "Movimiento", "5": "Toser", "6": "Reposo",
}


ACTIVITY_NAMES = {
    "swallowing": "Swallowing", "head_movement": "Head movement", "coughing": "Coughing",
    "speaking": "Speaking", "rest": "Rest", "uncertain": "Uncertain",
}
DEFAULT_ACTIVITY_MODEL = Path(__file__).resolve().parent / "models" / "activity_voice_cough.json"


DEFAULT_MOTION_COUGH = DEFAULT_ACTIVITY_MODEL.with_name("motion_cough.json")


def displayed_activity(category, window_end, previous):
    """Keep one ambiguous update readable after two consecutive speech estimates."""
    if category == "uncertain" and len(previous) == 2:
        (first_end, first), (last_end, last) = previous
        if (first == last == "speaking"
                and 0 < last_end - first_end <= 0.5 and 0 < window_end - last_end <= 0.5):
            return last
    return category


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
            area = fig.add_axes([0.48 + index * 0.085, 0.873, 0.080, 0.042])
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
    activity_text = history_ax.text(0.06, 0.96, "Activity estimate\nWaiting for signal",
                                    color=INK, fontsize=14, va="top", transform=history_ax.transAxes)
    counter_text = history_ax.text(
        0.06, 0.80, "0", color=TEAL, fontsize=19, fontweight="medium",
        va="top", ha="left", transform=history_ax.transAxes,
    )
    history_ax.text(
        0.06, 0.74, "Audio + motion triggers", color=MUTED, fontsize=10,
        va="top", ha="left", transform=history_ax.transAxes,
    )
    history_text = history_ax.text(
        0.06, 0.68, "", color=INK, fontsize=9, va="top", ha="left",
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
        "activity_text": activity_text,
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
            0.02, 0.90, "Audio + motion trigger",
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


def save_activity_report(session, activity_model, motion_cough):
    from train_activity_model import evaluate_activity_session

    if motion_cough is None:
        report = evaluate_activity_session(activity_model, session)
    else:
        from train_motion_cough import evaluate_motion_cough_session
        report = evaluate_motion_cough_session(activity_model, motion_cough, session)
    (session / "activity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(f"Actividades: {metrics['correct']}/{metrics['examples']} marcas correctas, "
          f"{metrics['uncertain']} inciertas. Informe: {session / 'activity_report.json'}")


def save_audio_comparison(session, model):
    report = compare_session(model, session)
    (session / "audio_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    timeline = report["timeline"]
    origin = timeline["origin"]
    # This export also runs on the browser monitor's recording thread.
    fig = Figure(figsize=(11, 4.2), layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    marks = timeline["marks"]
    rows = [
        ("Marcas de deglución", [m["time"] for m in marks if m["category"] in POSITIVE_CATEGORIES], "#175d82"),
        ("Marcas de control", [m["time"] for m in marks if m["category"] not in POSITIVE_CATEGORIES], "#696b75"),
        ("Regla de los sensores", timeline["rule"], "#ac6300"),
        ("Modelo de audio", timeline["audio"], "#21835e"),
    ]
    for i, (_, times, color) in enumerate(rows):
        axis.scatter([t - origin for t in times], [i] * len(times), color=color,
                     marker="|", s=150, linewidths=2)
    intervals = timeline["intervals"]
    for (_, end), (start, _) in zip(intervals, intervals[1:]):
        axis.axvspan(end - origin, start - origin, color="#c8c8c8", alpha=0.5)
    axis.set_xlim(-0.1, intervals[-1][1] - origin + 0.1)
    axis.set_yticks(range(len(rows)), [row[0] for row in rows])
    axis.set_ylim(3.6, -0.6)
    axis.set_xlabel("Segundos desde el inicio del primer tramo evaluado")
    axis.set_title("Marcas manuales y candidatos en una sesión independiente")
    axis.grid(axis="x", alpha=0.2)
    fig.savefig(session / "audio_comparison.png", dpi=130)
    print(f"Comparación sobre {report['scored_seconds']:.1f} s. "
          f"Marcas evaluadas: {report['scored_labels']}; fuera de cobertura: {report['excluded_labels']}.")
    for key, name in (("detector_actual", "Regla"), ("modelo_audio", "IA de audio")):
        metrics = report[key]
        print(f"{name}: {metrics['true_positives']} aciertos, {metrics['false_negatives']} omisiones, "
              f"{metrics['false_positives_per_minute']:.2f} falsas alarmas/min")
    print(f"Comparación guardada en {session / 'audio_comparison.png'}")


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
    parser.add_argument("--audio-model", type=Path, metavar="MODELO.json",
                        help="mostrar también los candidatos del modelo piloto de audio")
    parser.add_argument("--record-session", action="store_true",
                        help="activar el experimento y guardar audio continuo para comparar ambos detectores")
    parser.add_argument("--activity-model", type=Path, default=DEFAULT_ACTIVITY_MODEL,
                        help="clasificador en vivo de actividades de audio y movimiento")
    motion_options = parser.add_mutually_exclusive_group()
    motion_options.add_argument("--motion-cough-model", type=Path, help="ajuste para movimiento de cabeza y tos")
    motion_options.add_argument("--no-motion-cough", action="store_true", help="usar solo el modelo anterior")
    args = parser.parse_args(argv)
    args.experiment = args.experiment or args.record_session
    try:
        activity_model = ActivityModel.load(args.activity_model)
        motion_path = args.motion_cough_model
        if not args.no_motion_cough and motion_path is None and args.activity_model.resolve() == DEFAULT_ACTIVITY_MODEL:
            motion_path = DEFAULT_MOTION_COUGH
        motion_cough = MotionCoughSpecialist.load(motion_path) if motion_path else None
        model = AudioModel.load(args.audio_model) if args.audio_model else None
        if not set(activity_model.classes).issubset(ACTIVITY_NAMES):
            raise ValueError("El modelo contiene categorías de actividad desconocidas.")
    except (ValueError, OSError) as error:
        parser.error(f"No se pudo abrir el modelo: {error}")
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

    audio_ai = AudioStreamDetector(model, sample_rate) if model is not None else None
    # The UI drains these blocks outside PortAudio. Timestamps expose any gap
    # when an unusually slow redraw exhausts the five-second backlog.
    ai_pending = deque(maxlen=max(2, int(np.ceil(sample_rate * 5 / 1024))))
    ai_count = 0
    ai_recent = deque()
    recording = None
    last_block_end = None
    valid_frames = 0
    last_classified = None
    activity_predictions = []
    voice_history = deque(maxlen=2)
    screen_estimate = (None, "waiting", "waiting")
    estimates_at_marks = []

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
        print("  6 rest")
        if args.record_session:
            print("Se guardará audio continuo local mientras el detector y ambos sensores estén activos.")
        print("Cada marca guarda 2 segundos de audio y movimiento antes y después.")
        print("Deja al menos 5 segundos entre acciones y espera 2 segundos tras la última marca.")
        print(f"Match tolerance: {args.tolerance:.1f} s. Report: {session}")

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_clock_offset, last_block_end, valid_frames
        if stop_event.is_set():
            return
        stamp = time.monotonic()
        if status:
            print(status)
        chunk = np.array(indata[:, 0], copy=True)
        seen["audio"] = stamp
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
        if capture is not None:
            capture.add_audio(chunk, block_start)
        if detector.ready and signal_status(stamp, seen["imu"]) == "Connected" and not status:
            if recording is not None:
                recording.add_audio(chunk, block_start)
            if audio_ai is not None:
                with audio_lock:
                    ai_pending.append((block_start, chunk))
        with audio_lock:
            if last_block_end is None or abs(block_start - last_block_end) > 0.002:
                valid_frames = 0
            last_block_end = block_start + frames / sample_rate
            valid_frames = 0 if status else min(sample_rate, valid_frames + frames)
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
    if audio_ai is not None:
        dashboard["ai_status"] = dashboard["fig"].text(
            0.05, 0.873, "IA piloto (audio) · esperando señal", fontsize=9, color="#80df98",
        )
    recent_marks = deque()

    def update():
        nonlocal ai_count, last_classified, screen_estimate
        if stop_event.is_set():
            plt.close(dashboard["fig"])
            return False
        with audio_lock:
            audio = audio_buffer.copy()
            audio_end = last_block_end
            has_window = valid_frames >= sample_rate
            pending = list(ai_pending)
            ai_pending.clear()
        for block_start, chunk in pending:
            times = audio_ai.feed(chunk, block_start)
            ai_count += len(times)
            ai_recent.extend(times)
        with imu_lock:
            rows = list(imu_data)
        now = time.monotonic()
        if (not has_window or audio_end is None or not detector.ready or
                any(signal_status(now, last) != "Connected" for last in seen.values())):
            voice_history.clear()
            screen_estimate = (None, "waiting", "waiting")
            dashboard["activity_text"].set_text("Activity estimate\nWaiting for signal")
        elif last_classified is None or audio_end - last_classified >= 0.20:
            last_classified = audio_end
            category, score = "uncertain", None
            if sample_rate >= 25:
                try:
                    window = audio[-sample_rate:]
                    features = extract_activity_features(window, sample_rate, rows, audio_end)
                    voice = extract_voice_features(window, sample_rate) if activity_model.has_voice_specialist else None
                    if motion_cough is None or sample_rate < 50:
                        category, score = activity_model.predict(features, voice_features=voice)
                    else:
                        scores = activity_model.scores(features, voice)
                        motion_features = extract_motion_cough_features(window, sample_rate, rows, audio_end)
                        category, score = motion_cough.predict(scores, motion_features)
                except InsufficientMotion:
                    voice_history.clear()
            shown = displayed_activity(category, audio_end, voice_history)
            if shown != category and now - voice_history[-1][0] > 0.5:
                shown = category
            voice_history.append((audio_end, category))
            suffix = " (recent)" if shown != category else ""
            screen_estimate = (audio_end, category, shown)
            dashboard["activity_text"].set_text(f"Activity estimate\n{ACTIVITY_NAMES[shown]}{suffix}")
            if args.record_session:
                activity_predictions.append((audio_end, category, score, shown))
        if screen_estimate[1] != screen_estimate[2] and voice_history and now - voice_history[0][0] > 0.5:
            screen_estimate = (screen_estimate[0], "uncertain", "uncertain")
            dashboard["activity_text"].set_text("Activity estimate\nUncertain")
        if capture is not None:
            capture.flush(now)
        while recent_marks and recent_marks[0][0]["time"] < now - WINDOW_SECONDS:
            recent_marks.popleft()
        redraw(
            dashboard, audio, rows, detector_view(detector),
            seen["audio"], seen["imu"],
            now=now, manual_marks=recent_marks,
        )
        if audio_ai is not None:
            while ai_recent and ai_recent[0] < now - WINDOW_SECONDS:
                ai_recent.popleft()
            for candidate in ai_recent:
                dashboard["event_artists"].append(dashboard["audio_axis"].axvline(
                    candidate - now, color="#80df98", ls=":", lw=1.6,
                ))
            dashboard["ai_status"].set_text(f"IA piloto (audio) · candidatos: {ai_count} · línea verde")
        dashboard["fig"].canvas.draw_idle()

    if experiment is not None:
        def mark_action(key):
            chosen = CATEGORIES.get(key)
            if chosen is None or stop_event.is_set():
                return
            if args.record_session and (not detector.ready or any(
                signal_status(time.monotonic(), last) != "Connected" for last in seen.values()
            )):
                dashboard["label_status"].set_text("Espera la calibración y la señal de ambos sensores.")
                dashboard["fig"].canvas.draw_idle()
                return
            category, title = chosen
            label = experiment.add_label(category)
            if args.record_session:
                estimates_at_marks.append((label["id"], label["time"], category, *screen_estimate))
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
        if args.record_session:
            recording = SessionAudioRecorder(session, sample_rate)
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
                try:
                    capture.finish(ended)
                finally:
                    if recording is not None:
                        recording.close()
            finally:
                summary = experiment.finish(session_events, ended=ended)
                print(format_summary(summary))
                if args.record_session:
                    with (session / "activity_predictions.csv").open("w", newline="") as out:
                        writer = csv.writer(out)
                        writer.writerow(["window_end_monotonic", "estimated_activity", "score", "displayed_activity"])
                        writer.writerows(activity_predictions)
                    with (session / "activity_at_marks.csv").open("w", newline="") as out:
                        writer = csv.writer(out)
                        writer.writerow(["label_id", "mark_monotonic", "manual_activity",
                                         "window_end_monotonic", "estimated_activity", "displayed_activity"])
                        writer.writerows(estimates_at_marks)
    if args.record_session:
        try:
            save_activity_report(session, activity_model, motion_cough)
        except (OSError, ValueError) as error:
            print(f"Evaluación de actividades pendiente: {error}")
    if args.record_session and model is not None:
        try:
            save_audio_comparison(session, model)
        except (OSError, ValueError) as error:
            print(f"Comparación pendiente: {error}")
    if reader_error is not None:
        raise SystemExit(f"Arduino: {reader_error}") from reader_error
    print("DysphaGrip stopped.")


def experiment_directory(root=Path("experiments")):
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(tempfile.mkdtemp(prefix=f"{stamp}_", dir=root))


if __name__ == "__main__":
    import sys
    options = sys.argv[1:]
    if options in ([], ["--record-session"]):
        from local_dashboard import run
        try:
            run()
        except KeyboardInterrupt:
            pass
    else:
        main([arg for arg in options if arg != "--desktop"])
