"""Train the experimental audio model and compare it on an independent session."""

import argparse
import csv
import json
import math
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from audio_model import AudioModel, AudioStreamDetector, WINDOW_SECONDS, extract_features
from session_audio import METADATA_NAME, TIMESTAMP_TOLERANCE, iter_segments
from validation import CATEGORIES, POSITIVE_CATEGORIES, match_events, score_matches


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def session_details(session):
    rows = read_rows(session / "summary.csv")
    if len(rows) != 1 or not rows[0].get("started_at"):
        raise ValueError(f"Se necesita una sesión finalizada con summary.csv: {session}")
    # The recorder assigns this timestamp once. It survives copying or renaming the folder.
    return rows[0]["started_at"], float(rows[0]["tolerance_seconds"])


def read_labels(session):
    labels = []
    valid = {category for category, _ in CATEGORIES.values()} | {"rest"}
    for row in read_rows(session / "ground_truth.csv"):
        when = float(row["monotonic"])
        if row["category"] not in valid or not math.isfinite(when):
            raise ValueError(f"Etiqueta inválida en {session / 'ground_truth.csv'}")
        labels.append({"id": int(row["label_id"]), "category": row["category"], "time": when})
    return labels


def labeled_features(folder, label):
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (metadata["label_id"] != label["id"] or metadata["category"] != label["category"]
            or not math.isclose(metadata["label_monotonic"], label["time"], abs_tol=1e-5, rel_tol=0)):
        raise ValueError("La marca de la captura difiere de ground_truth.csv")
    # The protocol marks each action immediately afterwards. Use only the last second
    # before the mark, as the online model also sees only past samples.
    start_offset = metadata["wav_start_seconds_from_mark"]
    if start_offset is None:
        raise ValueError("Captura sin audio")
    with wave.open(str(folder / "audio.wav"), "rb") as wav:
        rate = wav.getframerate()
        if (wav.getnchannels(), wav.getsampwidth(), rate) != (1, 4, metadata["sample_rate_hz"]):
            raise ValueError("Se esperaba el WAV mono PCM32 producido por el tablero")
        frames = round(rate * WINDOW_SECONDS)
        left = round((-WINDOW_SECONDS - start_offset) * rate)
        if left < 0 or left + frames > wav.getnframes():
            raise ValueError("La ventana previa a la marca está incompleta")
        wav.setpos(left)
        raw = wav.readframes(frames)
    missing = sum(max(0.0, min(0, b) - max(-WINDOW_SECONDS, a))
                  for a, b in metadata["audio_missing_intervals_seconds_from_mark"])
    # Older captures can contain isolated one-sample holes from ADC timestamp rounding.
    if missing > 0.001:
        raise ValueError("Faltan más de 1 ms de audio dentro de la ventana")
    samples = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2**31
    return extract_features(samples, rate)


def train_sessions(sessions, model_path):
    identities = set()
    features = []
    targets = []
    counts = Counter()
    skipped = []
    for session in sessions:
        identity, _ = session_details(session)
        if identity in identities:
            raise ValueError(f"Sesión de entrenamiento repetida: {session}")
        identities.add(identity)
        for label in read_labels(session):
            folder = session / "captures" / f"mark_{label['id']:04d}"
            try:
                features.append(labeled_features(folder, label))
            except (OSError, wave.Error, ValueError, KeyError) as error:
                skipped.append(f"{folder}: {error}")
                continue
            counts[label["category"]] += 1
            targets.append(int(label["category"] in POSITIVE_CATEGORIES))
    if set(targets) != {0, 1}:
        raise ValueError("Faltan capturas utilizables de deglución o de control para entrenar. "
                         + (skipped[0] if skipped else ""))
    model = AudioModel.fit(np.asarray(features), np.asarray(targets), sessions=sorted(identities))
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return {"training_sessions": sorted(identities), "examples_by_category": dict(counts),
            "examples": len(targets), "skipped": skipped,
            "evaluation": "Pendiente de una sesión independiente con audio continuo."}


def compare_session(model, session):
    identity, tolerance = session_details(session)
    if not model.training_sessions:
        raise ValueError("El modelo carece de procedencia de entrenamiento; vuelve a entrenarlo.")
    if identity in model.training_sessions:
        raise ValueError("La sesión participó en el entrenamiento. Usa una sesión independiente.")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("La tolerancia de la sesión es inválida")
    if not (session / METADATA_NAME).is_file():
        raise ValueError("Esta sesión carece de audio continuo. Graba una sesión nueva con --record-session.")
    labels = read_labels(session)
    baseline = [{"id": int(row["event_id"]), "time": float(row["monotonic"])}
                for row in read_rows(session / "predictions.csv")]
    if not all(math.isfinite(item["time"]) for item in baseline):
        raise ValueError("La sesión contiene predicciones con tiempos inválidos")

    intervals = []
    ai_events = []
    stream = None
    previous_end = None
    previous_rate = None
    for start, samples, rate in iter_segments(session):
        if (previous_end is None or rate != previous_rate or abs(start - previous_end) > TIMESTAMP_TOLERANCE):
            if previous_end is not None and start < previous_end - TIMESTAMP_TOLERANCE:
                raise ValueError("Hay tramos de audio con tiempos solapados")
            stream = AudioStreamDetector(model, rate)
            intervals.append([start + WINDOW_SECONDS, start])
        for when in stream.feed(samples, start):
            ai_events.append({"id": len(ai_events) + 1, "time": when})
        previous_end = start + len(samples) / rate
        previous_rate = rate
        intervals[-1][1] = previous_end
    intervals = [(left, right) for left, right in intervals if right > left]
    duration = sum(right - left for left, right in intervals)
    if not intervals:
        raise ValueError("Falta audio continuo evaluable. Graba una sesión con --record-session.")

    totals = {"detector_actual": {}, "modelo_audio": {}}
    evaluated_labels = []
    timeline = {"origin": intervals[0][0], "marks": [], "rule": [], "audio": [],
                "intervals": intervals}
    for left, right in intervals:
        # Each contiguous stretch has its own matching, even for interruptions shorter
        # than the matching tolerance. Both detectors see the same labels and duration.
        local_labels = [item for item in labels if left <= item["time"] <= right]
        evaluated_labels.extend(local_labels)
        timeline["marks"].extend(local_labels)
        for name, events in (("detector_actual", baseline), ("modelo_audio", ai_events)):
            local_events = [item for item in events if left <= item["time"] <= right]
            timeline_key = "rule" if name == "detector_actual" else "audio"
            timeline[timeline_key].extend(item["time"] for item in local_events)
            matched = match_events(local_events, local_labels, tolerance)
            for key, rows in matched.items():
                totals[name].setdefault(key, []).extend(rows)
    if not evaluated_labels:
        raise ValueError("Ninguna marca manual cae dentro del audio evaluable.")
    report = {"test_session": identity, "timeline": timeline, "scored_seconds": duration,
              "scored_labels": len(evaluated_labels),
              "labels_by_category": dict(Counter(row["category"] for row in evaluated_labels)),
              "excluded_labels": len(labels) - len(evaluated_labels), "tolerance_seconds": tolerance}
    for name, matched in totals.items():
        report[name] = score_matches(matched, duration)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Ajustar un modelo piloto a capturas etiquetadas")
    train.add_argument("--session", action="append", required=True, type=Path,
                       help="Carpeta de entrenamiento. Repetir para cada sesión.")
    train.add_argument("--model", type=Path, required=True, help="Modelo JSON de salida")
    compare = commands.add_parser("compare", help="Puntuar modelo y regla en una sesión reservada")
    compare.add_argument("--session", type=Path, required=True, help="Sesión nueva con --record-session")
    compare.add_argument("--model", type=Path, required=True, help="Modelo JSON entrenado")
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            result = train_sessions(args.session, args.model)
        else:
            result = compare_session(AudioModel.load(args.model), args.session)
    except (OSError, ValueError, wave.Error, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
