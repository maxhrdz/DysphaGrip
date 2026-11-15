"""Train a head-movement/cough specialist while keeping its primary classifier frozen."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

from activity_model import ActivityModel, extract_activity_features, extract_voice_features
from motion_cough import MotionCoughSpecialist, extract_motion_cough_features
from train_activity_model import MOTION_FIELDS, classification_summary
from train_audio_model import labeled_audio, read_labels, read_rows, session_details
from validation import POSITIVE_CATEGORIES


PAIR = {"head_movement", "coughing"}


@dataclass
class Window:
    mark: int
    time: float
    category: str
    activity: str
    scores: dict
    primary_label: str
    primary_score: float
    features: np.ndarray | None


def read_windows(primary, session):
    windows = []
    skipped = []
    for label in read_labels(session):
        folder = session / "captures" / f"mark_{label['id']:04d}"
        try:
            signal, rate = labeled_audio(folder, label)
            motion = [[float(row[field]) for field in MOTION_FIELDS]
                      for row in read_rows(folder / "motion.csv")]
            base = extract_activity_features(signal, rate, motion, 0)
            voice = extract_voice_features(signal, rate)
            scores = primary.scores(base, voice)
            predicted, score = primary.predict(base, voice)
        except (OSError, ValueError, KeyError, wave.Error) as error:
            skipped.append({"mark": label["id"], "scope": "both", "reason": str(error)})
            continue
        try:
            features = extract_motion_cough_features(signal, rate, motion, 0)
        except ValueError as error:
            # Keep the example in comparison: the deployed fallback retains the primary.
            features = None
            skipped.append({"mark": label["id"], "scope": "specialist", "reason": str(error)})
        category = label["category"]
        activity = "swallowing" if category in POSITIVE_CATEGORIES else category
        windows.append(Window(label["id"], label["time"], category, activity, scores, predicted, score, features))
    return windows, skipped


def train_specialist(groups):
    features, labels, sources = [], [], []
    for identity, windows in groups.items():
        eligible = [w for w in windows if w.features is not None and w.activity in PAIR]
        if eligible:
            sources.append(identity)
            features.extend(w.features for w in eligible)
            labels.extend(w.activity for w in eligible)
    if set(labels) != PAIR:
        raise ValueError("Se necesitan capturas utilizables tanto de tos como de movimiento.")
    model = MotionCoughSpecialist.fit(np.asarray(features), labels, sessions=sources)
    return model, dict(Counter(labels))


def compare_windows(specialist, windows):
    before, after, changes = [], [], []
    for w in windows:
        new_label, new_score = specialist.predict(w.scores, w.features)
        mark = {"mark": w.mark, "time": w.time, "expected": w.activity, "category": w.category}
        before.append({**mark, "predicted": w.primary_label, "score": w.primary_score})
        after.append({**mark, "predicted": new_label, "score": float(new_score)})
        if new_label != w.primary_label:
            changes.append({**mark, "before": w.primary_label, "after": new_label})
    return {"test_marks": [w.mark for w in windows], "baseline": classification_summary(before),
            "candidate": classification_summary(after), "changes": changes,
            "decisions_before": before, "decisions_after": after}


def evaluate_motion_cough_session(primary, specialist, session):
    session = Path(session)
    identity, _ = session_details(session)
    if not primary.training_sessions or not specialist.training_sessions:
        raise ValueError("Se necesita la procedencia de ambos modelos para evaluar.")
    if identity in set(primary.training_sessions) | set(specialist.training_sessions):
        raise ValueError("La sesión participó en el entrenamiento. Usa una sesión independiente.")
    windows, skipped = read_windows(primary, session)
    if not windows:
        raise ValueError("Faltan marcas evaluables en esta sesión.")
    result = compare_windows(specialist, windows)
    return {"test_session": identity, "metrics": result["candidate"], "decisions": result["decisions_after"],
            "skipped_examples": skipped, **result}


def train_and_compare(primary_path, test_session, first_test_mark, model_path, report_path, replay=False, development_session=False):
    primary_path, test_session = Path(primary_path), Path(test_session)
    model_path, report_path = Path(model_path), Path(report_path)
    if len({p.resolve() for p in (primary_path, model_path, report_path)}) != 3:
        raise ValueError("El primario, especialista e informe deben tener rutas distintas.")
    primary = ActivityModel.load(primary_path)
    latest, _ = session_details(test_session)
    if not primary.training_sessions or latest in primary.training_sessions:
        raise ValueError("Elige una sesión que no haya visto el clasificador primario.")
    details, groups = {}, {}
    for summary in sorted(test_session.parent.glob("*/summary.csv")):
        session = summary.parent
        identity, _ = session_details(session)
        if datetime.fromisoformat(identity) > datetime.fromisoformat(latest):
            continue
        if identity in details:
            raise ValueError(f"Sesión duplicada: {session}")
        windows, skipped = read_windows(primary, session)
        details[identity] = {"directory": session.name, "usable": len(windows), "skipped": skipped}
        if windows:
            groups[identity] = sorted(windows, key=lambda w: w.time)
    if latest not in groups:
        raise ValueError("La sesión reservada no contiene ventanas evaluables.")
    if not set(primary.training_sessions).issubset(groups):
        raise ValueError("Faltan sesiones de origen del primario o su entrenamiento es posterior a la reserva.")
    older = {key: windows for key, windows in groups.items() if key != latest}
    test = groups[latest]
    cut_time = next((w.time for w in test if w.mark == first_test_mark), None)
    if cut_time is None:
        raise ValueError("La marca inicial de la reserva temporal no está entre las ventanas evaluables.")
    early = [w for w in test if w.time <= cut_time - 4]
    tail = [w for w in test if w.time >= cut_time]
    if not early:
        raise ValueError("Faltan marcas de entrenamiento separadas por al menos 4 s de la reserva.")
    prior_model, prior_counts = train_specialist(older)
    tail_model, tail_counts = train_specialist({**older, latest: early})
    full_result = {"training_examples": prior_counts, "training_sessions": sorted(older),
                   **compare_windows(prior_model, test)}
    tail_result = {"training_examples": tail_counts, "training_marks_in_latest": [w.mark for w in early],
                   "excluded_near_split": [w.mark for w in test if early[-1].time < w.time < cut_time],
                   "gap_seconds": cut_time - early[-1].time, **compare_windows(tail_model, tail)}
    if replay:
        from replay_activity_marks import evaluate_replay
        for specialist, result in ((prior_model, full_result), (tail_model, tail_result)):
            result["replay"] = evaluate_replay(test_session, primary, candidate=specialist,
                                               holdout_marks=result["test_marks"])
    final_model, counts = train_specialist(groups)
    report = {
        "scope": "Clasificación por marca manual, con una ventana causal de un segundo.",
        "used_for_model_design": development_session,
        "evaluation_note": ("Se revisaron resultados de esta sesión al diseñar el especialista. "
                            "La comparación es retrospectiva y requiere una nueva sesión independiente."
                            if development_session else
                            "Los pesos del especialista se ajustaron antes de puntuar las marcas de cada reserva."),
        "primary_model": str(primary_path), "primary_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
        "test_session": latest, "first_test_mark": first_test_mark, "sessions": details,
        "latest_whole_session": full_result, "latest_chronological_block": tail_result,
        "final_model": {"examples": counts, "training_sessions": list(final_model.training_sessions)},
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=report_path.parent) as tmp:
        draft = Path(tmp) / "report.json"
        draft.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        final_model.save(model_path)
        draft.replace(report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--test-session", type=Path, required=True, help="Última sesión a utilizar")
    parser.add_argument("--first-test-mark", type=int, required=True, help="Primera marca de la reserva temporal")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--replay", action="store_true", help="Incluir la reproducción temporal de la interfaz")
    parser.add_argument("--development-session", action="store_true",
                        help="Marcar como retrospectivos los resultados de una sesión usada al diseñar el modelo")
    args = parser.parse_args(argv)
    try:
        report = train_and_compare(args.primary, args.test_session, args.first_test_mark,
                                   args.model, args.report, args.replay, args.development_session)
    except (OSError, ValueError, KeyError, wave.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    for key in ("latest_whole_session", "latest_chronological_block"):
        result = report[key]
        print(f"{key}: {result['baseline']['correct']} → {result['candidate']['correct']} "
              f"aciertos de {len(result['test_marks'])} marcas.")
    print(f"Especialista: {args.model}. Informe: {args.report}.")


if __name__ == "__main__":
    main()
