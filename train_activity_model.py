"""Train the audio + motion activity classifier and validate it by entire session."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
import wave

import numpy as np

from activity_model import ActivityModel, extract_activity_features, extract_voice_features
from train_audio_model import labeled_audio, read_labels, read_rows, session_details
from validation import POSITIVE_CATEGORIES


ACTIVITIES = ("swallowing", "speaking", "head_movement", "coughing", "rest")
MOTION_FIELDS = ("seconds_from_mark", "tilt_x", "tilt_y", "tilt_z", "turn_x", "turn_y", "turn_z")


@dataclass
class Example:
    mark: int
    time: float
    category: str
    activity: str
    features: np.ndarray
    voice: np.ndarray


def read_examples(session):
    examples = []
    skipped = []
    for label in read_labels(session):
        folder = session / "captures" / f"mark_{label['id']:04d}"
        try:
            audio, rate = labeled_audio(folder, label)
            rows = [[float(row[field]) for field in MOTION_FIELDS]
                    for row in read_rows(folder / "motion.csv")]
            # CSV motion times are relative to the mark. Both modalities end at the
            # mark using only the last second, just as the live decision would.
            features = extract_activity_features(audio, rate, rows, end_time=0.0)
            voice = extract_voice_features(audio, rate)
        except (OSError, ValueError, KeyError, wave.Error) as error:
            skipped.append({"mark": label["id"], "category": label["category"], "reason": str(error)})
            continue
        category = label["category"]
        activity = "swallowing" if category in POSITIVE_CATEGORIES else category
        examples.append(Example(label["id"], label["time"], category, activity, features, voice))
    return examples, skipped


def fit_sessions(groups):
    examples = [example for group in groups.values() for example in group]
    if len({example.activity for example in examples}) < 2:
        raise ValueError("El entrenamiento requiere ejemplos válidos de al menos dos actividades.")
    return ActivityModel.fit(np.array([e.features for e in examples]),
                             [e.activity for e in examples], sessions=sorted(groups))


def fit_specialist(baseline, training):
    examples = [e for session in training.values() for e in session if e.activity in {"speaking", "coughing"}]
    if len({e.activity for e in examples}) != 2:
        return baseline
    return baseline.with_voice_specialist(np.array([e.voice for e in examples]),
                                          [e.activity for e in examples], sessions=sorted(training))


def classification_summary(decisions):
    counts = Counter((row["expected"], row["predicted"]) for row in decisions)
    correct = sum(counts[(category, category)] for category in ACTIVITIES)
    classified = sum(row["predicted"] != "uncertain" for row in decisions)
    per_class = {}
    for category in ACTIVITIES:
        support = sum(n for (expected, _), n in counts.items() if expected == category)
        predictions = sum(n for (_, prediction), n in counts.items() if prediction == category)
        hits = counts[category, category]
        per_class[category] = {
            "examples": support, "correct": hits, "predicted": predictions,
            "uncertain": counts[category, "uncertain"],
            "recall": hits / support if support else None,
            "precision": hits / predictions if predictions else None,
            "confusion": {name: counts[category, name] for name in (*ACTIVITIES, "uncertain")},
        }
    return {
        "examples": len(decisions), "correct": correct, "uncertain": len(decisions) - classified,
        "accuracy_including_uncertain": correct / len(decisions) if decisions else None,
        "coverage": classified / len(decisions) if decisions else None,
        "per_class": per_class,
    }


def evaluate_examples(model, examples):
    decisions = []
    for example in examples:
        predicted, score = model.predict(example.features, example.voice)
        decisions.append({"mark": example.mark, "time": example.time, "category": example.category,
                          "expected": example.activity, "predicted": predicted, "score": float(score)})
    return {
        "classes_in_training": list(model.classes),
        "test_classes_absent_from_training": sorted({e.activity for e in examples} - set(model.classes)),
        "metrics": classification_summary(decisions), "decisions": decisions,
    }


def evaluate_activity_session(model, session):
    session = Path(session)
    identity, _ = session_details(session)
    if not model.training_sessions:
        raise ValueError("El modelo carece de procedencia de entrenamiento.")
    if identity in model.training_sessions:
        raise ValueError("La sesión participó en el entrenamiento. Usa una sesión independiente.")
    examples, skipped = read_examples(session)
    if not examples:
        raise ValueError("La sesión carece de marcas con audio y movimiento utilizables.")
    return {"test_session": identity, "skipped_examples": skipped,
            **evaluate_examples(model, examples)}


def evaluate_pair(baseline, training, test):
    updated = fit_specialist(baseline, training)
    return {"baseline": evaluate_examples(baseline, test),
            "candidate": evaluate_examples(updated, test)}


def validate_latest_split(groups, published=None):
    if not groups:
        return {"skipped_reason": "Sin sesiones utilizables."}
    latest = max(groups, key=datetime.fromisoformat)
    examples = sorted(groups[latest], key=lambda e: e.time)
    boundary = len(examples) * 2 // 3
    if boundary < 1 or boundary >= len(examples):
        return {"skipped_reason": "La sesión más reciente carece de suficientes marcas."}
    early = examples[:boundary]
    # Keep +/- 2-s archived clips disjoint across the split.
    late = [e for e in examples[boundary:] if e.time - early[-1].time >= 4]
    training = dict(groups)
    training[latest] = early
    if not late or len({e.activity for rows in training.values() for e in rows}) < 2:
        return {"skipped_reason": "Sin suficientes clases o marcas después de la separación."}
    split = {"method": "chronological_tail_of_latest_session", "session": latest,
             "other_training_sessions": sorted(key for key in groups if key != latest),
             "training_marks_in_latest": [e.mark for e in early],
             "boundary_gap_seconds": late[0].time - early[-1].time,
             "excluded_near_split": len(examples) - len(early) - len(late)}
    if published is None:
        return {**split, **evaluate_examples(fit_sessions(training), late)}
    if latest in published.training_sessions:
        return {"skipped_reason": "El modelo publicado ya vio la última sesión."}
    return {**split, **evaluate_pair(published, training, late)}


def validate_by_session(groups, compare_specialist=False):
    folds = []
    all_decisions = []
    for held_out, test_examples in groups.items():
        training = {key: rows for key, rows in groups.items() if key != held_out}
        fold = {"test_session": held_out, "training_sessions": sorted(training)}
        if len({e.activity for rows in training.values() for e in rows}) < 2:
            fold["skipped_reason"] = "El resto de sesiones contiene menos de dos clases de entrenamiento."
            folds.append(fold)
            continue
        primary = fit_sessions(training)
        if compare_specialist:
            pair = evaluate_pair(primary, training, test_examples)
            fold["baseline"] = pair["baseline"]
            scored = pair["candidate"]
        else:
            scored = evaluate_examples(primary, test_examples)
        fold.update(scored)
        folds.append(fold)
        all_decisions.extend(scored["decisions"])
    return folds, classification_summary(all_decisions)


def train_activity_sessions(sessions, model_path, report_path, voice_baseline=None):
    model_path, report_path = Path(model_path), Path(report_path)
    if model_path.resolve() == report_path.resolve():
        raise ValueError("El modelo y el informe necesitan archivos distintos.")
    if voice_baseline is not None and Path(voice_baseline).resolve() in {report_path.resolve(), model_path.resolve()}:
        raise ValueError("Elige una salida distinta para conservar el modelo original.")
    groups = {}
    details = {}
    for session in sessions:
        identity, _ = session_details(session)
        if identity in details:
            raise ValueError(f"Sesión duplicada: {session}")
        examples, skipped = read_examples(session)
        details[identity] = {"directory": session.name, "usable_examples": len(examples),
                             "categories": dict(Counter(e.category for e in examples)),
                             "skipped_examples": skipped}
        if examples:
            groups[identity] = examples
    if not groups:
        raise ValueError("Las sesiones no contienen ejemplos utilizables.")
    published = ActivityModel.load(voice_baseline) if voice_baseline else None
    if published is not None and (not published.training_sessions or not set(published.training_sessions) <= groups.keys()):
        raise ValueError("Incluye las sesiones originales del modelo para comprobar su procedencia.")
    folds, metrics = validate_by_session(groups, compare_specialist=published is not None)
    split = validate_latest_split(groups, published)
    previous = {}
    if published is not None:
        baseline_decisions = [d for fold in folds if "baseline" in fold for d in fold["baseline"]["decisions"]]
        previous["paired_baseline_out_of_session"] = classification_summary(baseline_decisions)
        latest = max(groups, key=datetime.fromisoformat)
        if latest not in published.training_sessions:
            historical = {s: rows for s, rows in groups.items() if s != latest}
            previous["published_vs_candidate_latest_session"] = {
                "test_session": latest, "training_sessions": sorted(historical),
                **evaluate_pair(published, historical, groups[latest]),
            }
        model = fit_specialist(published, groups)
    else:
        model = fit_sessions(groups)
    report = {
        "method": "leave_one_session_out",
        "scope": "Clasificación de la ventana de un segundo anterior a cada marca manual.",
        "metric_note": "La exactitud cuenta inciertos como fallos. Estas métricas por marca "
                       "no miden la frecuencia de falsas alarmas de un monitor continuo.",
        "sessions": details, "folds": folds, "out_of_session": metrics,
        "within_latest_session_chronological": split, **previous,
        "final_model": {"training_sessions": list(model.training_sessions), "classes": list(model.classes),
                        "examples": sum(len(examples) for examples in groups.values())},
    }
    if published is not None:
        report["final_model"]["primary_model"] = str(voice_baseline)
        report["final_model"]["primary_training_sessions"] = list(model.primary_training_sessions)
        report["final_model"]["voice_training_sessions"] = list(model.voice_training_sessions)
        report["final_model"]["voice_examples"] = dict(Counter(e.activity for group in groups.values()
                                                               for e in group if e.activity in {"speaking", "coughing"}))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=report_path.parent) as temp:
        report_tmp = Path(temp) / "evaluation.json"
        report_tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                              encoding="utf-8")
        model.save(model_path)
        report_tmp.replace(report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", type=Path, required=True,
                        help="Carpeta de sesión. Repetir por cada sesión completa.")
    parser.add_argument("--model", type=Path, required=True, help="Modelo JSON final")
    parser.add_argument("--report", type=Path, required=True, help="Informe JSON de validación")
    parser.add_argument("--voice-baseline", type=Path,
                        help="Mantener este modelo base y entrenar la distinción de habla y tos")
    args = parser.parse_args(argv)
    try:
        result = train_activity_sessions(args.session, args.model, args.report, args.voice_baseline)
    except (ValueError, OSError, KeyError, wave.Error) as error:
        parser.exit(2, f"Error: {error}\n")
    total = result["out_of_session"]
    print(f"Modelo guardado en {args.model} con {result['final_model']['examples']} marcas.")
    print(f"Evaluación entre sesiones: {total['correct']}/{total['examples']} aciertos, "
          f"{total['uncertain']} inciertos. Informe: {args.report}")


if __name__ == "__main__":
    main()
