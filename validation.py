"""Controlled checks for the existing swallow-event detector.

A result here is not a dysphagia diagnosis and does not detect aspiration.
Positive labels are saliva and water swallows. Speaking, head movement, and
coughing are controls: a detection lined up with one of those is a false positive.
"""

import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from statistics import median


CATEGORIES = {
    "1": ("swallowing_saliva", "Swallowing saliva"),
    "2": ("swallowing_water", "Swallowing water"),
    "3": ("speaking", "Speaking"),
    "4": ("head_movement", "Head movement"),
    "5": ("coughing", "Coughing"),
    "6": ("rest", "Rest"),
}

POSITIVE_CATEGORIES = {"swallowing_saliva", "swallowing_water"}

DISCLAIMER = (
    "Candidate swallow-event check only. "
    "Not a dysphagia diagnosis and not an aspiration test."
)


def match_events(predictions, labels, tolerance):
    """One-to-one match by smallest absolute time gap within tolerance.

    Each prediction and each label is used at most once. The returned latency
    is prediction time minus label time, in seconds.
    """
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    candidates = []
    for prediction_index, prediction in enumerate(predictions):
        for label_index, label in enumerate(labels):
            latency = float(prediction["time"]) - float(label["time"])
            gap = abs(latency)
            if gap <= tolerance:
                candidates.append((gap, prediction_index, label_index, latency))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    used_predictions = set()
    used_labels = set()
    matches = []
    for _gap, prediction_index, label_index, latency in candidates:
        if prediction_index in used_predictions or label_index in used_labels:
            continue
        used_predictions.add(prediction_index)
        used_labels.add(label_index)
        label = labels[label_index]
        positive = label["category"] in POSITIVE_CATEGORIES
        matches.append({
            "prediction_index": prediction_index,
            "label_index": label_index,
            "prediction_id": prediction_id(predictions[prediction_index]),
            "label_id": label["id"],
            "category": label["category"],
            "latency": latency,
            "kind": "true_positive" if positive else "false_positive_control",
        })

    true_positives = [item for item in matches if item["kind"] == "true_positive"]
    false_positive_controls = [
        item for item in matches if item["kind"] == "false_positive_control"
    ]
    unmatched_predictions = [
        predictions[index]
        for index in range(len(predictions))
        if index not in used_predictions
    ]
    false_negatives = [
        labels[index]
        for index in range(len(labels))
        if index not in used_labels and labels[index]["category"] in POSITIVE_CATEGORIES
    ]
    controls_without_detection = [
        labels[index]
        for index in range(len(labels))
        if index not in used_labels and labels[index]["category"] not in POSITIVE_CATEGORIES
    ]
    return {
        "matches": matches,
        "true_positives": true_positives,
        "false_positives": false_positive_controls + unmatched_predictions,
        "false_negatives": false_negatives,
        "controls_without_detection": controls_without_detection,
    }


def prediction_id(prediction):
    return prediction.get("id", prediction.get("event_id"))


def score_matches(matched, duration_seconds):
    """Precision and recall stay unset when their denominator is zero."""
    if duration_seconds < 0:
        raise ValueError("duration_seconds must be >= 0")

    true_positives = len(matched["true_positives"])
    false_positives = len(matched["false_positives"])
    false_negatives = len(matched["false_negatives"])
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) else None
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) else None
    )
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None if precision is None or recall is None else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    latencies = [item["latency"] for item in matched["true_positives"]]
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives_per_minute": (
            false_positives / (duration_seconds / 60.0)
            if duration_seconds > 0 else None
        ),
        "mean_latency_seconds": (
            sum(latencies) / len(latencies) if latencies else None
        ),
        "median_latency_seconds": median(latencies) if latencies else None,
        "latencies": latencies,
        "controls_without_detection": len(matched["controls_without_detection"]),
    }


def predictions_from_events(events):
    rows = []
    for event in events:
        if "monotonic" not in event:
            continue
        rows.append({
            "id": event["event_id"],
            "time": float(event["monotonic"]),
            "timestamp": event["timestamp"],
            "audio_rms": event["rms"],
            "imu_activity": event["imu_activity"],
        })
    return rows


def _blank(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


class ExperimentLog:
    """Manual labels captured during a live session. Writes no invented events."""

    def __init__(self, directory, tolerance=1.5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.tolerance = float(tolerance)
        self.labels = []
        self.lock = threading.Lock()
        self.started = time.monotonic()
        self.started_at = datetime.now().astimezone().isoformat(timespec="milliseconds")

    def add_label(self, category, when=None):
        if category not in POSITIVE_CATEGORIES and category not in {
            "speaking", "head_movement", "coughing", "rest",
        }:
            raise ValueError(f"Unknown category: {category}")
        moment = time.monotonic() if when is None else float(when)
        with self.lock:
            label = {
                "id": len(self.labels) + 1,
                "category": category,
                "time": moment,
                "timestamp": datetime.now().astimezone().isoformat(
                    timespec="milliseconds"
                ),
            }
            self.labels.append(label)
            self._write_labels()
            return dict(label)

    def note(self):
        with self.lock:
            count = len(self.labels)
            last = self.labels[-1]["category"] if self.labels else "none"
        return f"Experiment   {count} labels   last {last}"

    def label_times(self):
        with self.lock:
            return [label["time"] for label in self.labels]

    def finish(self, detector_events, ended):
        with self.lock:
            labels = [dict(label) for label in self.labels]
        predictions = predictions_from_events(detector_events)
        matched = match_events(predictions, labels, self.tolerance)
        duration = max(0.0, float(ended) - self.started)
        metrics = score_matches(matched, duration)
        summary = {
            "started_at": self.started_at,
            "tolerance_seconds": self.tolerance,
            "duration_seconds": duration,
            "prediction_count": len(predictions),
            "label_count": len(labels),
            "positive_label_count": sum(
                label["category"] in POSITIVE_CATEGORIES for label in labels
            ),
            "control_label_count": sum(
                label["category"] not in POSITIVE_CATEGORIES for label in labels
            ),
            **{key: value for key, value in metrics.items() if key != "latencies"},
            "disclaimer": DISCLAIMER,
        }
        self._write_labels(labels)
        self._write_predictions(predictions)
        self._write_matches(matched["matches"])
        self._write_summary(summary)
        summary["directory"] = str(self.directory)
        summary["latencies"] = metrics["latencies"]
        return summary

    def _write_labels(self, labels=None):
        rows = self.labels if labels is None else labels
        path = self.directory / "ground_truth.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["label_id", "timestamp", "monotonic", "category"])
            for label in rows:
                writer.writerow([
                    label["id"], label["timestamp"],
                    f"{label['time']:.6f}", label["category"],
                ])

    def _write_predictions(self, predictions):
        path = self.directory / "predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "event_id", "timestamp", "monotonic", "audio_rms", "imu_activity",
            ])
            for prediction in predictions:
                writer.writerow([
                    prediction["id"],
                    prediction["timestamp"],
                    f"{prediction['time']:.6f}",
                    prediction["audio_rms"],
                    prediction["imu_activity"],
                ])

    def _write_matches(self, matches):
        path = self.directory / "matches.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([
                "prediction_id", "label_id", "category",
                "latency_seconds", "kind",
            ])
            for item in matches:
                writer.writerow([
                    item["prediction_id"], item["label_id"], item["category"],
                    f"{item['latency']:.6f}", item["kind"],
                ])

    def _write_summary(self, summary):
        path = self.directory / "summary.csv"
        fields = [
            "started_at", "tolerance_seconds", "duration_seconds",
            "prediction_count", "label_count", "positive_label_count",
            "control_label_count", "true_positives", "false_positives",
            "false_negatives", "precision", "recall", "f1",
            "false_positives_per_minute", "mean_latency_seconds",
            "median_latency_seconds", "controls_without_detection",
            "disclaimer",
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: _blank(summary[key]) for key in fields})


def format_summary(summary):
    def show(value, digits=3):
        if value is None or value == "":
            return "n/a"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    return "\n".join([
        "DysphaGrip validation",
        DISCLAIMER,
        f"Duration: {show(summary['duration_seconds'], 1)} s",
        f"Tolerance: {show(summary['tolerance_seconds'], 1)} s",
        (
            f"Labels: {summary['label_count']} "
            f"(swallows {summary['positive_label_count']}, "
            f"other actions {summary['control_label_count']})"
        ),
        f"Predictions: {summary['prediction_count']}",
        f"True positives: {summary['true_positives']}",
        f"False positives: {summary['false_positives']}",
        f"False negatives: {summary['false_negatives']}",
        f"Precision: {show(summary['precision'])}",
        f"Recall: {show(summary['recall'])}",
        f"F1: {show(summary['f1'])}",
        f"False positives per minute: {show(summary['false_positives_per_minute'])}",
        f"Mean detection latency: {show(summary['mean_latency_seconds'])} s",
        f"Saved: {summary['directory']}",
    ])
