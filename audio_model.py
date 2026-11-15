"""Experimental audio-only swallow candidate classifier, trained on manual labels."""

import json
import math
import tempfile
from pathlib import Path

import numpy as np


WINDOW_SECONDS = 1.0
HOP_SECONDS = 0.1
FEATURE_NAMES = (
    "log_rms", "log_peak_frame_rms", "peak_to_median_frame_db",
    "active_frame_fraction", "energy_fraction_0_150_hz",
    "energy_fraction_150_600_hz", "energy_fraction_600_2000_hz",
)
MODEL_VERSION = 1


def _sample_rate(value):
    rate = int(value)
    if rate != value or rate < 25:
        raise ValueError("sample rate must be an integer of at least 25 Hz")
    return rate


def extract_features(samples, sample_rate):
    """Describe one full second of normalized mono audio ending at decision time."""
    rate = _sample_rate(sample_rate)
    audio = np.asarray(samples, dtype=float)
    if audio.shape != (rate,) or not np.all(np.isfinite(audio)):
        raise ValueError("features require one second of finite mono audio")
    audio = audio - audio.mean()
    # Short-time energy distinguishes a brief impulse from sustained activity.
    # Splitting into 25 frames also works at sample rates not divisible by 25.
    edges = np.arange(26) * rate // 25
    frame_power = np.add.reduceat(audio * audio, edges[:-1]) / np.diff(edges)
    peak = frame_power.max()
    floor = 1e-12
    log_power = 10 * np.log10(max(np.mean(audio * audio), floor))
    log_peak = 10 * np.log10(max(peak, floor))
    median_db = 10 * np.log10(max(np.median(frame_power), floor))
    active = np.mean(frame_power > max(floor, peak / 4))

    # With a one-second window, the Fourier bin spacing is exactly 1 Hz.
    spectrum = np.abs(np.fft.rfft(audio * np.hanning(rate))) ** 2
    total = max(float(spectrum.sum()), 1e-20)
    bands = [spectrum[low:high].sum() / total
             for low, high in ((0, 150), (150, 600), (600, 2000))]
    return np.array([log_power, log_peak, log_peak - median_db, active, *bands])


def _sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -50, 50)))


class AudioModel:
    """Class-balanced, L2-regularized logistic regression; scores are uncalibrated."""

    def __init__(self, mean, scale, weights, intercept, training_sessions=()):
        self.mean, self.scale, self.weights = (
            np.array(values, dtype=float, copy=True) for values in (mean, scale, weights)
        )
        for values in (self.mean, self.scale, self.weights):
            if values.shape != (len(FEATURE_NAMES),) or not np.all(np.isfinite(values)):
                raise ValueError("invalid model feature vector")
        self.intercept = float(intercept)
        if (self.scale <= 0).any() or not math.isfinite(self.intercept):
            raise ValueError("invalid model scaling or intercept")
        if (not isinstance(training_sessions, (list, tuple, set)) or
                not all(isinstance(s, str) and s for s in training_sessions)):
            raise ValueError("training sessions must be a list of session identifiers")
        self.training_sessions = tuple(sorted(set(training_sessions)))

    @classmethod
    def fit(cls, X, y, sessions=None):
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)
        if (features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES)
                or not np.all(np.isfinite(features))):
            raise ValueError("training requires a matrix of finite audio features")
        if labels.shape != (len(features),) or set(labels.tolist()) != {0, 1}:
            raise ValueError("training requires both positive and negative labels")
        labels = labels.astype(float)
        # Give each class half the training weight irrespective of mark counts.
        weights = np.where(labels == 1, .5 / labels.sum(), .5 / (len(labels) - labels.sum()))
        mean = np.average(features, axis=0, weights=weights)
        scale = np.sqrt(np.average((features - mean) ** 2, axis=0, weights=weights))
        scale[scale < 1e-8] = 1.0
        design = np.column_stack(((features - mean) / scale, np.ones(len(labels))))
        coefficients = np.zeros(design.shape[1])
        penalty = np.full(design.shape[1], 0.1)
        penalty[-1] = 0.0  # Do not penalize the intercept.
        # The Lipschitz bound gives a fixed step for the convex logistic loss.
        covariance = (design.T * weights) @ design
        step = 1.0 / (0.25 * np.linalg.eigvalsh(covariance)[-1] + penalty.max())
        for _ in range(600):
            errors = _sigmoid(design @ coefficients) - labels
            gradient = design.T @ (weights * errors) + penalty * coefficients
            coefficients -= step * gradient
            if np.max(np.abs(gradient)) < 1e-7:
                break
        return cls(mean, scale, coefficients[:-1], coefficients[-1],
                   () if sessions is None else sessions)

    def score(self, features):
        row = np.asarray(features, dtype=float)
        if row.shape != self.mean.shape or not np.all(np.isfinite(row)):
            raise ValueError("score requires one finite audio feature vector")
        return float(_sigmoid(((row - self.mean) / self.scale) @ self.weights + self.intercept))

    def save(self, path):
        path = Path(path)
        payload = {
            "version": MODEL_VERSION, "features": list(FEATURE_NAMES),
            "window_seconds": WINDOW_SECONDS, "training_sessions": list(self.training_sessions),
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "intercept": self.intercept,
        }
        # Keep the previous model usable if an update is interrupted.
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / "model.json"
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("version") != MODEL_VERSION
                or data.get("features") != list(FEATURE_NAMES)
                or data.get("window_seconds") != WINDOW_SECONDS):
            raise ValueError("unsupported audio model or feature schema")
        try:
            return cls(data["mean"], data["scale"], data["weights"], data["intercept"],
                       data["training_sessions"])
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete audio model") from exc


class AudioStreamDetector:
    """Evaluate overlapping one-second windows and emit the first hit in each run."""

    def __init__(self, model, sample_rate, threshold=0.5, cooldown=1.5):
        self.model = model
        self.sample_rate = _sample_rate(sample_rate)
        self.threshold = float(threshold)
        self.cooldown = float(cooldown)
        if not 0 < self.threshold < 1 or not math.isfinite(self.cooldown) or self.cooldown < 0:
            raise ValueError("invalid threshold or cooldown")
        self.hop = max(1, round(HOP_SECONDS * self.sample_rate))
        self.reset()

    def reset(self):
        self.buffer = np.empty(0, dtype=np.float32)
        self.expected_start = None
        self.last_score = None
        self.active = False
        self.last_event = -math.inf

    def feed(self, samples, start_time):
        """Accept a contiguous mono block and the acquisition time of its first sample."""
        block = np.asarray(samples, dtype=np.float32)
        if block.ndim != 1 or not np.all(np.isfinite(block)) or not math.isfinite(start_time):
            self.reset()
            raise ValueError("stream requires finite mono audio and acquisition times")
        if not len(block):
            return []
        if self.expected_start is not None:
            if abs(start_time - self.expected_start) > max(2 / self.sample_rate, 0.002):
                self.reset()
        self.expected_start = start_time + len(block) / self.sample_rate
        # Anchor to the most recent acquisition time: a slow device clock drift
        # across many uninterrupted blocks must not look like a dropped block.
        buffer_start = start_time - len(self.buffer) / self.sample_rate
        audio = np.concatenate((self.buffer, block))
        events = []
        offset = 0
        while offset + self.sample_rate <= len(audio):
            end = offset + self.sample_rate
            stamp = buffer_start + end / self.sample_rate
            self.last_score = self.model.score(extract_features(audio[offset:end], self.sample_rate))
            above = self.last_score >= self.threshold
            if above and not self.active and stamp - self.last_event >= self.cooldown:
                events.append(stamp)
                self.last_event = stamp
            self.active = above
            offset += self.hop
        # Copy the short tail to avoid retaining a whole offline recording.
        self.buffer = audio[offset:].copy()
        return events
