"""Experimental tie-breaker for unresolved head movement vs. coughing.

The conservative policy was chosen after inspecting a recorded session. Results
on that session describe an exploratory adjustment; validate on a new session.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from activity_model import (
    UNCERTAIN, WINDOW_SECONDS, _decisive, _fit_softmax, _softmax, extract_motion_features,
)


CLASSES = ("coughing", "head_movement")
FEATURE_NAMES = (
    "audio_peak_over_median_db", "audio_rise_fraction", "audio_above_half_peak_fraction",
    "peak_audio_relative_to_gyro_db", "mean_gyro_log", "gyro_direction_consistency",
    "accel_net_change_log", "accel_change_rate_log",
    "audio_300_1200_vs_5_80_db", "audio_1200_5000_vs_5_80_db",
)


def extract_motion_cough_features(audio, sample_rate, motion_rows, end_time):
    """Compare a short audio impulse with the amount and direction of actual motion."""
    rate = int(sample_rate)
    signal = np.asarray(audio, dtype=float)
    if sample_rate != rate or rate < 50 or signal.shape != (rate,) or not np.isfinite(signal).all():
        raise ValueError("expected one second of finite mono audio, sampled at 50 Hz or more")
    motion_features = extract_motion_features(motion_rows, end_time)
    rows = np.asarray(motion_rows, dtype=float)
    rows = rows[(rows[:, 0] >= end_time - WINDOW_SECONDS) & (rows[:, 0] <= end_time)]
    dt = np.diff(rows[:, 0])
    gyro = rows[:, 4:]
    # Net angular displacement divided by the integrated path magnitude:
    # reversals reduce this ratio, while a sustained turn approaches one.
    gyro_magnitude = np.linalg.norm(gyro, axis=1)
    angular_path = dt @ ((gyro_magnitude[1:] + gyro_magnitude[:-1]) / 2)
    angular_vector = dt @ ((gyro[1:] + gyro[:-1]) / 2)
    direction_consistency = np.linalg.norm(angular_vector) / max(angular_path, 1e-12)
    net_accel_change = np.linalg.norm(rows[-1, 1:4] - rows[0, 1:4])

    centered = signal - signal.mean()
    edges = np.arange(51) * rate // 50
    rms = np.sqrt(np.add.reduceat(centered ** 2, edges[:-1]) / np.diff(edges))
    # Keep level ratios finite near silence using a fixed full-scale floor.
    peak = max(float(rms.max()), 1e-6)
    peak_over_median = 20 * np.log10(peak / max(float(np.median(rms)), 1e-6))
    relative_rise = np.maximum(np.diff(rms), 0).max() / peak
    # One-second FFT bins are 1 Hz wide. Compare levels against slow contact
    # vibration. Use a fixed normalized-power floor for silent windows and
    # neutral contrast for bands unavailable at this sample rate.
    energy = np.abs(np.fft.rfft(centered * np.hanning(rate)) / rate) ** 2
    floor = 1e-16
    low = max(energy[5:80].sum(), floor)
    contrasts = [10 * np.log10(max(energy[a:b].sum(), floor) / low) if rate >= 2 * b else 0.
                 for a, b in ((300, 1200), (1200, 5000))]
    mean_gyro = angular_path / dt.sum()
    return np.array([
        peak_over_median, relative_rise, np.mean(rms >= .5 * peak),
        20 * np.log10(peak / (1 + mean_gyro)), np.log1p(mean_gyro),
        direction_consistency, np.log1p(net_accel_change), motion_features[0], *contrasts,
    ])


class MotionCoughSpecialist:
    """Class-balanced pair model; all other activity scores are inherited."""

    def __init__(self, mean, scale, weights, training_sessions=(), spectral_iqr=None):
        self.mean, self.scale, self.weights = (
            np.array(x, dtype=float, copy=True) for x in (mean, scale, weights)
        )
        size = len(FEATURE_NAMES)
        if (self.mean.shape != (size,) or self.scale.shape != (size,)
                or self.weights.shape != (size + 1, len(CLASSES))
                or (self.scale <= 0).any()
                or not all(np.isfinite(x).all() for x in (self.mean, self.scale, self.weights))):
            raise ValueError("invalid motion/cough weights")
        if (not isinstance(training_sessions, (list, tuple, set))
                or not all(isinstance(s, str) and s for s in training_sessions)):
            raise ValueError("expected session identifiers")
        self.training_sessions = tuple(sorted(set(training_sessions)))

        self.spectral_iqr = None
        if spectral_iqr is not None:
            bands = np.array(spectral_iqr, dtype=float, copy=True)
            # In CLASSES order; rows are Q1 and Q3, columns are audio bands.
            if (bands.shape != (2, 2, 2) or not np.isfinite(bands).all()
                    or (bands[:, 0] > bands[:, 1]).any()):
                raise ValueError("invalid spectral quartiles")
            self.spectral_iqr = bands

    def _both_bands_favor_head(self, contrast):
        if self.spectral_iqr is None:
            return False
        cough, head = self.spectral_iqr
        head_typical = (contrast >= head[0]) & (contrast <= head[1])
        cough_atypical = (contrast < cough[0]) | (contrast > cough[1])
        return bool(np.all(head_typical & cough_atypical))

    @classmethod
    def fit(cls, X, labels, sessions=()):
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(labels)
        if labels.ndim != 1 or matrix.shape != (labels.size, len(FEATURE_NAMES)):
            raise ValueError("expected one motion/cough feature vector per label")
        mask = np.isin(labels, CLASSES)
        if set(labels[mask]) != set(CLASSES) or not np.isfinite(matrix[mask]).all():
            raise ValueError("need usable examples of both head movement and coughing")
        _, mean, scale, weights = _fit_softmax(matrix[mask], labels[mask])
        # The last two columns hold the two acoustic contrasts. Zero signals an
        # unresolved/neutral band. Require at least four training windows per class to
        # estimate quartiles before permitting an override of a confident cough.
        contrast_by_class = []
        for name in CLASSES:
            contrasts = matrix[labels == name, -2:]
            contrast_by_class.append(contrasts[(contrasts != 0.).all(axis=1)])
        spectral_iqr = None
        if all(len(rows) >= 4 for rows in contrast_by_class):
            quartiles = np.array([np.percentile(rows, [25, 75], axis=0) for rows in contrast_by_class])
            if (quartiles[:, 1] > quartiles[:, 0]).all():
                spectral_iqr = quartiles
        return cls(mean, scale, weights, sessions, spectral_iqr)

    def refine(self, primary_scores, features):
        scores = dict(primary_scores)
        if features is None or not all(name in scores for name in CLASSES):
            return scores
        primary = np.array([scores[k] for k in CLASSES])
        strongest_other = max((v for k, v in scores.items() if k not in CLASSES), default=-1.)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        decisive = _decisive(scores[best], scores[second])
        # A clear cough may be corrected only with a separate spectral check.
        # Protect every other clear outcome, and mixed doubts with another class.
        if decisive:
            if best != "coughing" or self.spectral_iqr is None:
                return scores
        elif primary.min() <= strongest_other:
            return scores
        row = np.asarray(features, dtype=float)
        if row.shape != self.mean.shape or not np.isfinite(row).all():
            raise ValueError("expected one finite motion/cough feature vector")
        standardized = np.clip((row - self.mean) / self.scale, -6, 6)
        pair = _softmax(standardized @ self.weights[:-1] + self.weights[-1])
        if pair.max() < .65:
            return scores
        refined = .25 * primary + .75 * primary.sum() * pair
        if decisive:
            if ((row[-2:] == 0.).any() or not self._both_bands_favor_head(row[-2:])
                    or pair[1] <= pair[0]
                    or not _decisive(refined[1], max(refined[0], strongest_other))):
                return scores
        if refined.max() > strongest_other:
            scores.update(zip(CLASSES, refined.tolist()))
        return scores

    def predict(self, primary_scores, features):
        scores = self.refine(primary_scores, features)
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        if _decisive(scores[best], scores[second]):
            return best, scores[best]
        return UNCERTAIN, scores[best]

    def save(self, path):
        path = Path(path)
        data = {
            "type": "motion_cough_v2", "classes": list(CLASSES), "features": list(FEATURE_NAMES),
            "window_seconds": WINDOW_SECONDS, "mean": self.mean.tolist(),
            "scale": self.scale.tolist(), "weights": self.weights.tolist(),
            "training_sessions": list(self.training_sessions),
            "spectral_iqr": self.spectral_iqr.tolist() if self.spectral_iqr is not None else None,
        }
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            tmp = Path(directory) / "model.json"
            tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("type") not in ("motion_cough_v1", "motion_cough_v2")
                or data.get("classes") != list(CLASSES) or data.get("window_seconds") != WINDOW_SECONDS):
            raise ValueError("incompatible motion/cough model")
        legacy = data["type"] == "motion_cough_v1"
        names = FEATURE_NAMES[:-2] if legacy else FEATURE_NAMES
        if data.get("features") != list(names):
            raise ValueError("incompatible motion/cough features")
        try:
            mean, scale, weights = data["mean"], data["scale"], data["weights"]
            if legacy:
                # Old artifacts have no coefficients for the two added bands.
                mean, scale = [*mean, 0., 0.], [*scale, 1., 1.]
                weights = [*weights[:-1], [0., 0.], [0., 0.], weights[-1]]
            return cls(mean, scale, weights, data["training_sessions"], data.get("spectral_iqr"))
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("invalid motion/cough model") from error
