"""Experimental classification of one observed activity from contact audio and IMU."""

import json
import tempfile
from pathlib import Path

import numpy as np

from audio_model import FEATURE_NAMES as AUDIO_FEATURE_NAMES
from audio_model import WINDOW_SECONDS, extract_features


FEATURE_NAMES = AUDIO_FEATURE_NAMES + (
    "log_mean_accel_change_per_second", "log_peak_accel_change_per_second",
    "log_mean_gyro_magnitude", "log_peak_gyro_magnitude",
)
MAX_MOTION_GAP_SECONDS = 0.5
MIN_SCORE = 0.5
MIN_MARGIN = 0.12
UNCERTAIN = "uncertain"
VOICE_CLASSES = ("coughing", "speaking")
VOICE_FEATURE_NAMES = (
    "mean_periodicity", "voiced_fraction", "envelope_peak_to_mean",
    "onset_peak_to_mean", "mean_spectral_flatness",
)


class InsufficientMotion(ValueError):
    """No sufficiently continuous motion signal in the causal audio window."""


def extract_motion_features(motion_rows, end_time):
    """Summarize [end-1, end] rows: time in seconds, XYZ acceleration, XYZ gyro."""
    if not np.isfinite(end_time):
        raise ValueError("motion window must have a finite end time")
    rows = np.asarray(motion_rows, dtype=float)
    if not rows.size:
        raise InsufficientMotion("no motion samples in the window")
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.all(np.isfinite(rows[:, 0])):
        raise ValueError("expected seven columns with finite motion timestamps")
    start = end_time - WINDOW_SECONDS
    rows = rows[(rows[:, 0] >= start) & (rows[:, 0] <= end_time)]
    if len(rows) < 2:
        raise InsufficientMotion("need two or more motion samples in the window")
    if not np.all(np.isfinite(rows)):
        raise InsufficientMotion("invalid motion sample in the window")
    times = rows[:, 0]
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise InsufficientMotion("motion samples must have strictly increasing times")
    if max(times[0] - start, end_time - times[-1], dt.max()) > MAX_MOTION_GAP_SECONDS:
        raise InsufficientMotion("incomplete motion coverage in the window")

    # The Arduino's sampling interval varies. Use elapsed time in both rates
    # and averages, and vector magnitudes to reduce dependence on orientation.
    accel_speed = np.linalg.norm(np.diff(rows[:, 1:4], axis=0), axis=1) / dt
    gyro = np.linalg.norm(rows[:, 4:7], axis=1)
    mean_gyro = np.average((gyro[:-1] + gyro[1:]) * 0.5, weights=dt)
    return np.log1p([np.average(accel_speed, weights=dt), accel_speed.max(),
                     mean_gyro, gyro.max()])


def extract_activity_features(audio, sample_rate, motion_rows, end_time):
    motion = extract_motion_features(motion_rows, end_time)
    return np.concatenate((extract_features(audio, sample_rate), motion))


def _softmax(logits):
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def _decisive(best_score, runner_up_score):
    return best_score >= MIN_SCORE and best_score - runner_up_score >= MIN_MARGIN


def extract_voice_features(audio, sample_rate):
    """Measure pitch-period repetition and impulsiveness in the same causal second."""
    signal = np.asarray(audio, dtype=float)
    rate = int(sample_rate)
    if rate != sample_rate or rate < 25 or signal.shape != (rate,) or not np.isfinite(signal).all():
        raise ValueError("voice features require one second of finite mono audio")
    if rate < 1000:
        # At low rates the pitch band is unresolved. Keep all five features finite.
        return np.zeros(len(VOICE_FEATURE_NAMES))
    # Low-pass before resampling. A one-second FFT has integer-Hz bins and
    # makes the same duration and frequency range available at 44.1 or 48 kHz.
    work_rate = min(4000, rate)
    spectrum = np.fft.rfft(signal)[:work_rate // 2 + 1].copy()
    spectrum[:65] = 0  # Remove DC, slow motion and mains below the voice band.
    spectrum[-1] = 0
    band = np.fft.irfft(spectrum, n=work_rate) * (work_rate / rate)
    size, hop = round(.05 * work_rate), round(.025 * work_rate)
    frames = np.lib.stride_tricks.sliding_window_view(band, size)[::hop]
    power = np.mean(frames ** 2, axis=1)
    if power.max() < 1e-12:
        return np.zeros(len(VOICE_FEATURE_NAMES))
    rms = np.sqrt(power)
    weights = rms / rms.sum()

    # Normalize lagged correlations by the energy in the overlapping part
    # of each frame, looking for repeat periods between about 2.5 and 14 ms.
    lags = np.arange(round(work_rate / 400), round(work_rate / 70) + 1)
    padded = np.fft.rfft(frames, n=2 * size, axis=1)
    acf = np.fft.irfft(np.abs(padded) ** 2, n=2 * size, axis=1)[:, lags]
    energy = np.column_stack((np.zeros(len(frames)), np.cumsum(frames ** 2, axis=1)))
    denom = np.sqrt(energy[:, size - lags] * np.maximum(0, energy[:, -1:] - energy[:, lags]))
    periodicity = np.clip((acf / np.maximum(denom, 1e-15)).max(axis=1), 0, 1)
    power_spectrum = np.abs(np.fft.rfft(frames * np.hanning(size), axis=1)) ** 2
    power_spectrum = power_spectrum[:, 4:]  # 80 Hz and higher at 50 ms/frame.
    mean_power = np.maximum(power_spectrum.mean(axis=1), 1e-20)
    relative_power = power_spectrum / mean_power[:, None]
    flatness = np.exp(np.mean(np.log(np.maximum(relative_power, 1e-12)), axis=1))
    return np.array([weights @ periodicity, weights @ (periodicity > .65),
                     rms.max() / rms.mean(), max(0., np.diff(rms).max()) / rms.mean(),
                     weights @ flatness])


def _fit_softmax(X, labels):
    classes, index, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if len(classes) < 2:
        raise ValueError("training requires at least two observed activity classes")
    # Each activity contributes equal total weight irrespective of capture count.
    sample_weights = 1.0 / (len(classes) * counts[index])
    mean = np.average(X, axis=0, weights=sample_weights)
    scale = np.sqrt(np.average((X - mean) ** 2, axis=0, weights=sample_weights))
    scale[scale < 1e-8] = 1
    design = np.column_stack(((X - mean) / scale, np.ones(len(X))))
    targets = np.eye(len(classes))[index]
    weights = np.zeros((design.shape[1], len(classes)))
    penalty = 0.05
    covariance = (design.T * sample_weights) @ design
    # A fixed Lipschitz step for softmax cross entropy with ridge penalty.
    step = 1 / (0.5 * np.linalg.eigvalsh(covariance)[-1] + penalty)
    for _ in range(1000):
        errors = _softmax(design @ weights) - targets
        gradient = design.T @ (sample_weights[:, None] * errors)
        gradient[:-1] += penalty * weights[:-1]
        weights -= step * gradient
        if np.max(np.abs(gradient)) < 1e-7:
            break
    return classes.tolist(), mean, scale, weights


class ActivityModel:
    """Class-balanced regularized softmax. Scores are relative and uncalibrated."""

    def __init__(self, classes, mean, scale, weights, training_sessions=(),
                 voice_parameters=None, voice_training_sessions=()):
        if (not isinstance(classes, (list, tuple)) or
                not all(isinstance(c, str) and c and c != UNCERTAIN for c in classes) or
                len(set(classes)) != len(classes) or len(classes) < 2):
            raise ValueError("model requires two or more distinct named activities")
        self.classes = tuple(classes)
        self.mean, self.scale, self.weights = (
            np.array(values, dtype=float, copy=True) for values in (mean, scale, weights)
        )
        size = len(FEATURE_NAMES)
        if (self.mean.shape != (size,) or self.scale.shape != (size,)
                or self.weights.shape != (size + 1, len(self.classes))
                or not all(np.isfinite(v).all() for v in (self.mean, self.scale, self.weights))
                or (self.scale <= 0).any()):
            raise ValueError("invalid activity model weights or scaling")
        for group in (training_sessions, voice_training_sessions):
            if (not isinstance(group, (list, tuple, set))
                    or not all(isinstance(s, str) and s for s in group)):
                raise ValueError("training sessions must contain session identifiers")
        self.primary_training_sessions = tuple(sorted(set(training_sessions)))
        self.voice_training_sessions = tuple(sorted(set(voice_training_sessions)))
        self.training_sessions = tuple(sorted(set(training_sessions) | set(voice_training_sessions)))
        self.voice_parameters = None
        if voice_parameters is not None:
            vmean, vscale, vweights = (np.array(v, dtype=float, copy=True) for v in voice_parameters)
            size = len(VOICE_FEATURE_NAMES)
            if (vmean.shape != (size,) or vscale.shape != (size,) or vweights.shape != (size + 1, 2)
                    or not all(np.isfinite(v).all() for v in (vmean, vscale, vweights))
                    or (vscale <= 0).any() or not set(VOICE_CLASSES).issubset(self.classes)):
                raise ValueError("invalid voice specialist parameters")
            self.voice_parameters = (vmean, vscale, vweights)

    @property
    def has_voice_specialist(self):
        return self.voice_parameters is not None

    @classmethod
    def fit(cls, X, y, sessions=None, voice_features=None):
        X = np.asarray(X, dtype=float)
        labels = list(y)
        if (X.shape != (len(labels), len(FEATURE_NAMES)) or not np.isfinite(X).all()
                or not all(isinstance(c, str) and c and c != UNCERTAIN for c in labels)):
            raise ValueError("expected a row of finite features per labeled activity")
        classes, mean, scale, weights = _fit_softmax(X, labels)
        model = cls(classes, mean, scale, weights, () if sessions is None else sessions)
        if voice_features is not None and set(VOICE_CLASSES).issubset(classes):
            return model.with_voice_specialist(voice_features, labels, sessions=sessions)
        return model

    def with_voice_specialist(self, voice_X, labels, sessions=None):
        """Keep the primary weights frozen and fit only the conditional voice model."""
        features = np.asarray(voice_X, dtype=float)
        labels = np.asarray(labels)
        if (features.shape != (len(labels), len(VOICE_FEATURE_NAMES)) or labels.ndim != 1
                or not np.isfinite(features).all()):
            raise ValueError("voice training requires a finite feature vector per label")
        mask = np.isin(labels, VOICE_CLASSES)
        if not (set(VOICE_CLASSES).issubset(self.classes) and set(labels[mask]) == set(VOICE_CLASSES)):
            raise ValueError("specialist requires both speaking and coughing")
        _, mean, scale, weights = _fit_softmax(features[mask], labels[mask])
        return ActivityModel(self.classes, self.mean, self.scale, self.weights,
                             self.primary_training_sessions, (mean, scale, weights),
                             () if sessions is None else sessions)

    def scores(self, features, voice_features=None):
        row = np.asarray(features, dtype=float)
        if row.shape != self.mean.shape or not np.isfinite(row).all():
            raise ValueError("scoring requires one row of finite activity features")
        logits = ((row - self.mean) / self.scale) @ self.weights[:-1] + self.weights[-1]
        scores = dict(zip(self.classes, _softmax(logits).tolist()))
        if self.voice_parameters is None or voice_features is None:
            return scores
        best, second = sorted(scores, key=scores.get, reverse=True)[:2]
        strongest_other = max((v for k, v in scores.items() if k not in VOICE_CLASSES), default=-1.)
        if best not in VOICE_CLASSES:
            return scores
        # A voice-only model cannot resolve an uncertain comparison with a
        # swallow, motion or rest. Include ties for second in that protection.
        if not _decisive(scores[best], scores[second]) and min(scores[k] for k in VOICE_CLASSES) <= strongest_other:
            return scores
        vmean, vscale, vweights = self.voice_parameters
        voice = np.asarray(voice_features, dtype=float)
        if voice.shape != vmean.shape or not np.isfinite(voice).all():
            raise ValueError("scoring requires one finite voice feature vector")
        if not np.any(voice):
            return scores
        z = np.clip((voice - vmean) / vscale, -6, 6)
        pair = _softmax(z @ vweights[:-1] + vweights[-1])
        if pair.max() < .65:
            return scores
        # Reallocate only the speaking/coughing mass. The frozen primary still
        # contributes to the decision if its preferred member strongly disagrees.
        primary = np.array([scores[k] for k in VOICE_CLASSES])
        refined = primary * .25 + .75 * primary.sum() * pair
        # Outside activities can never be promoted by the pair specialist.
        if refined.max() > strongest_other:
            scores.update(zip(VOICE_CLASSES, refined.tolist()))
        return scores

    def predict(self, features, voice_features=None):
        ranked = sorted(self.scores(features, voice_features).items(), key=lambda pair: pair[1], reverse=True)
        best, score = ranked[0]
        if not _decisive(score, ranked[1][1]):
            return UNCERTAIN, score
        return best, score

    def save(self, path):
        path = Path(path)
        payload = {
            "type": "activity_softmax_v1", "classes": list(self.classes),
            "features": list(FEATURE_NAMES), "window_seconds": WINDOW_SECONDS,
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": self.weights.tolist(), "training_sessions": list(self.training_sessions),
        }
        if self.has_voice_specialist:
            mean, scale, weights = self.voice_parameters
            payload.update(type="activity_softmax_voice_v2",
                           primary_training_sessions=list(self.primary_training_sessions),
                           voice={"features": list(VOICE_FEATURE_NAMES), "classes": list(VOICE_CLASSES),
                                  "mean": mean.tolist(), "scale": scale.tolist(), "weights": weights.tolist(),
                                  "training_sessions": list(self.voice_training_sessions)})
        with tempfile.TemporaryDirectory(dir=path.parent) as directory:
            temporary = Path(directory) / "activity.json"
            temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if (not isinstance(data, dict) or data.get("type") not in ("activity_softmax_v1", "activity_softmax_voice_v2")
                or data.get("features") != list(FEATURE_NAMES)
                or data.get("window_seconds") != WINDOW_SECONDS):
            raise ValueError("incompatible activity model")
        try:
            if data["type"] == "activity_softmax_v1":
                return cls(data["classes"], data["mean"], data["scale"], data["weights"],
                           data["training_sessions"])
            voice = data["voice"]
            if voice["features"] != list(VOICE_FEATURE_NAMES) or voice["classes"] != list(VOICE_CLASSES):
                raise ValueError("incompatible voice specialist")
            model = cls(data["classes"], data["mean"], data["scale"], data["weights"],
                        data["primary_training_sessions"], (voice["mean"], voice["scale"], voice["weights"]),
                        voice["training_sessions"])
            if list(model.training_sessions) != data["training_sessions"]:
                raise ValueError("inconsistent training session identifiers")
            return model
        except (KeyError, TypeError) as exc:
            raise ValueError("invalid activity model fields") from exc
