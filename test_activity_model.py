import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from activity_model import (
    ActivityModel, FEATURE_NAMES, InsufficientMotion, UNCERTAIN,
    VOICE_FEATURE_NAMES, extract_activity_features, extract_motion_features, extract_voice_features,
)


def movement(times):
    rows = np.zeros((len(times), 7))
    rows[:, 0] = times
    rows[:, 1] = 2 * np.asarray(times)
    rows[:, 3] = 1.0
    rows[:, 5] = times
    return rows


class ActivityModelTests(unittest.TestCase):
    def test_motion_is_time_weighted_and_future_data_is_not_used(self):
        dense = movement(np.linspace(0, 1, 26))
        sparse = movement([0, .31, .79, 1])
        sparse = np.vstack((sparse, [1.01, 300, 0, 0, 500, 0, 0]))
        reference = np.log1p([2.0, 2.0, .5, 1.0])
        np.testing.assert_allclose(extract_motion_features(dense, 1), reference)
        np.testing.assert_allclose(extract_motion_features(sparse, 1), reference)
        vector = extract_activity_features(np.zeros(1000), 1000, sparse, 1)
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        np.testing.assert_allclose(vector[-4:], reference)

    def test_motion_gap_stale_and_duplicate_samples_are_rejected(self):
        for times in ([], [.9], [-2, -1], [.1, .8], [.3, .3], [.6, .8], [.13, .35]):
            with self.subTest(times=times), self.assertRaises(InsufficientMotion):
                extract_motion_features(movement(times), 1)

    def test_multiclass_round_trip_and_abstention(self):
        features = np.eye(len(FEATURE_NAMES))[:3]
        labels = ["saliva", "speaking", "rest"]
        # Extra instances of rest should not bias this deliberately symmetric fit.
        X = np.vstack((features, [features[2]] * 4))
        model = ActivityModel.fit(X, labels + ["rest"] * 4, sessions=["a", "b"])
        for label, row in zip(labels, features):
            self.assertEqual(model.predict(row)[0], label)
        self.assertEqual(model.predict(features.mean(axis=0))[0], UNCERTAIN)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            loaded = ActivityModel.load(path)
            self.assertEqual(loaded.training_sessions, ("a", "b"))
            self.assertEqual(loaded.scores(features[0]), model.scores(features[0]))
            self.assertAlmostEqual(sum(loaded.scores(features[1]).values()), 1.)
            damaged = json.loads(path.read_text())
            damaged["features"].reverse()
            path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                ActivityModel.load(path)

    def test_voice_periodicity_and_transients_remain_comparable_across_rates(self):
        rng = np.random.default_rng(14)
        for rate in (1000, 8000, 44100, 48000):
            time = np.arange(rate) / rate
            tone = .05 * np.sin(2 * np.pi * 160 * time)
            burst = .05 * rng.normal(size=rate) * ((time > .51) & (time < .58))
            periodic = extract_voice_features(tone, rate)
            impulsive = extract_voice_features(burst, rate)
            self.assertGreater(periodic[0], impulsive[0])
            self.assertLess(periodic[2], impulsive[2])
            np.testing.assert_allclose(periodic, extract_voice_features(tone * 1.5, rate), atol=1e-7)
        np.testing.assert_array_equal(extract_voice_features(np.zeros(1000), 1000), 0)

    def test_frozen_voice_specialist_only_redistributes_eligible_pair(self):
        size = len(FEATURE_NAMES)
        weights = np.zeros((size + 1, 3))
        weights[-1] = np.log([.40, .55, .05])
        weights[0, 2] = 10  # A separate observation originally classified as rest.
        primary = ActivityModel(["speaking", "coughing", "rest"], np.zeros(size),
                                np.ones(size), weights, ["base"])
        voice = np.ones(len(VOICE_FEATURE_NAMES))
        specialist = primary.with_voice_specialist([voice, -voice], ["speaking", "coughing"], ["added"])
        window = np.zeros(size)
        self.assertFalse(primary.has_voice_specialist)
        self.assertEqual(primary.predict(window)[0], "coughing")
        self.assertEqual(specialist.predict(window, voice)[0], "speaking")
        primary_scores = primary.scores(window)
        result = specialist.scores(window, voice)
        self.assertEqual(result["rest"], primary_scores["rest"])
        self.assertAlmostEqual(sum(result.values()), 1.)
        np.testing.assert_array_equal(specialist.weights, primary.weights)
        self.assertEqual(specialist.scores(window, voice * 0), primary_scores)
        rest = window.copy()
        rest[0] = 1
        self.assertEqual(primary.predict(rest)[0], "rest")
        self.assertEqual(specialist.scores(rest, voice), primary.scores(rest))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.json"
            specialist.save(path)
            loaded = ActivityModel.load(path)
            self.assertEqual(loaded.training_sessions, ("added", "base"))
            self.assertEqual(loaded.voice_training_sessions, ("added",))
            self.assertEqual(loaded.primary_training_sessions, ("base",))
            self.assertEqual(loaded.scores(window, voice), result)

    def test_uncertain_voice_refinement_requires_both_primary_choices_to_be_voice(self):
        size = len(FEATURE_NAMES)
        voice = np.ones(len(VOICE_FEATURE_NAMES))
        classes = ["coughing", "speaking", "swallowing"]
        window = np.zeros(size)
        for probabilities, expected in (([.46, .14, .40], UNCERTAIN),
                                        ([.46, .27, .27], UNCERTAIN),
                                        ([.46, .40, .14], "coughing")):
            with self.subTest(probabilities=probabilities):
                weights = np.zeros((size + 1, 3))
                weights[-1] = np.log(probabilities)
                base = ActivityModel(classes, np.zeros(size), np.ones(size), weights)
                model = base.with_voice_specialist([voice, -voice], ["coughing", "speaking"])
                self.assertEqual(base.predict(window)[0], UNCERTAIN)
                self.assertEqual(model.predict(window, voice)[0], expected)
                if expected == UNCERTAIN:
                    self.assertEqual(model.scores(window, voice), base.scores(window))

    def test_abstains_when_top_two_scores_are_close_even_above_half(self):
        size = len(FEATURE_NAMES)
        weights = np.zeros((size + 1, 3))
        weights[-1] = np.log([.51, .46, .03])
        model = ActivityModel(["rest", "speaking", "coughing"], np.zeros(size), np.ones(size), weights)
        label, score = model.predict(np.zeros(size))
        self.assertEqual(label, UNCERTAIN)
        self.assertAlmostEqual(score, .51)


if __name__ == "__main__":
    unittest.main()
