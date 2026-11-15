import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from activity_model import InsufficientMotion
from motion_cough import FEATURE_NAMES, MotionCoughSpecialist, extract_motion_cough_features


class MotionCoughTests(unittest.TestCase):
    def test_features_resolve_turn_reversal_and_transient_with_sparse_motion(self):
        time = np.arange(8000) / 8000
        tone = .03 * np.sin(2 * np.pi * 200 * time)
        burst = tone * ((time > .31) & (time < .40))
        rows = np.zeros((3, 7))
        rows[:, 0] = [-.9, -.5, -.1]
        rows[:, 4] = 10
        steady = extract_motion_cough_features(tone, 8000, rows, 0)
        self.assertAlmostEqual(steady[5], 1.)
        rows[1, 4] = -10
        reversal = extract_motion_cough_features(burst, 8000, rows, 0)
        self.assertAlmostEqual(reversal[5], 0.)
        self.assertGreater(reversal[0], steady[0])
        self.assertLess(reversal[2], steady[2])
        # Future samples must never influence the live/replayed decision.
        extra = np.vstack([rows, [.01, 100, 100, 100, 100, 100, 100]])
        np.testing.assert_array_equal(extract_motion_cough_features(burst, 8000, extra, 0), reversal)
        with self.assertRaises(InsufficientMotion):
            extract_motion_cough_features(tone, 8000, rows[:1], 0)
        # Two valid observations in the window are sufficient at historical rates.
        self.assertTrue(np.isfinite(extract_motion_cough_features(tone, 8000, rows[:2], 0)).all())

    def test_spectral_contrast_separates_bands_independently_of_gain(self):
        rate = 16000
        time = np.arange(rate) / rate
        rows = np.zeros((3, 7))
        rows[:, 0] = [.1, .5, .9]
        low = np.sin(2 * np.pi * 50 * time)
        middle = np.sin(2 * np.pi * 600 * time)
        high = np.sin(2 * np.pi * 2000 * time)
        self.assertTrue((extract_motion_cough_features(low, rate, rows, 1)[-2:] < 0).all())
        self.assertGreater(extract_motion_cough_features(middle, rate, rows, 1)[-2], 0)
        self.assertGreater(extract_motion_cough_features(high, rate, rows, 1)[-1], 0)
        for gain in (.004, .09):
            mixed = gain * (low + middle + high)
            np.testing.assert_allclose(extract_motion_cough_features(mixed,rate,rows,1)[-2:], [0,0], atol=1e-5)

    def test_direction_score_is_rotation_independent(self):
        rows = np.array([[0., 1, 0, 1, 5, 0, -2], [.4, 1, .1, .9, -2, 0, 4],
                         [.9, 1, .2, .8, 0, 1, 7]])
        rotate = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        aligned = rows.copy()
        aligned[:, 1:4] = rows[:, 1:4] @ rotate
        aligned[:, 4:7] = rows[:, 4:7] @ rotate
        np.testing.assert_allclose(extract_motion_cough_features(np.zeros(1000), 1000, rows, 1),
                                   extract_motion_cough_features(np.zeros(1000), 1000, aligned, 1))

    def test_predictions_only_refine_pair_and_preserve_nonpair_uncertainty(self):
        cue = np.ones(len(FEATURE_NAMES))
        model = MotionCoughSpecialist.fit([cue, -cue, cue], ["head_movement", "coughing", "rest"], ["take"])
        protected = ({"coughing": .43, "head_movement": .17, "swallowing": .40},
                     {"coughing": .32, "head_movement": .19, "speaking": .49},
                     {"coughing": .40, "head_movement": .30, "swallowing": .30})
        for scores in protected:
            self.assertEqual(model.refine(scores, cue), scores)
        for favored, other, opposing in (("coughing", "head_movement", cue),
                                          ("head_movement", "coughing", -cue)):
            decided = {favored: .83, other: .15, "rest": .02}
            self.assertEqual(model.refine(decided, opposing), decided)
        primary = {"coughing": .42, "head_movement": .38, "swallowing": .20}
        self.assertEqual(model.predict(primary, None), ("uncertain", .42))
        self.assertEqual(model.predict(primary, cue)[0], "head_movement")
        self.assertEqual(model.predict(primary, -cue)[0], "coughing")
        refined = model.refine(primary, cue)
        self.assertEqual(refined["swallowing"], primary["swallowing"])
        self.assertAlmostEqual(sum(refined.values()), 1.)

    def test_decided_cough_changes_only_when_both_bands_and_classifier_agree(self):
        labels = ["coughing"] * 6 + ["head_movement"] * 6
        values = np.arange(6) * .1
        data = np.ones((12, len(FEATURE_NAMES)))
        data[:6] *= -1
        data[:6, -2:] = (-4. + values)[:, None]
        data[6:, -2:] = (4. + values)[:, None]
        model = MotionCoughSpecialist.fit(data, labels, ["train"])
        marked_cough = {"coughing": .84, "head_movement": .13, "rest": .03}
        typical_head = np.median(data[6:], axis=0)
        self.assertEqual(model.predict(marked_cough, typical_head)[0], "head_movement")
        only_one_band = typical_head.copy()
        only_one_band[-1] = -3.75
        self.assertEqual(model.refine(marked_cough, only_one_band), marked_cough)
        unrelated = typical_head.copy()
        unrelated[:-2] = -2.
        self.assertEqual(model.refine(marked_cough, unrelated), marked_cough)
        clear_head = {"coughing": .12, "head_movement": .85, "rest": .03}
        self.assertEqual(model.refine(clear_head, data[0]), clear_head)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "consensus.json"
            model.save(path)
            loaded = MotionCoughSpecialist.load(path)
        np.testing.assert_array_equal(loaded.spectral_iqr, model.spectral_iqr)
        self.assertEqual(loaded.predict(marked_cough, typical_head), model.predict(marked_cough, typical_head))

    def test_legacy_model_loads_with_neutral_new_spectral_weight(self):
        size = len(FEATURE_NAMES) - 2
        legacy = {"type": "motion_cough_v1", "window_seconds": 1., "classes": ["coughing","head_movement"],
                  "features": list(FEATURE_NAMES[:-2]), "training_sessions": ["a"],
                  "mean": [0.] * size, "scale": [1.] * size,
                  "weights": [[0., 0.]]*size + [[1., -1.]]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps(legacy))
            loaded = MotionCoughSpecialist.load(path)
        old = {"coughing": .45, "head_movement": .45, "rest": .1}
        feature = np.ones(len(FEATURE_NAMES))
        self.assertEqual(loaded.predict(old, feature), loaded.predict(old, -feature))
        self.assertEqual(loaded.predict(old, feature)[0], "coughing")

    def test_save_load_and_malformed_models(self):
        cue = np.ones(len(FEATURE_NAMES))
        model = MotionCoughSpecialist.fit([cue, -cue], ["coughing", "head_movement"], ["session"])
        with self.assertRaises(ValueError):
            MotionCoughSpecialist.fit([cue, -cue], ["coughing", "rest"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sidecar.json"
            model.save(path)
            loaded = MotionCoughSpecialist.load(path)
            self.assertEqual(loaded.training_sessions, ("session",))
            scores = {"coughing": .48, "head_movement": .52}
            self.assertEqual(model.refine(scores, cue), loaded.refine(scores, cue))
            data = json.loads(path.read_text())
            data["classes"].reverse()
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                MotionCoughSpecialist.load(path)


if __name__ == "__main__":
    unittest.main()
