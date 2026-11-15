import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from audio_model import AudioModel, AudioStreamDetector, extract_features


class AudioModelTests(unittest.TestCase):
    def test_features_preserve_hz_across_sample_rates(self):
        vectors = []
        for rate in (8000, 11025, 44100, 48000):
            signal = .01 * np.sin(2 * np.pi * 250 * np.arange(rate) / rate)
            features = extract_features(signal, rate)
            self.assertGreater(features[5], .99)
            vectors.append(features)
        for vector in vectors[1:]:
            np.testing.assert_allclose(vector, vectors[0], atol=.1)

    def test_silence_is_finite_and_incomplete_audio_is_rejected(self):
        self.assertTrue(np.isfinite(extract_features(np.zeros(100), 100)).all())
        with self.assertRaises(ValueError):
            extract_features(np.zeros(99), 100)
        with self.assertRaises(ValueError):
            extract_features(np.full(100, np.nan), 100)

    def test_training_round_trip_and_invalid_schema(self):
        rate = 8000
        axis = np.arange(rate) / rate
        quiet = extract_features(np.zeros(rate), rate)
        active = extract_features(.1 * np.sin(2 * np.pi * 250 * axis), rate)
        model = AudioModel.fit([quiet, active, active], [0, 1, 1], ["session-A"])
        self.assertLess(model.score(quiet), .5)
        self.assertGreater(model.score(active), .5)
        with self.assertRaises(ValueError):
            AudioModel.fit([active, active], [1, 1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio.json"
            model.save(path)
            restored = AudioModel.load(path)
            self.assertEqual(restored.training_sessions, ("session-A",))
            self.assertEqual(restored.score(active), model.score(active))
            damaged = json.loads(path.read_text())
            damaged["features"].reverse()
            path.write_text(json.dumps(damaged))
            with self.assertRaises(ValueError):
                AudioModel.load(path)

    def test_stream_causal_times_and_onsets_independent_of_chunking(self):
        class PeakRule:
            def score(self, features):
                return float(features[1] > -30)

        rate = 1000
        audio = np.zeros(6000)
        for start in (1300, 3700):
            audio[start:start + 400] = .6 * np.sin(2 * np.pi * .08 * np.arange(400))
        offline = AudioStreamDetector(PeakRule(), rate)
        expected = offline.feed(audio, 0.)
        np.testing.assert_allclose(expected, [1.4, 3.8])
        streamed = AudioStreamDetector(PeakRule(), rate)
        actual = []
        for start in range(0, len(audio), 137):
            actual.extend(streamed.feed(audio[start:start + 137], 500. + start / rate))
        np.testing.assert_allclose(actual, np.asarray(expected) + 500.)

    def test_gap_discards_incomplete_window(self):
        class HighScore:
            def score(self, _):
                return .9

        detector = AudioStreamDetector(HighScore(), 1000)
        self.assertEqual(detector.feed(np.zeros(900), 7.0), [])
        self.assertEqual(detector.feed(np.zeros(990), 8.), [])
        self.assertIsNone(detector.last_score)
        self.assertEqual(detector.feed(np.zeros(10), 8.99), [9.])


if __name__ == "__main__":
    unittest.main()
