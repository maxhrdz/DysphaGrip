import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from session_audio import INDEX_NAME, METADATA_NAME, SessionAudioRecorder, iter_segments


class TestSessionAudio(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.session = Path(temporary.name)

    def test_preserves_original_blocks_and_times_without_inventing_audio(self):
        recorder = SessionAudioRecorder(self.session, 48000)
        block = np.array([0.0, -1., .123], dtype=np.float32)
        expected = block.copy()
        recorder.add_audio(block, 123.0)
        block[:] = 0.25
        recorder.add_audio(block, 123.0 + 3 / 48000)
        recorder.add_audio(block, 124.0)
        recorder.close(active_from=123.)
        recorder.close()
        segments = list(iter_segments(self.session))
        np.testing.assert_allclose(segments[0][1], expected, atol=1e-9)
        np.testing.assert_allclose(segments[1][1], block)
        self.assertEqual([x[0] for x in segments], [123., 123. + 3 / 48000, 124.])
        self.assertEqual([x[2] for x in segments], [48000] * 3)
        meta = json.loads((self.session / METADATA_NAME).read_text())
        self.assertEqual((meta["wav_frames"], meta["gap_count"], meta["dropped_frames"]), (9, 1, 0))
        self.assertEqual(meta["active_from_monotonic"], 123.)
        with self.assertRaises(FileExistsError):
            SessionAudioRecorder(self.session, 48000)

    def test_rejects_incomplete_index(self):
        recorder = SessionAudioRecorder(self.session, 48000)
        recorder.add_audio(np.zeros(32), 1.)
        recorder.close()
        index = self.session / INDEX_NAME
        index.write_text(index.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "índice"):
            list(iter_segments(self.session))

    def test_rejects_input_overflow(self):
        recorder = SessionAudioRecorder(self.session, 48000)
        recorder.add_audio(np.zeros(32), 1., overflow=True)
        recorder.close()
        with self.assertRaisesRegex(ValueError, "perdieron muestras"):
            list(iter_segments(self.session))

    def test_close_propagates_write_error_even_if_queue_has_filled(self):
        recorder = SessionAudioRecorder(self.session, 48000)
        recorder.add_audio([float("nan")], 1.)
        for i in range(256):
            recorder.add_audio([0.], 2. + i / 48000)
        with self.assertRaisesRegex(RuntimeError, "guardar"):
            recorder.close()
        self.assertFalse((self.session / METADATA_NAME).exists())

    def test_empty_session(self):
        recorder = SessionAudioRecorder(self.session, 48000)
        recorder.close()
        self.assertEqual(list(iter_segments(self.session)), [])


if __name__ == "__main__":
    unittest.main()
