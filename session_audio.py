"""Record timestamped mono audio for replay of complete experimental trials."""

import csv
import json
import math
import queue
import threading
import wave
from contextlib import ExitStack
from pathlib import Path

import numpy as np


WAV_NAME = "session_audio.wav"
INDEX_NAME = "session_audio_blocks.csv"
METADATA_NAME = "session_audio.json"
TIMESTAMP_TOLERANCE = 0.002  # ADC callbacks can jitter slightly even on an uninterrupted stream.


class SessionAudioRecorder:
    """Call add_audio during capture, then close after stopping the input stream."""

    def __init__(self, session, sample_rate):
        self.session = Path(session)
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0 or self.sample_rate != sample_rate:
            raise ValueError("La frecuencia de muestreo debe ser un entero positivo.")
        self.session.mkdir(parents=True, exist_ok=True)
        if (self.session / METADATA_NAME).exists():
            raise FileExistsError(self.session / METADATA_NAME)

        self._blocks = queue.Queue(maxsize=128)
        self._state_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._closed = False
        self._published = False
        self._error = None
        self._dropped_frames = 0
        self._overflow_count = 0
        self._frames = 0
        self._block_count = 0
        self._start = self._end = None
        self._gaps = self._overlaps = 0

        with ExitStack() as files:
            raw = files.enter_context((self.session / WAV_NAME).open("xb"))
            index = files.enter_context((self.session / INDEX_NAME).open("x", newline="", encoding="utf-8"))
            wav = files.enter_context(wave.open(raw, "wb"))
            wav.setparams((1, 4, self.sample_rate, 0, "NONE", "not compressed"))
            rows = csv.writer(index)
            rows.writerow(["start_monotonic", "wav_frame", "frame_count"])
            self._worker = threading.Thread(
                target=self._write, args=(files.pop_all(), wav, rows), name="session audio", daemon=True
            )
            self._worker.start()

    def add_audio(self, chunk, block_start, overflow=False):
        """Queue a copy of one block. Never wait for disk writes or queue space."""
        start = float(block_start)
        samples = np.array(chunk, dtype=np.float32, copy=True)
        if samples.ndim != 1 or not math.isfinite(start):
            raise ValueError("Se requiere audio mono con tiempo de adquisición finito.")
        if samples.size == 0:
            return
        with self._state_lock:
            if self._closed:
                raise RuntimeError("La grabación de audio ya terminó.")
            self._overflow_count += int(bool(overflow))
            try:
                self._blocks.put_nowait((start, samples))
            except queue.Full:
                self._dropped_frames += len(samples)

    def _write(self, files, wav, rows):
        try:
            with files:
                while True:
                    block = self._blocks.get()
                    if block is None:
                        return
                    start, samples = block
                    if not np.all(np.isfinite(samples)):
                        raise ValueError("El audio contiene muestras no finitas.")
                    if self._end is None:
                        self._start = start
                    else:
                        delta = start - self._end
                        self._gaps += int(delta > TIMESTAMP_TOLERANCE)
                        self._overlaps += int(delta < -TIMESTAMP_TOLERANCE)
                    pcm = np.rint(samples.astype(np.float64) * (2 ** 31))
                    wav.writeframesraw(np.clip(pcm, -(2 ** 31), 2 ** 31 - 1).astype("<i4").tobytes())
                    rows.writerow([repr(start), self._frames, len(samples)])
                    self._block_count += 1
                    self._frames += len(samples)
                    self._end = start + len(samples) / self.sample_rate
        except Exception as error:
            self._error = error

    def close(self, active_from=None):
        """Finish writing, propagate disk errors and publish metadata last."""
        with self._close_lock:
            with self._state_lock:
                already_closed = self._closed
                self._closed = True
            if not already_closed:
                # If disk failed with a full queue, the writer may have already exited.
                while self._worker.is_alive():
                    try:
                        self._blocks.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                self._worker.join()
            if self._error is not None:
                raise RuntimeError("No se pudo guardar el audio continuo.") from self._error
            if self._published:
                return
            metadata = {
                "format_version": 1,
                "sample_rate_hz": self.sample_rate,
                "block_count": self._block_count,
                "wav_frames": self._frames,
                "start_monotonic": self._start,
                "end_monotonic": self._end,
                "active_from_monotonic": active_from,
                "gap_count": self._gaps,
                "overlap_count": self._overlaps,
                "dropped_frames": self._dropped_frames,
                "overflow_count": self._overflow_count,
            }
            tmp = self.session / (METADATA_NAME + ".tmp")
            tmp.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            tmp.replace(self.session / METADATA_NAME)
            self._published = True


def iter_segments(session):
    """Yield original (monotonic start, float32 samples, Hz) blocks without filling gaps."""
    session = Path(session)
    metadata = json.loads((session / METADATA_NAME).read_text(encoding="utf-8"))
    if metadata.get("format_version") != 1:
        raise ValueError("Versión desconocida de la grabación continua.")
    if metadata["dropped_frames"] or metadata["overflow_count"]:
        raise ValueError("Se perdieron muestras en la grabación continua. Repite la sesión de evaluación.")
    with wave.open(str(session / WAV_NAME), "rb") as wav, (session / INDEX_NAME).open(encoding="utf-8", newline="") as index:
        rate = wav.getframerate()
        if (wav.getnchannels(), wav.getsampwidth(), rate) != (1, 4, metadata["sample_rate_hz"]):
            raise ValueError("El formato del WAV difiere de los metadatos.")
        if wav.getnframes() != metadata["wav_frames"]:
            raise ValueError("El tamaño del WAV difiere de los metadatos.")
        position = count = 0
        last_start = None
        rows = csv.DictReader(index)
        if rows.fieldnames != ["start_monotonic", "wav_frame", "frame_count"]:
            raise ValueError("El encabezado del índice de audio es inválido.")
        for row in rows:
            start = float(row["start_monotonic"])
            frames = int(row["frame_count"])
            if (frames <= 0 or int(row["wav_frame"]) != position or not math.isfinite(start)
                    or (last_start is not None and start <= last_start)):
                raise ValueError("El índice temporal del audio contiene un bloque inválido.")
            if frames > wav.getnframes() - position:
                raise ValueError("El índice contiene muestras fuera del WAV.")
            data = wav.readframes(frames)
            if len(data) != frames * 4:
                raise ValueError("La grabación de audio está truncada.")
            position += frames
            count += 1
            last_start = start
            samples = np.frombuffer(data, dtype="<i4").astype(np.float32) * np.float32(2 ** -31)
            yield start, samples, rate
        if position != wav.getnframes() or count != metadata["block_count"]:
            raise ValueError("El índice del audio está incompleto.")
