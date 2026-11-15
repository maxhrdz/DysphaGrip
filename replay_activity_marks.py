"""Replay the causal windows archived at keypresses, with the displayed result as reference."""

import argparse
from collections import deque
import json
from pathlib import Path

import numpy as np

from activity_model import InsufficientMotion, extract_activity_features, extract_voice_features
from motion_cough import MotionCoughSpecialist, extract_motion_cough_features
from session_audio import TIMESTAMP_TOLERANCE, iter_segments
from train_audio_model import read_rows
from validation import POSITIVE_CATEGORIES


MOTION_FIELDS = ("monotonic", "tilt_x", "tilt_y", "tilt_z", "turn_x", "turn_y", "turn_z")


def exact_windows(session, ends):
    """Use the original block ends and frame counts, without re-sampling the WAV."""
    chunks = deque()
    frame_count = 0
    previous_end = None
    windows = {}
    for start, data, rate in iter_segments(session):
        if previous_end is not None and abs(start - previous_end) > TIMESTAMP_TOLERANCE:
            chunks.clear()
            frame_count = 0
        chunks.append(data)
        frame_count += len(data)
        while chunks and frame_count - len(chunks[0]) >= rate:
            frame_count -= len(chunks.popleft())
        previous_end = start + len(data) / rate
        if previous_end in ends and frame_count >= rate:
            windows[previous_end] = (np.concatenate(chunks)[-rate:], rate)
    return windows


def archived_motion(session, marks):
    """Merge overlapping marked captures without duplicating IMU packets."""
    motion = {}
    coverage = []
    for mark in marks:
        directory = session / "captures" / f"mark_{int(mark['label_id']):04d}"
        metadata = json.loads((directory / "metadata.json").read_text())
        time = float(metadata["label_monotonic"])
        start = time - metadata["requested_seconds_before"]
        end = time + metadata["requested_seconds_after"]
        if metadata["closed_before_window_end"]:
            end = time + (metadata["last_motion_seconds_from_mark"] or 0)
        coverage.append((start, end))
        for packet in read_rows(directory / "motion.csv"):
            row = tuple(float(packet[field]) for field in MOTION_FIELDS)
            motion[row[0]] = row
    return sorted(motion.values()), coverage


def score_windows(session, ends, marks, baseline, candidate):
    motion, coverage = archived_motion(session, marks)
    scored = {}
    for end, (audio, rate) in exact_windows(session, ends).items():
        if not any(start <= end - 1 and end <= stop for start, stop in coverage):
            continue
        try:
            features = extract_activity_features(audio, rate, motion, end)
            voice = extract_voice_features(audio, rate) if baseline.has_voice_specialist else None
            old, _ = baseline.predict(features, voice_features=voice)
            if candidate is None or rate < 50:
                new = old
            else:
                scores = baseline.scores(features, voice_features=voice)
                detail = extract_motion_cough_features(audio, rate, motion, end)
                new, _ = candidate.predict(scores, detail)
        except InsufficientMotion:
            old = new = "uncertain"
        scored[end] = (old, new)
    return scored


def possible_screen(raw, history, archived_history, logged):
    """Constrain the display without inventing the unrecorded GUI redraw time."""
    if raw != "uncertain":
        return [raw]
    if len(history) < 2:
        return [raw]
    first, second, current = logged
    if second - first > 0.5 or current - second > 0.5:
        return [raw]
    a, b = history
    if a is None or b is None:
        # A missing previous prediction matters only when a voice hold is possible.
        known = b if a is None else a
        if known is not None and known != "speaking":
            return [raw]
        return [raw, "speaking"]
    if a != b or b != "speaking":
        return [raw]
    old_a, old_b, old_raw, old_display = archived_history
    if old_raw == "uncertain" and old_a == old_b and old_a in ("coughing", "speaking"):
        # The actual mark tells us whether the UI hold survived its expiry timer.
        return [raw] if old_display == raw else [b]
    return [raw, b]


def evaluate_replay(session, baseline, candidate=None, holdout_marks=None):
    """Compare a previously fitted model on specified label IDs; never train on these rows."""
    session = Path(session)
    all_marks = read_rows(session / "activity_at_marks.csv")
    selected_ids = None if holdout_marks is None else set(holdout_marks)
    marks = [m for m in all_marks if selected_ids is None or int(m["label_id"]) in selected_ids]
    if selected_ids is not None and selected_ids != {int(m["label_id"]) for m in marks}:
        raise ValueError("Requested mark IDs are absent from the archived screen log.")
    stream = read_rows(session / "activity_predictions.csv")
    ends = [float(x["window_end_monotonic"]) for x in stream]
    index = {end: i for i, end in enumerate(ends)}
    if len(ends) != len(index) or ends != sorted(ends):
        raise ValueError("Stream windows must be ordered and unique.")
    wanted = set()
    for m in marks:
        if m["window_end_monotonic"]:
            i = index[float(m["window_end_monotonic"])]
            wanted.update(ends[max(0, i - 2):i + 1])
    scored = score_windows(session, wanted, all_marks, baseline, candidate) if wanted else {}
    decisions = []
    for mark in marks:
        expected = mark["manual_activity"]
        expected = "swallowing" if expected in POSITIVE_CATEGORIES else expected
        current = float(mark["window_end_monotonic"]) if mark["window_end_monotonic"] else None
        old_raw = mark["estimated_activity"]
        screen = mark["displayed_activity"]
        row = {"mark": int(mark["label_id"]), "expected": expected,
               "window_end_monotonic": current, "original_raw": old_raw, "original_screen": screen,
               "replayed_primary_raw": None, "candidate_raw": None, "candidate_screen_options": []}
        if current in scored:
            i = index[current]
            old, new = scored[current]
            row["replayed_primary_raw"] = old
            if candidate is not None:
                row["candidate_raw"] = new
                previous = ends[max(0, i - 2):i]
                context = [scored.get(e, (None, None))[1] for e in previous]
                past = [stream[index[e]]["estimated_activity"] for e in previous]
                row["candidate_screen_options"] = possible_screen(new, context, past + [old_raw, screen],
                                                                   previous + [current])
        elif current is None and screen == "waiting" and candidate is not None:
            row["candidate_screen_options"] = ["waiting"]
        decisions.append(row)
    rows_replayed = [d for d in decisions if d["replayed_primary_raw"] is not None]
    summary = {"session": session.name, "selected_marks": len(marks), "replayed_windows": len(rows_replayed),
               "primary_raw_reproduced": sum(d["replayed_primary_raw"] == d["original_raw"] for d in rows_replayed),
               "archived_display_correct": sum(d["expected"] == d["original_screen"] for d in decisions),
               "decisions": decisions}
    if candidate is not None:
        summary.update(candidate_raw_correct=sum(d["expected"] == d["candidate_raw"] for d in rows_replayed),
                       candidate_screen_correct_min=sum(d["candidate_screen_options"] == [d["expected"]]
                                                        for d in decisions),
                       candidate_screen_correct_max=sum(not d["candidate_screen_options"] or
                                                        d["expected"] in d["candidate_screen_options"]
                                                        for d in decisions))
    summary["by_activity"] = {}
    for activity in sorted({d["expected"] for d in decisions}):
        group = [d for d in decisions if d["expected"] == activity]
        counts = {"marks": len(group),
                  "archived_screen_correct": sum(d["original_screen"] == activity for d in group)}
        if candidate is not None:
            counts["candidate_screen_correct_min"] = sum(d["candidate_screen_options"] == [activity] for d in group)
            counts["candidate_screen_correct_max"] = sum(not d["candidate_screen_options"] or
                                                         activity in d["candidate_screen_options"] for d in group)
        summary["by_activity"][activity] = counts
    return summary


def main():
    from activity_model import ActivityModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path(__file__).resolve().parent / "models/activity_voice_cough.json")
    parser.add_argument("--candidate", type=Path, help="An already fitted motion/cough sidecar.")
    parser.add_argument("--marks", type=int, nargs="+")
    args = parser.parse_args()
    candidate = None
    if args.candidate:
        candidate = MotionCoughSpecialist.load(args.candidate)
    print(json.dumps(evaluate_replay(args.session, ActivityModel.load(args.baseline), candidate, args.marks),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
