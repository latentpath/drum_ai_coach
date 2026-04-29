"""
Feature extraction for Drum AI Coach.

This module now has two main responsibilities:
1. Read a WAV file from `data/raw/`
2. Extract timing + dynamics features and save them as JSON

The JSON output is designed to be easy for `rules.py` and the LLM layer to read.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


AudioInput = Union[str, Path]
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("data/processed")


def _ensure_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _classify_strength(value_db: float, *, low_threshold: float, high_threshold: float) -> str:
    if value_db <= low_threshold:
        return "light"
    if value_db >= high_threshold:
        return "accent"
    return "medium"


def _estimate_tempo(onset_env: np.ndarray, sample_rate: int, hop_length: int) -> Optional[float]:
    """Support both newer and older librosa tempo APIs."""
    if hasattr(librosa.feature, "rhythm") and hasattr(librosa.feature.rhythm, "tempo"):
        tempo_arr = librosa.feature.rhythm.tempo(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
        )
    else:
        tempo_arr = librosa.beat.tempo(
            onset_envelope=onset_env,
            sr=sample_rate,
            hop_length=hop_length,
        )

    if tempo_arr is None or len(tempo_arr) == 0:
        return None
    return float(tempo_arr[0])


def load_wav_from_raw(
    filename: str,
    *,
    raw_dir: AudioInput = DEFAULT_RAW_DIR,
    target_sr: int = 22050,
    max_duration_sec: Optional[float] = 60.0,
) -> Tuple[np.ndarray, int, str]:
    """
    Load a `.wav` file from `data/raw`.

    Returns:
        waveform: mono float32 audio
        sample_rate: resampled sample rate
        source_path: resolved file path string
    """
    raw_root = Path(raw_dir)
    audio_path = (raw_root / filename).resolve()

    if audio_path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {audio_path.name}")
    if not audio_path.exists():
        raise FileNotFoundError(f"WAV file not found: {audio_path}")

    waveform, sample_rate = sf.read(str(audio_path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim == 2:
        waveform = np.mean(waveform, axis=1, dtype=np.float32)

    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    if max_duration_sec is not None and max_duration_sec > 0:
        max_samples = int(sample_rate * max_duration_sec)
        waveform = waveform[:max_samples]

    if sample_rate != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr

    return waveform.astype(np.float32), int(sample_rate), str(audio_path)


def extract_features_to_json(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    source_path: Optional[str] = None,
    output_path: Optional[AudioInput] = None,
    target_bpm: Optional[float] = None,
    note_factor: float = 1.0,
    hop_length: int = 512,
    n_fft: int = 2048,
) -> Dict[str, Any]:
    """
    Extract drum timing/dynamics features and save them to JSON.

    Main outputs:
    - interval information: onset times and hit-to-hit intervals
    - strength information: per-hit loudness and simple light/medium/accent labels
    """
    if waveform.size == 0:
        raise ValueError("Waveform is empty.")

    onset_env = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
        n_fft=n_fft,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=hop_length)

    hit_intervals_sec = np.diff(onset_times) if len(onset_times) >= 2 else np.array([], dtype=np.float32)
    hit_intervals_ms = hit_intervals_sec * 1000.0

    bpm = _estimate_tempo(onset_env, sample_rate, hop_length)
    recording_bpm = bpm

    expected_hit_bpm: Optional[float] = None
    beat_interval_source = "recording_bpm"
    if target_bpm is not None and target_bpm > 0 and note_factor > 0:
        expected_hit_bpm = float(target_bpm) * float(note_factor)
        beat_interval_sec = 60.0 / expected_hit_bpm
        beat_interval_source = "target_bpm"
    else:
        beat_interval_sec = (60.0 / recording_bpm) if recording_bpm else None

    timing_deviation_ms: List[float] = []
    timing_interval_error_ms: List[float] = []
    true_interval_ms: Optional[float] = None
    if beat_interval_sec and len(onset_times) >= 2:
        true_interval_ms = float(beat_interval_sec * 1000.0)
        record_intervals_ms = np.diff(onset_times) * 1000.0
        timing_interval_error_ms = (record_intervals_ms - true_interval_ms).tolist()
        timing_deviation_ms = timing_interval_error_ms

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=n_fft,
        hop_length=hop_length,
    )[0]
    rms_db_curve = librosa.amplitude_to_db(rms, ref=np.max)

    onset_strength_db: List[float] = []
    for frame in onset_frames:
        frame_index = int(frame)
        if 0 <= frame_index < len(rms_db_curve):
            onset_strength_db.append(float(rms_db_curve[frame_index]))

    dynamic_range_db = (
        float(np.max(onset_strength_db) - np.min(onset_strength_db))
        if len(onset_strength_db) >= 2
        else None
    )
    mean_strength_db = float(np.mean(onset_strength_db)) if onset_strength_db else None
    strength_std_db = float(np.std(onset_strength_db)) if len(onset_strength_db) >= 2 else None

    low_threshold = float(np.percentile(onset_strength_db, 33)) if onset_strength_db else -80.0
    high_threshold = float(np.percentile(onset_strength_db, 66)) if onset_strength_db else -20.0

    hits: List[Dict[str, Any]] = []
    for index, onset_time in enumerate(onset_times.tolist()):
        strength_db = onset_strength_db[index] if index < len(onset_strength_db) else None
        interval_sec = float(hit_intervals_sec[index - 1]) if index > 0 and index - 1 < len(hit_intervals_sec) else None
        interval_ms = float(hit_intervals_ms[index - 1]) if index > 0 and index - 1 < len(hit_intervals_ms) else None

        hits.append(
            {
                "hit_index": index + 1,
                "time_sec": float(onset_time),
                "interval_from_previous_sec": interval_sec,
                "interval_from_previous_ms": interval_ms,
                "strength_db": strength_db,
                "strength_label": (
                    _classify_strength(
                        strength_db,
                        low_threshold=low_threshold,
                        high_threshold=high_threshold,
                    )
                    if strength_db is not None
                    else None
                ),
            }
        )

    features: Dict[str, Any] = {
        "source_path": source_path,
        "sample_rate": int(sample_rate),
        "duration_sec": float(len(waveform) / sample_rate),
        "analysis_type": "drum_timing_and_dynamics",
        "onset_count": int(len(onset_times)),
        "bpm_estimate": _ensure_jsonable(recording_bpm),
        "recording_bpm": _ensure_jsonable(recording_bpm),
        "target_bpm": _ensure_jsonable(target_bpm),
        "note_factor": _ensure_jsonable(note_factor),
        "expected_hit_bpm": _ensure_jsonable(expected_hit_bpm),
        "beat_interval_sec": _ensure_jsonable(beat_interval_sec),
        "beat_interval_source": beat_interval_source,
        "true_interval_ms": _ensure_jsonable(true_interval_ms),
        "onset_times_seconds": [float(t) for t in onset_times.tolist()],
        "hit_intervals_sec": [float(x) for x in hit_intervals_sec.tolist()],
        "hit_intervals_ms": [float(x) for x in hit_intervals_ms.tolist()],
        "average_interval_ms": _ensure_jsonable(float(np.mean(hit_intervals_ms)) if len(hit_intervals_ms) > 0 else None),
        "interval_std_ms": _ensure_jsonable(float(np.std(hit_intervals_ms)) if len(hit_intervals_ms) > 1 else None),
        "timing_deviation_ms": [float(x) for x in timing_deviation_ms],
        "timing_interval_error_ms": [float(x) for x in timing_interval_error_ms],
        "mean_timing_error_ms": _ensure_jsonable(
            float(np.mean(np.abs(timing_interval_error_ms))) if len(timing_interval_error_ms) > 0 else None
        ),
        "timing_deviation_mean_ms": _ensure_jsonable(
            float(np.mean(timing_deviation_ms)) if len(timing_deviation_ms) > 0 else None
        ),
        "timing_deviation_std_ms": _ensure_jsonable(
            float(np.std(timing_deviation_ms)) if len(timing_deviation_ms) > 1 else None
        ),
        "onset_rms_db": [float(v) for v in onset_strength_db],
        "average_strength_db": _ensure_jsonable(mean_strength_db),
        "strength_std_db": _ensure_jsonable(strength_std_db),
        "dynamic_range_db": _ensure_jsonable(dynamic_range_db),
        "rms_db_curve": [float(v) for v in rms_db_curve.tolist()],
        "rms_times_seconds": [
            float(t)
            for t in librosa.frames_to_time(
                np.arange(len(rms_db_curve)),
                sr=sample_rate,
                hop_length=hop_length,
            ).tolist()
        ],
        "hits": hits,
    }

    destination: Optional[Path] = Path(output_path) if output_path is not None else None
    if destination is None:
        stem = Path(source_path).stem if source_path else "audio"
        destination = DEFAULT_OUTPUT_DIR / f"{stem}_features.json"

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(features, ensure_ascii=False, indent=2), encoding="utf-8")
    features["json_output_path"] = str(destination.resolve())

    return features


def extract_features(
    audio_input: AudioInput,
    max_duration_sec: Optional[float] = 60.0,
    *,
    output_path: Optional[AudioInput] = None,
    target_bpm: Optional[float] = None,
    note_factor: float = 1.0,
    target_sr: int = 22050,
) -> Dict[str, Any]:
    """
    Compatibility wrapper used by the rest of the app.

    If `audio_input` is a file path, we load it and also export a JSON features file.
    """
    audio_path = Path(audio_input)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if audio_path.suffix.lower() != ".wav":
        raise ValueError("This extractor currently supports `.wav` files only.")

    waveform, sample_rate, source_path = load_wav_from_raw(
        audio_path.name,
        raw_dir=audio_path.parent,
        target_sr=target_sr,
        max_duration_sec=max_duration_sec,
    )

    return extract_features_to_json(
        waveform,
        sample_rate,
        source_path=source_path,
        output_path=output_path,
        target_bpm=target_bpm,
        note_factor=note_factor,
    )
