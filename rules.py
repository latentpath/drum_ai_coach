"""
Rule layer for Drum AI Coach.

This module keeps as many raw extracted features as possible, then adds
rule-based observations on top for the next feedback stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _copy_if_present(features: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in keys:
        if key in features:
            out[key] = features[key]
    return out


def get_rules(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return:
    - raw_features: as much original extracted data as possible
    - analyzer_features: the main signals rules looked at
    - rule_hits: rule-based problems found from the features
    """
    mean_timing_error_ms = _safe_float(
        features.get("mean_timing_error_ms", features.get("timing_deviation_mean_ms"))
    )
    timing_std_ms = _safe_float(
        features.get("timing_std_ms", features.get("timing_deviation_std_ms"))
    )
    dynamic_range_db = _safe_float(features.get("dynamic_range_db"))

    rule_hits: List[Dict[str, Any]] = []

    if mean_timing_error_ms is not None:
        if mean_timing_error_ms > 60:
            rule_hits.append(
                {
                    "issue": "behind_the_beat",
                    "summary": "The student is behind the beat.",
                    "suggestion": "Practice with a slow metronome.",
                    "evidence": {
                        "mean_timing_error_ms": mean_timing_error_ms,
                    },
                }
            )
        elif mean_timing_error_ms < -60:
            rule_hits.append(
                {
                    "issue": "ahead_of_the_beat",
                    "summary": "The student is ahead of the beat.",
                    "suggestion": "Practice waiting for the click with a slow metronome.",
                    "evidence": {
                        "mean_timing_error_ms": mean_timing_error_ms,
                    },
                }
            )

    if timing_std_ms is not None and timing_std_ms > 40:
        rule_hits.append(
            {
                "issue": "unstable_timing",
                "summary": "The student's timing is unstable.",
                "suggestion": "Practice counting subdivisions.",
                "evidence": {
                    "timing_std_ms": timing_std_ms,
                },
            }
        )

    if dynamic_range_db is not None and dynamic_range_db < 6:
        rule_hits.append(
            {
                "issue": "flat_dynamics",
                "summary": "The student's dynamics are quite flat.",
                "suggestion": "Practice alternating light hits and accents.",
                "evidence": {
                    "dynamic_range_db": dynamic_range_db,
                },
            }
        )

    analyzer_features = {
        "mean_timing_error_ms": mean_timing_error_ms,
        "timing_std_ms": timing_std_ms,
        "dynamic_range_db": dynamic_range_db,
        "bpm_estimate": _safe_float(features.get("bpm_estimate")),
        "onset_count": features.get("onset_count"),
        "average_interval_ms": features.get("average_interval_ms"),
        "interval_std_ms": features.get("interval_std_ms"),
        "average_strength_db": features.get("average_strength_db"),
        "strength_std_db": features.get("strength_std_db"),
    }

    raw_feature_preview = _copy_if_present(
        features,
        [
            "source_path",
            "sample_rate",
            "duration_sec",
            "analysis_type",
            "onset_count",
            "bpm_estimate",
            "beat_interval_sec",
            "average_interval_ms",
            "interval_std_ms",
            "timing_deviation_mean_ms",
            "timing_deviation_std_ms",
            "average_strength_db",
            "strength_std_db",
            "dynamic_range_db",
            "onset_times_seconds",
            "hit_intervals_ms",
            "timing_deviation_ms",
            "onset_rms_db",
            "hits",
            "json_output_path",
        ],
    )

    return {
        "raw_features": features,
        "raw_feature_preview": raw_feature_preview,
        "analyzer_features": analyzer_features,
        "rule_hits": rule_hits,
    }
