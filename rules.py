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


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(abs(denominator))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _classify_tempo_bias(mean_timing_error_ms: Optional[float]) -> str:
    if mean_timing_error_ms is None:
        return "on_time"
    if mean_timing_error_ms > 60:
        return "behind"
    if mean_timing_error_ms < -60:
        return "ahead"
    return "on_time"


def _classify_timing_stability(timing_std_ms: Optional[float]) -> str:
    if timing_std_ms is None:
        return "stable"
    if timing_std_ms > 80:
        return "unstable"
    if timing_std_ms > 40:
        return "slightly_unstable"
    return "stable"


def _classify_dynamics(dynamic_range_db: Optional[float], strength_std_db: Optional[float]) -> str:
    if dynamic_range_db is None:
        return "controlled"
    if dynamic_range_db < 6:
        return "flat"
    if strength_std_db is not None and strength_std_db > 12:
        return "too_variable"
    if dynamic_range_db >= 12:
        return "good_contrast"
    return "controlled"


def _is_metronome_like(
    *,
    timing_std_ms: Optional[float],
    mean_timing_error_ms: Optional[float],
    dynamic_range_db: Optional[float],
    recording_bpm: Optional[float],
    target_bpm: Optional[float],
    onset_count: Optional[int],
) -> bool:
    if onset_count is None or onset_count < 8:
        return False
    if timing_std_ms is None or timing_std_ms > 15:
        return False
    if mean_timing_error_ms is None or abs(mean_timing_error_ms) > 20:
        return False
    if dynamic_range_db is None or dynamic_range_db > 6:
        return False
    if recording_bpm is not None and target_bpm is not None and abs(recording_bpm - target_bpm) > 5:
        return False
    return True


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _label_score(score: float) -> str:
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs work"
    return "Focus required"


def _timing_accuracy_score(
    mean_timing_error_ms: Optional[float],
    true_interval_ms: Optional[float],
) -> float:
    if mean_timing_error_ms is None or true_interval_ms in (None, 0):
        return 0.0

    error_pct = abs(mean_timing_error_ms) / true_interval_ms * 100.0

    if error_pct <= 2:
        return 100.0
    if error_pct <= 5:
        return _clamp_score(100.0 - (error_pct - 2.0) / 3.0 * 15.0)
    if error_pct <= 10:
        return _clamp_score(85.0 - (error_pct - 5.0) / 5.0 * 25.0)
    if error_pct <= 20:
        return _clamp_score(60.0 - (error_pct - 10.0) / 10.0 * 40.0)
    return _clamp_score(20.0 - (error_pct - 20.0) / 20.0 * 20.0)


def _timing_stability_score(
    timing_std_ms: Optional[float],
    true_interval_ms: Optional[float],
) -> float:
    if timing_std_ms is None or true_interval_ms in (None, 0):
        return 0.0

    std_pct = timing_std_ms / true_interval_ms * 100.0

    if std_pct <= 2:
        return 100.0
    if std_pct <= 4:
        return _clamp_score(100.0 - (std_pct - 2.0) / 2.0 * 20.0)
    if std_pct <= 8:
        return _clamp_score(80.0 - (std_pct - 4.0) / 4.0 * 30.0)
    if std_pct <= 15:
        return _clamp_score(50.0 - (std_pct - 8.0) / 7.0 * 40.0)
    return _clamp_score(10.0 - (std_pct - 15.0) / 10.0 * 10.0)


def _strength_stability_score(strength_std_db: Optional[float]) -> float:
    value = float(strength_std_db or 0.0)
    if value <= 8:
        return 100.0
    if value <= 12:
        return _clamp_score(100.0 - (value - 8.0) / 4.0 * 50.0)
    return _clamp_score(50.0 - (value - 12.0) / 12.0 * 50.0)


def _compute_score_summary(
    *,
    mean_timing_error_ms: Optional[float],
    timing_std_ms: Optional[float],
    true_interval_ms: Optional[float],
    dynamic_range_db: Optional[float],
    strength_std_db: Optional[float],
) -> Dict[str, Any]:
    accuracy_score = _timing_accuracy_score(mean_timing_error_ms, true_interval_ms)
    stability_score = _timing_stability_score(timing_std_ms, true_interval_ms)

    timing_score = _clamp_score(0.4 * accuracy_score + 0.6 * stability_score)

    range_score: float
    dynamic_range = dynamic_range_db or 0.0
    if dynamic_range < 6:
        range_score = _clamp_score(dynamic_range / 6.0 * 60.0)
    elif dynamic_range <= 18:
        range_score = _clamp_score(80.0 + (dynamic_range - 6.0) / 12.0 * 20.0)
    elif dynamic_range <= 25:
        range_score = _clamp_score(100.0 - (dynamic_range - 18.0) / 7.0 * 20.0)
    else:
        range_score = 60.0

    strength_consistency_score = _strength_stability_score(strength_std_db)

    dynamics_score = _clamp_score(0.6 * range_score + 0.4 * strength_consistency_score)
    base_score = 0.5 * accuracy_score + 0.3 * stability_score + 0.2 * strength_consistency_score
    penalty = max(0.0, (60.0 - min(accuracy_score, stability_score, strength_consistency_score)) * 0.5)
    overall_score = _clamp_score(base_score - penalty)

    return {
        "overall_score": round(overall_score, 1),
        "timing_score": round(timing_score, 1),
        "dynamics_score": round(dynamics_score, 1),
        "overall": {
            "base_score": round(base_score, 1),
            "penalty": round(penalty, 1),
            "formula": "0.5*accuracy + 0.3*stability + 0.2*strength_stability - penalty",
        },
        "timing": {
            "accuracy_score": round(accuracy_score, 1),
            "stability_score": round(stability_score, 1),
            "label": _label_score(timing_score),
            "formula": "0.4*accuracy_score + 0.6*stability_score",
        },
        "dynamics": {
            "range_score": round(range_score, 1),
            "label": _label_score(dynamics_score),
            "formula": "0.6*range_score + 0.4*strength_internal",
        },
    }


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
    true_interval_ms = _safe_float(features.get("true_interval_ms"))
    onset_count = features.get("onset_count")
    average_interval_ms = _safe_float(features.get("average_interval_ms"))
    interval_std_ms = _safe_float(features.get("interval_std_ms"))
    average_strength_db = _safe_float(features.get("average_strength_db"))
    strength_std_db = _safe_float(features.get("strength_std_db"))
    bpm_estimate = _safe_float(features.get("bpm_estimate"))
    target_bpm = _safe_float(features.get("target_bpm"))

    interval_cv = _safe_ratio(interval_std_ms, average_interval_ms)
    accent_contrast = dynamic_range_db
    tempo_bias = _classify_tempo_bias(mean_timing_error_ms)
    timing_stability = _classify_timing_stability(timing_std_ms)
    dynamics_interpretation = _classify_dynamics(dynamic_range_db, strength_std_db)
    metronome_like = _is_metronome_like(
        timing_std_ms=timing_std_ms,
        mean_timing_error_ms=mean_timing_error_ms,
        dynamic_range_db=dynamic_range_db,
        recording_bpm=bpm_estimate,
        target_bpm=target_bpm,
        onset_count=int(onset_count) if onset_count is not None else None,
    )
    audio_type = "metronome_like" if metronome_like else "performance_like"

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

    if not metronome_like and dynamic_range_db is not None and dynamic_range_db < 6:
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

    timing_issue = tempo_bias != "on_time" or timing_stability != "stable"
    dynamics_issue = (not metronome_like) and dynamics_interpretation in {"flat", "too_variable"}

    if timing_issue and dynamics_issue:
        main_issue = "both"
    elif timing_issue:
        main_issue = "timing"
    elif dynamics_issue:
        main_issue = "dynamics"
    else:
        main_issue = "none"

    if timing_stability == "unstable" or dynamics_interpretation == "too_variable":
        severity = "high"
    elif main_issue in {"timing", "dynamics", "both"}:
        severity = "medium"
    else:
        severity = "low"

    if main_issue == "timing":
        short_summary = "Timing needs more consistency, while dynamics look manageable."
        priority = "Practice timing first before focusing on accents."
    elif main_issue == "dynamics":
        short_summary = "Dynamics need more control, while timing looks manageable."
        priority = "Practice dynamic control after keeping the groove steady."
    elif main_issue == "both":
        short_summary = "Timing is unstable, while dynamics also need attention."
        priority = "Practice timing first before focusing on accents."
    else:
        short_summary = "Timing and dynamics look reasonably controlled in this sample."
        priority = "Keep reinforcing consistency with slow, repeatable practice."

    analyzer_summary = {
        "timing": {
            "onset_count": onset_count,
            "average_interval_ms": average_interval_ms,
            "interval_std_ms": interval_std_ms,
            "interval_cv": interval_cv,
            "mean_timing_error_ms": mean_timing_error_ms,
            "timing_std_ms": timing_std_ms,
            "tempo_bias": tempo_bias,
            "stability": timing_stability,
        },
        "dynamics": {
            "average_strength_db": average_strength_db,
            "dynamic_range_db": dynamic_range_db,
            "accent_contrast": accent_contrast,
            "interpretation": "reference_click" if metronome_like else dynamics_interpretation,
        },
        "overall": {
            "main_issue": main_issue,
            "severity": severity,
            "short_summary": short_summary,
            "priority": priority,
            "audio_type": audio_type,
        },
    }

    score_summary = _compute_score_summary(
        mean_timing_error_ms=mean_timing_error_ms,
        timing_std_ms=timing_std_ms,
        true_interval_ms=true_interval_ms,
        dynamic_range_db=dynamic_range_db,
        strength_std_db=strength_std_db,
    )

    analyzer_features = {
        "mean_timing_error_ms": mean_timing_error_ms,
        "timing_std_ms": timing_std_ms,
        "dynamic_range_db": dynamic_range_db,
        "bpm_estimate": bpm_estimate,
        "target_bpm": target_bpm,
        "onset_count": onset_count,
        "average_interval_ms": average_interval_ms,
        "interval_std_ms": interval_std_ms,
        "average_strength_db": average_strength_db,
        "audio_type": audio_type,
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
        "analyzer_summary": analyzer_summary,
        "score_summary": score_summary,
        "analyzer_features": analyzer_features,
        "audio_type": audio_type,
        "rule_hits": rule_hits,
    }
