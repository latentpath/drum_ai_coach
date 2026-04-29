"""
Template-based feedback layer for Drum AI Coach.

For now this does not call the OpenAI API. It produces a structured feedback
result and saves it into `data/processed/` so the pipeline is easy to inspect
and later swap over to an LLM-backed implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_OUTPUT_DIR = Path("data/processed")


def build_prompt(features: Dict[str, Any], rules: Dict[str, Any], *, target_bpm: int = 60) -> str:
    return f"""
You are a friendly drum practice coach.

Student features:
{features}

Rule-based observations:
{rules}

Target BPM:
{target_bpm}

Please output in this format:

## Analyzer Summary
Briefly summarize what the features suggest.

## Coaching Advice
Give 1-2 concrete practice suggestions.

## Encouragement
Give one warm and encouraging sentence.

Rules:
- Do not invent information not supported by the features.
- Mention timing/dynamics only if they appear in the rules.
- Keep the feedback concise and beginner-friendly.
"""


def _format_number(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "unknown"


def _build_analyzer_summary(features: Dict[str, Any], rules: Dict[str, Any]) -> str:
    analyzer_features = rules.get("analyzer_features", {})
    rule_hits = rules.get("rule_hits", [])
    if not isinstance(analyzer_features, dict):
        analyzer_features = {}
    if not isinstance(rule_hits, list):
        rule_hits = []

    bpm = analyzer_features.get("bpm_estimate", features.get("bpm_estimate"))
    onset_count = analyzer_features.get("onset_count", features.get("onset_count"))
    avg_interval_ms = analyzer_features.get("average_interval_ms", features.get("average_interval_ms"))
    interval_std_ms = analyzer_features.get("interval_std_ms", features.get("interval_std_ms"))
    avg_strength_db = analyzer_features.get("average_strength_db", features.get("average_strength_db"))
    strength_std_db = analyzer_features.get("strength_std_db", features.get("strength_std_db"))
    dynamic_range_db = analyzer_features.get("dynamic_range_db", features.get("dynamic_range_db"))

    parts: List[str] = []
    parts.append(
        f"The recording is around {_format_number(bpm, 0)} BPM with {onset_count if onset_count is not None else 'an unknown number of'} detected hits."
    )
    parts.append(
        f"The average gap between hits is about {_format_number(avg_interval_ms)} ms, with interval variation around {_format_number(interval_std_ms)} ms."
    )
    parts.append(
        f"Loudness averages around {_format_number(avg_strength_db)} dB, with strength variation {_format_number(strength_std_db)} dB and dynamic range {_format_number(dynamic_range_db)} dB."
    )

    if rule_hits:
        issue_summaries = [hit.get("summary") for hit in rule_hits if isinstance(hit.get("summary"), str)]
        if issue_summaries:
            parts.append(" ".join(issue_summaries))
    else:
        parts.append("The sample does not show a major rule-triggered timing or dynamics problem.")

    return " ".join(parts)


def _build_coaching_advice(rules: Dict[str, Any], *, target_bpm: int) -> str:
    rule_hits = rules.get("rule_hits", [])
    if not isinstance(rule_hits, list):
        rule_hits = []

    suggestions: List[str] = []
    for hit in rule_hits[:2]:
        suggestion = hit.get("suggestion")
        if isinstance(suggestion, str) and suggestion not in suggestions:
            suggestions.append(suggestion)

    if not suggestions:
        suggestions.append(f"Practice with a {target_bpm} BPM metronome and keep the spacing between hits as even as possible.")
        suggestions.append("Record another short take and compare whether the timing and dynamics stay consistent.")
    else:
        suggestions[0] = f"Practice with a {target_bpm} BPM metronome. {suggestions[0]}"

    return "\n".join(f"- {item}" for item in suggestions[:2])


def _build_encouragement(rules: Dict[str, Any]) -> str:
    rule_hits = rules.get("rule_hits", [])
    if not isinstance(rule_hits, list):
        rule_hits = []

    if any(hit.get("issue") == "unstable_timing" for hit in rule_hits):
        return "Good job getting a full take recorded - a little focused timing work will make the groove feel much stronger."
    if any(hit.get("issue") == "flat_dynamics" for hit in rule_hits):
        return "Good work - you already have the pattern in place, and a bit more dynamic contrast will make it sound more musical."
    return "Good work recording your playing - steady small improvements will add up quickly."


def _build_markdown(analyzer_summary: str, coaching_advice: str, encouragement: str) -> str:
    return (
        "## Analyzer Summary\n"
        f"{analyzer_summary}\n\n"
        "## Coaching Advice\n"
        f"{coaching_advice}\n\n"
        "## Encouragement\n"
        f"{encouragement}\n"
    )


def _infer_feedback_output_path(features: Dict[str, Any], output_path: Optional[str]) -> Path:
    if output_path:
        return Path(output_path)

    source_path = features.get("source_path")
    stem = Path(source_path).stem if isinstance(source_path, str) and source_path else "audio"
    return DEFAULT_OUTPUT_DIR / f"{stem}_feedback.md"


def generate_feedback(
    features: Dict[str, Any],
    rules: Dict[str, Any],
    *,
    target_bpm: int = 60,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    analyzer_summary = _build_analyzer_summary(features, rules)
    coaching_advice = _build_coaching_advice(rules, target_bpm=target_bpm)
    encouragement = _build_encouragement(rules)
    markdown = _build_markdown(analyzer_summary, coaching_advice, encouragement)

    markdown_destination = _infer_feedback_output_path(features, output_path)
    json_destination = markdown_destination.with_suffix(".json")
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text(markdown, encoding="utf-8")

    feedback_json = {
        "analyzer_summary": analyzer_summary,
        "coaching_advice": coaching_advice,
        "encouragement": encouragement,
        "markdown": markdown,
        "target_bpm": target_bpm,
        "rule_hits": rules.get("rule_hits", []),
        "analyzer_features": rules.get("analyzer_features", {}),
        "source_path": features.get("source_path"),
        "features_json_path": features.get("json_output_path"),
    }
    json_destination.write_text(
        json.dumps(feedback_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "analyzer_summary": analyzer_summary,
        "coaching_advice": coaching_advice,
        "encouragement": encouragement,
        "markdown": markdown,
        "feedback_output_path": str(markdown_destination.resolve()),
        "feedback_json_output_path": str(json_destination.resolve()),
    }
