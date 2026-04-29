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
    analyzer_summary = rules.get("analyzer_summary", {})
    analyzer_features = rules.get("analyzer_features", {})
    rule_hits = rules.get("rule_hits", [])
    if not isinstance(analyzer_summary, dict):
        analyzer_summary = {}
    if not isinstance(analyzer_features, dict):
        analyzer_features = {}
    if not isinstance(rule_hits, list):
        rule_hits = []

    timing = analyzer_summary.get("timing", {})
    dynamics = analyzer_summary.get("dynamics", {})
    overall = analyzer_summary.get("overall", {})

    bpm = analyzer_features.get("bpm_estimate", features.get("bpm_estimate"))
    onset_count = timing.get("onset_count", analyzer_features.get("onset_count", features.get("onset_count")))
    avg_interval_ms = timing.get("average_interval_ms", analyzer_features.get("average_interval_ms", features.get("average_interval_ms")))
    interval_std_ms = timing.get("interval_std_ms", analyzer_features.get("interval_std_ms", features.get("interval_std_ms")))
    avg_strength_db = dynamics.get("average_strength_db", analyzer_features.get("average_strength_db", features.get("average_strength_db")))
    strength_std_db = dynamics.get("strength_std_db", analyzer_features.get("strength_std_db", features.get("strength_std_db")))
    dynamic_range_db = dynamics.get("dynamic_range_db", analyzer_features.get("dynamic_range_db", features.get("dynamic_range_db")))

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

    if isinstance(overall.get("short_summary"), str) and overall.get("short_summary"):
        parts.append(overall["short_summary"])
    elif rule_hits:
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


def _translate_analyzer_summary(analyzer_summary_en: str) -> str:
    return (
        analyzer_summary_en
        .replace("The recording is around", "这段录音大约是")
        .replace(" BPM with ", " BPM，一共检测到 ")
        .replace(" detected hits.", " 次敲击。")
        .replace("The average gap between hits is about", "平均每次敲击之间的间隔约为")
        .replace(", with interval variation around", "，间隔波动大约为")
        .replace(" ms.", " 毫秒。")
        .replace("Loudness averages around", "平均力度约为")
        .replace(", with strength variation", "，力度波动约为")
        .replace(" dB and dynamic range ", " dB，动态范围约为 ")
        .replace("The student's timing is unstable.", "节奏稳定性还有提升空间。")
        .replace("The student is behind the beat.", "整体节拍偏慢，略微落在拍子后面。")
        .replace("The student is ahead of the beat.", "整体节拍偏快，略微冲在拍子前面。")
        .replace("The sample does not show a major rule-triggered timing or dynamics problem.", "这段样本里没有明显的节奏或力度问题。")
        .replace("Timing needs more consistency, while dynamics look manageable.", "节奏一致性需要优先加强，力度整体还可以。")
        .replace("Dynamics need more control, while timing looks manageable.", "力度控制需要加强，节奏整体还可以。")
        .replace("Timing and dynamics look reasonably controlled in this sample.", "这段样本里的节奏和力度整体控制得还不错。")
        .replace("Timing is unstable, while dynamics also need attention.", "节奏不够稳定，同时力度控制也需要继续关注。")
    )


def _translate_advice_lines(advice_en: str) -> str:
    lines = [line.strip() for line in advice_en.splitlines() if line.strip()]
    translated: List[str] = []
    for line in lines:
        text = line.removeprefix("- ").strip()
        text = (
            text.replace("Practice with a ", "用 ")
            .replace(" BPM metronome. ", " BPM 的节拍器练习。")
            .replace("Practice counting subdivisions.", "练习数拍和细分音符。")
            .replace("Practice with a slow metronome.", "先用较慢的节拍器练习。")
            .replace("Practice waiting for the click with a slow metronome.", "用较慢的节拍器练习，学会更耐心地等点击声。")
            .replace("Practice alternating light hits and accents.", "练习轻击和重音之间的对比。")
            .replace("Keep the spacing between hits as even as possible.", "尽量让每次敲击之间的间隔保持均匀。")
            .replace("Record another short take and compare whether the timing and dynamics stay consistent.", "再录一小段，对比节奏和力度是否更稳定。")
        )
        translated.append(f"- {text}")
    return "\n".join(translated)


def _translate_encouragement(encouragement_en: str) -> str:
    return (
        encouragement_en
        .replace(
            "Good job getting a full take recorded - a little focused timing work will make the groove feel much stronger.",
            "这次完整录下来已经很不错了，再专注练一点节奏稳定性，律动就会明显更扎实。",
        )
        .replace(
            "Good work - you already have the pattern in place, and a bit more dynamic contrast will make it sound more musical.",
            "你已经把基本型打出来了，再多一点强弱对比，听起来会更有音乐感。",
        )
        .replace(
            "Good work recording your playing - steady small improvements will add up quickly.",
            "能把自己的演奏录下来就是很好的开始，持续一点点进步，很快就会听出差别。",
        )
    )


def _build_markdown(
    analyzer_summary_en: str,
    analyzer_summary_zh: str,
    coaching_advice_en: str,
    coaching_advice_zh: str,
    encouragement_en: str,
    encouragement_zh: str,
) -> str:
    return (
        "## Analyzer Summary / 分析总结\n"
        f"EN: {analyzer_summary_en}\n\n"
        f"中文：{analyzer_summary_zh}\n\n"
        "## Coaching Advice / 练习建议\n"
        "EN:\n"
        f"{coaching_advice_en}\n\n"
        "中文：\n"
        f"{coaching_advice_zh}\n\n"
        "## Encouragement / 鼓励\n"
        f"EN: {encouragement_en}\n\n"
        f"中文：{encouragement_zh}\n"
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
    analyzer_summary_zh = _translate_analyzer_summary(analyzer_summary)
    coaching_advice = _build_coaching_advice(rules, target_bpm=target_bpm)
    coaching_advice_zh = _translate_advice_lines(coaching_advice)
    encouragement = _build_encouragement(rules)
    encouragement_zh = _translate_encouragement(encouragement)
    markdown = _build_markdown(
        analyzer_summary,
        analyzer_summary_zh,
        coaching_advice,
        coaching_advice_zh,
        encouragement,
        encouragement_zh,
    )

    markdown_destination = _infer_feedback_output_path(features, output_path)
    json_destination = markdown_destination.with_suffix(".json")
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.write_text(markdown, encoding="utf-8")

    feedback_json = {
        "analyzer_summary_zh": analyzer_summary_zh,
        "analyzer_summary_structured": rules.get("analyzer_summary", {}),
        "analyzer_summary": analyzer_summary,
        "coaching_advice_zh": coaching_advice_zh,
        "coaching_advice": coaching_advice,
        "encouragement_zh": encouragement_zh,
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
        "analyzer_summary_zh": analyzer_summary_zh,
        "analyzer_summary": analyzer_summary,
        "coaching_advice_zh": coaching_advice_zh,
        "coaching_advice": coaching_advice,
        "encouragement_zh": encouragement_zh,
        "encouragement": encouragement,
        "markdown": markdown,
        "feedback_output_path": str(markdown_destination.resolve()),
        "feedback_json_output_path": str(json_destination.resolve()),
    }
