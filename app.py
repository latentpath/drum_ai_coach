from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from feature_extractor import extract_features
from llm_coach import generate_feedback
from rules import get_rules


RAW_DIR = Path("data/raw")


def _fmt(value, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def _score_label(score):
    if score is None:
        return "Unknown"
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs Work"
    return "Focus Required"


def _score_badge(label):
    colors = {
        "Excellent": "#16a34a",
        "Good": "#2563eb",
        "Needs Work": "#f59e0b",
        "Focus Required": "#dc2626",
        "Unknown": "#6b7280",
    }
    return f"""
    <span style="
        background-color:{colors.get(label, '#6b7280')};
        color:white;
        padding:4px 10px;
        border-radius:999px;
        font-size:13px;
        font-weight:600;">
        {label}
    </span>
    """


def _score_card(title, score):
    label = _score_label(score)
    score_display = "—" if score is None else f"{score:.0f}/100"

    st.markdown(f"### {title}")
    st.markdown(
        f"<div style='font-size:32px;font-weight:700;'>{score_display}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_score_badge(label), unsafe_allow_html=True)
    if score is not None:
        st.progress(int(score) / 100)


st.title("Drum AI Coach")
st.caption(
    "EN: Record a short drum practice clip, choose a target BPM, then click Analyze to generate a feedback report. "
    "Reports are saved in data/processed/.\n\n"
    "中文：录一小段打鼓练习，选择目标 BPM，然后点击 Analyze 生成反馈报告。"
    "报告会保存到 data/processed/ 目录。"
)

audio_value = st.audio_input("Record your drum practice")
target_bpm = st.number_input("Target BPM", value=60, step=1)

if audio_value:
    st.audio(audio_value)

if st.button("Analyze"):
    if audio_value is None:
        st.error("Please record audio first.")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_value.getvalue())
            temp_audio_path = tmp.name

        try:
            features = extract_features(temp_audio_path, int(target_bpm))
            rules = get_rules(features)

            report_path = f"data/processed/drum_report_{timestamp}.md"
            feedback = generate_feedback(
                features,
                rules,
                target_bpm=int(target_bpm),
                output_path=report_path,
            )
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        score_summary = rules.get("score_summary", {})
        overall_score = score_summary.get("overall_score")
        timing_score = score_summary.get("timing_score")
        dynamics_score = score_summary.get("dynamics_score")

        st.subheader("Practice Score / 练习评分")

        score_col1, score_col2, score_col3 = st.columns(3)

        with score_col1:
            _score_card("Overall", overall_score)

        with score_col2:
            _score_card("Timing", timing_score)

        with score_col3:
            _score_card("Dynamics", dynamics_score)

        st.divider()

        st.subheader("Performance Details / 演奏细节")

        tab1, tab2 = st.tabs(["Timing / 节奏", "Dynamics / 力度"])

        with tab1:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Total Hits", features.get("onset_count", "—"))

            with col2:
                st.metric(
                    "Avg Interval",
                    _fmt(features.get("average_interval_ms"), suffix=" ms"),
                    help="Average time gap between detected drum hits.",
                )

            with col3:
                st.metric(
                    "Timing Deviation",
                    _fmt(features.get("timing_deviation_std_ms"), suffix=" ms"),
                    help="Lower is better. Shows how much your hits deviate from the expected beat.",
                )

        with tab2:
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Average Strength",
                    _fmt(features.get("average_strength_db"), suffix=" dB"),
                    help="Average loudness of your hits.",
                )

            with col2:
                st.metric(
                    "Strength Stability",
                    _fmt(features.get("strength_std_db"), suffix=" dB"),
                    help="Lower means your hit strength is more consistent.",
                )

            with col3:
                st.metric(
                    "Dynamic Range",
                    _fmt(features.get("dynamic_range_db"), suffix=" dB"),
                    help="Difference between softer and louder hits.",
                )

        st.subheader("Feedback / 反馈")
        st.markdown(feedback["markdown"])
        st.caption(f"Saved: {feedback['feedback_output_path']}")

        st.write("Feedback saved to:")
        st.code(feedback["feedback_output_path"])
