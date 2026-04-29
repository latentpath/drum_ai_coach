from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime

import streamlit as st

from feature_extractor import extract_features
from llm_coach import generate_feedback
from rules import get_rules


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


def _init_state() -> None:
    defaults = {
        "features": None,
        "rules": None,
        "feedback": None,
        "saved_report_path": None,
        "saved_report_name": None,
        "saved_report_json_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _render_results() -> None:
    features = st.session_state.get("features")
    rules = st.session_state.get("rules")
    feedback = st.session_state.get("feedback")
    if not features or not rules or not feedback:
        return

    score_summary = rules.get("score_summary", {})
    overall_score = score_summary.get("overall_score")
    timing_score = score_summary.get("timing_score")
    dynamics_score = score_summary.get("dynamics_score")

    st.subheader("Practice Score / 练习评分")

    score_col1, score_col2, score_col3 = st.columns(3)

    with score_col1:
        _score_card("Overall / 总分", overall_score)

    with score_col2:
        _score_card("Timing / 节奏", timing_score)

    with score_col3:
        _score_card("Dynamics / 力度", dynamics_score)

    st.caption(
        "Timing score = 0.4 × Timing Accuracy + 0.6 × Timing Stability | "
        "Dynamics score = 0.6 × Dynamic Range + 0.4 × Strength Stability | "
        "Overall score = (0.5 × Accuracy + 0.3 × Stability + 0.2 × Strength Stability) - penalty"
    )

    st.divider()

    st.subheader("Performance Details / 演奏细节")

    tab1, tab2 = st.tabs(["Timing / 节奏", "Dynamics / 力度"])

    with tab1:
        col1, col2 = st.columns(2)
        mean_error = features.get("mean_timing_error_ms") or features.get("timing_deviation_mean_ms")
        timing_std = features.get("timing_deviation_std_ms")

        with col1:
            st.metric(
                "Mean Timing Error / 平均偏差",
                _fmt(mean_error, suffix=" ms"),
                help="Average difference between your hits and the expected beat.",
            )
            st.caption(
                """
                **参考标准：**
                - <20 ms：人耳几乎听不出偏差（很强）
                - 20–50 ms：基本 OK
                - >50 ms：开始明显抢拍 / 拖拍
                """
            )

        with col2:
            st.metric(
                "Timing Stability / 稳定性",
                _fmt(timing_std, suffix=" ms"),
                help="How consistent your timing is across hits.",
            )
            st.caption(
                """
                **参考标准：**
                - <20 ms：非常稳（职业级）
                - 20–40 ms：还不错
                - >40 ms：开始明显“抖”
                """
            )

        if mean_error is not None and abs(float(mean_error)) > 50:
            st.warning("你的节奏偏差较大，可能存在明显抢拍或拖拍。")

        if timing_std is not None and float(timing_std) > 40:
            st.warning("你的节奏稳定性还有提升空间，击打间的时间波动比较明显。")

    with tab2:
        col1, col2 = st.columns(2)
        dynamic_range = features.get("dynamic_range_db")
        strength_std = features.get("strength_std_db")

        with col1:
            st.metric(
                "Dynamics Consistency / 力度均匀性",
                _fmt(dynamic_range, suffix=" dB"),
                help="For basic control practice, this shows how evenly your hit strengths are distributed.",
            )
            st.caption(
                """
                **参考标准：**
                - < 6 dB：力度很均匀，适合基础稳定练习
                - 6–12 dB：有轻微力度变化，还可以
                - 12–18 dB：力度变化较明显，可能不够均匀
                - 18–25 dB：力度波动较大
                - > 25 dB：可能有误击、环境噪音、某些 hit 过重/过轻
                """
            )

        with col1:
            st.metric(
                "Strength Stability / 力度稳定性",
                _fmt(strength_std, suffix=" dB"),
                help="数值越小越好，表示你的每一下击打力度越一致。",
            )
            st.caption(
                """
                **参考标准：**
                - < 4 dB：每一下力度非常均匀，控制优秀
                - 4–8 dB：整体比较均匀，但仍有轻微波动
                - 8–12 dB：力度差异明显，控制不够稳定
                - > 12 dB：力度波动较大，听起来不均匀
                """
            )

        with col2:
            st.metric(
                "Accent Contrast / 轻重音对比",
                _fmt(dynamic_range, suffix=" dB"),
                help="For groove or accent practice, this shows how clearly your soft and loud hits separate.",
            )
            st.caption(
                """
                **参考标准：**
                - < 6 dB：太平，轻重音不明显
                - 6–18 dB：比较理想
                - 18–25 dB：对比强，但可能有点夸张
                - > 25 dB：可能不受控
                """
            )

        if dynamic_range is not None and float(dynamic_range) < 6:
            st.info("你的力度差异比较小，适合基础稳定练习；如果目标是 groove，可以尝试把轻重音做得更明显。")
        elif dynamic_range is not None and 6 <= float(dynamic_range) <= 18:
            st.success("你的力度对比处在比较理想的范围，兼顾了控制和表达。")
        elif dynamic_range is not None and 18 < float(dynamic_range) <= 25:
            st.warning("你的力度对比已经比较强了，如果目标是均匀练习，可能稍微有些夸张。")
        elif dynamic_range is not None and float(dynamic_range) > 25:
            st.warning("你的力度波动很大，可能有误击、环境噪音，或某些 hit 过重/过轻。")

    st.subheader("Feedback / 反馈")
    st.markdown(feedback["markdown"])

    action_col1, action_col2 = st.columns(2)

    with action_col1:
        if st.button("保存当前分析"):
            target_bpm = int(st.session_state.get("target_bpm", 60))
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"practice_{target_bpm}bpm_{timestamp}.md"
            report_path = f"data/processed/{filename}"

            saved_feedback = generate_feedback(
                features,
                rules,
                target_bpm=target_bpm,
                output_path=report_path,
                save_output=True,
            )

            st.session_state["feedback"] = saved_feedback
            st.session_state["saved_report_path"] = saved_feedback["feedback_output_path"]
            st.session_state["saved_report_json_path"] = saved_feedback["feedback_json_output_path"]
            st.session_state["saved_report_name"] = filename
            st.success(f"已保存：{filename}")

    saved_report_path = st.session_state.get("saved_report_path")
    saved_report_name = st.session_state.get("saved_report_name")
    saved_report_json_path = st.session_state.get("saved_report_json_path")

    with action_col2:
        if saved_report_path and saved_report_name and os.path.exists(saved_report_path):
            with open(saved_report_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="下载报告到手机",
                    data=f.read(),
                    file_name=saved_report_name,
                    mime="text/markdown",
                )

    if saved_report_path:
        st.caption(f"Saved markdown: {saved_report_path}")
    if saved_report_json_path:
        st.caption(f"Saved json: {saved_report_json_path}")


_init_state()

st.title("Drum AI Coach")
st.caption(
    "EN: Record a short drum practice clip, choose a target BPM, then click Analyze to generate a feedback report. "
    "Save only when you want, then download the report to your device.\n\n"
    "中文：录一小段打鼓练习，选择目标 BPM，然后点击 Analyze 生成反馈。"
    "需要时再点击保存，并下载报告到你的设备。"
)
st.info(
    "EN: Keep the environment as quiet as possible. You can use headphones for the metronome and avoid playing it through speakers.\n\n"
    "中文：请尽量保持当前环境没有噪音干扰。可以戴耳机听节拍器，不要外放。"
)
st.info(
    "EN: After you start recording, you must stop the recorder first. When the audio player appears below, the recording is ready to analyze.\n\n"
    "中文：开始录音后，需要先停止录音。只有下面出现可回放的音频播放器后，才表示录音已经成功，可以点击 Analyze。"
)

audio_value = st.audio_input("Record your drum practice")
target_bpm = st.number_input("Target BPM", value=60, step=1, key="target_bpm")

if audio_value:
    st.success("Recording ready / 录音已准备好")
    st.audio(audio_value)

analyze_clicked = st.button("Analyze", disabled=audio_value is None)

if analyze_clicked:
    if audio_value is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_value.getvalue())
            temp_audio_path = tmp.name

        try:
            features = extract_features(temp_audio_path, int(target_bpm))
            rules = get_rules(features)
            feedback = generate_feedback(
                features,
                rules,
                target_bpm=int(target_bpm),
                save_output=False,
            )
            st.session_state["features"] = features
            st.session_state["rules"] = rules
            st.session_state["feedback"] = feedback
            st.session_state["saved_report_path"] = None
            st.session_state["saved_report_name"] = None
            st.session_state["saved_report_json_path"] = None
        finally:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    else:
        st.error("Please finish recording first. Wait until the audio player appears, then click Analyze.")

_render_results()
