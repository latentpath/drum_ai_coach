from __future__ import annotations

from pathlib import Path

import streamlit as st

from feature_extractor import extract_features
from llm_coach import generate_feedback
from rules import get_rules


RAW_DIR = Path("data/raw")


def _save_uploaded_file(uploaded_file) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    destination = RAW_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getvalue())
    return destination


st.title("Drum AI Coach")

uploaded_file = st.file_uploader("Upload audio", type=["wav"])
target_bpm = st.number_input("Target BPM", value=60, step=1)

if st.button("Analyze"):
    if uploaded_file is None:
        st.error("Please upload a .wav file first.")
    else:
        audio_path = _save_uploaded_file(uploaded_file)

        features = extract_features(str(audio_path), 60)
        rules = get_rules(features)
        feedback = generate_feedback(features, rules, target_bpm=int(target_bpm))

        st.subheader("Basic Statistics")
        st.write(f"Total hits: {features.get('onset_count')}")
        st.write(f"Average interval between hits (ms): {features.get('average_interval_ms')}")
        st.write(f"Interval stability std (ms): {features.get('interval_std_ms')}")
        st.write(f"Timing deviation std (ms): {features.get('timing_deviation_std_ms')}")
        st.write(f"Average hit strength (dB): {features.get('average_strength_db')}")
        st.write(f"Strength stability std (dB): {features.get('strength_std_db')}")
        st.write(f"Dynamic range (dB): {features.get('dynamic_range_db')}")

        st.subheader("Feedback")
        st.markdown(feedback["markdown"])

        st.write("Feedback saved to:")
        st.code(feedback["feedback_output_path"])
