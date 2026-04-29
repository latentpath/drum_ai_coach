# test_main.py

from feature_extractor import extract_features
from rules import get_rules
from llm_coach import generate_feedback

audio_path = "data/raw/test1.wav"

features = extract_features(audio_path, 60)
rules = get_rules(features)
feedback = generate_feedback(features, rules)

print("Features:", features)
print("Rules:", rules)
print("Feedback:", feedback)