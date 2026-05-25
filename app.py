import os
import pandas as pd
import gradio as gr
from classifier import PhraseClassifier

# ----------------------------------------------------------------------
# 1. Load the classifier (model path can be an env var or default)
# ----------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "classifier.pt")
classifier = PhraseClassifier(MODEL_PATH)

# ----------------------------------------------------------------------
# 2. Load the symptoms lookup table
# ----------------------------------------------------------------------
CSV_PATH = "symptoms_dataset.csv"
symptoms_df = pd.read_csv(CSV_PATH)

# Build a dictionary: symptom_description (cleaned) -> (body_part, sub_category)
lookup = {}
for _, row in symptoms_df.iterrows():
    key = row["symptom_description"].strip().lower()
    lookup[key] = (row["body_part"], row["sub_category"])

# ----------------------------------------------------------------------
# 3. Prediction function for Gradio
# ----------------------------------------------------------------------
def classify_and_lookup(audio_filepath):
    """
    audio_filepath : str – path to the recorded WAV file (Gradio supplies this)
    Returns a markdown string with the result.
    """
    if audio_filepath is None:
        return "⚠️ No audio recorded. Please try again."

    # Run the classifier
    predicted_phrase = classifier.predict(audio_filepath)
    if predicted_phrase is None:
        return "❌ Could not extract an embedding from the audio."

    # Normalise the predicted phrase for lookup
    clean_phrase = predicted_phrase.strip().lower()
    match = lookup.get(clean_phrase, None)

    if match:
        body_part, sub_issue = match
        result = (
            f"### 🩺 Matched Symptom\n"
            f"**Heard:** _{predicted_phrase}_\n\n"
            f"**Body Part:** {body_part}\n\n"
            f"**Sub‑Issue:** {sub_issue}"
        )
    else:
        result = (
            f"### ❓ No exact match found\n"
            f"**Heard:** _{predicted_phrase}_\n\n"
            f"_This phrase is not in the symptoms database._"
        )
    return result

# ----------------------------------------------------------------------
# 4. Build the Gradio interface
# ----------------------------------------------------------------------
iface = gr.Interface(
    fn=classify_and_lookup,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="Record your symptom"),
    outputs=gr.Markdown(),
    title="Twi Symptom Classifier",
    description=(
        "Press **Record** and describe your symptom in Twi or English. "
        "The app will recognise the phrase and show the affected body part and issue."
    ),
    allow_flagging="never",
)

iface.launch()