import os
import gradio as gr
from classifier import PhraseClassifier

# ----------------------------------------------------------------------
# 1. Load the classifier
# ----------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "classifier.pt")
classifier = PhraseClassifier(MODEL_PATH)

# ----------------------------------------------------------------------
# 2. Prediction function - direct text prediction only
# ----------------------------------------------------------------------
def classify_phrase(audio_filepath):
    if audio_filepath is None:
        return (
            "<div style='color:#e74c3c; font-size:18px;'>"
            "⚠️ No audio recorded. Please try again."
            "</div>"
        )

    predicted_phrase = classifier.predict(audio_filepath)
    if predicted_phrase is None:
        return (
            "<div style='color:#e74c3c; font-size:18px;'>"
            "❌ Could not extract an embedding from the audio."
            "</div>"
        )

    # Display the predicted text directly
    result = (
        f"<div style='background:#f0f4ff; border-left:6px solid #3498db; padding:1.5rem; border-radius:12px; margin-top:1rem;'>"
        f"<h2 style='margin-top:0;'>🔮 Predicted Text</h2>"
        f"<p style='font-size:24px; font-weight:600; color:#1e3c72;'>{predicted_phrase}</p>"
        f"<p style='color:#555;'>The model predicts this text based on the audio input.</p>"
        f"</div>"
    )
    return result

# ----------------------------------------------------------------------
# 3. Modern Gradio UI with Blocks + custom CSS
# ----------------------------------------------------------------------
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif;
}
#app-title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
    color: #1e3c72;
}
#app-subtitle {
    text-align: center;
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 2rem;
}
.record-box {
    background: #ffffff;
    border-radius: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    padding: 2rem;
    margin-bottom: 1rem;
}
.result-area {
    margin-top: 1rem;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        "<div id='app-title'>🎯 Speech-to-Text Predictor</div>"
        "<div id='app-subtitle'>Record audio and get the model's text prediction directly</div>"
    )

    with gr.Row(elem_classes="record-box"):
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Record your audio",
            )
            submit_btn = gr.Button("🔮 Predict Text", variant="primary", size="lg")
            clear_btn = gr.Button("🗑️ Clear", variant="secondary", size="sm")

    output_display = gr.HTML(label="Prediction Result")

    # Event handlers
    submit_btn.click(
        fn=classify_phrase,
        inputs=audio_input,
        outputs=output_display,
    )
    clear_btn.click(
        fn=lambda: (None, ""),
        inputs=[],
        outputs=[audio_input, output_display],
    )

    # Example footer
    gr.Markdown(
        "<div style='text-align:center; margin-top:2rem; color:#888;'>"
        "Built with ❤️ using SpeechBrain + PyTorch<br>"
        "Direct Text Prediction Model"
        "</div>"
    )

demo.launch()
