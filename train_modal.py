#!/usr/bin/env python3
"""
Train phrase classifier on Modal – with persistent audio caching.
Edit the variables below and run:
    modal run train_modal.py
"""
import subprocess
import os
import modal

# ========== EDIT THESE ==========
PHRASES_FILE = "sample_phrases/twi-test.txt"   # relative to project root
LANG = "twi"
VOICE = "twi_GH-kofi-medium"
SAMPLES = 15
EPOCHS = 50
HIDDEN_DIM = 2048
# ================================

app = modal.App("train-twi-classifier")
volume = modal.Volume.from_name("phrase-classifier-models", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", "librosa", "soundfile", "datasets",
        "tqdm", "openwakeword", "onnxruntime", "piper-tts",
    )
    .add_local_dir(".", remote_path="/phrase_classifier")
)

# Writable temporary copy of the project
WORKDIR = "/tmp/phrase_classifier"

@app.function(
    image=image,
    cpu=8,
    timeout=10 * 3600,
    volumes={"/models": volume},
)
def train():
    import shutil
    import tempfile

    # Copy the read‑only image code to a writable location
    if not os.path.exists(WORKDIR):
        shutil.copytree("/phrase_classifier", WORKDIR)
    os.chdir(WORKDIR)

    # Patch train.py to persist audio in the volume and skip existing files
    patch_train_py()

    cmd = [
        "python", "train.py",
        "--phrases", PHRASES_FILE,
        "--lang", LANG,
        "--voice", VOICE,
        "--samples", str(SAMPLES),
        "--epochs", str(EPOCHS),
        "--hidden-dim", str(HIDDEN_DIM),
        "--output", "/models/trained_model.pt",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done! Model saved to /models/trained_model.pt")
    print("Generated audio cached in /models/audio_cache/")

def patch_train_py():
    """Modify train.py to use a persistent audio directory and skip existing files."""
    train_py = os.path.join(WORKDIR, "train.py")
    with open(train_py, "r") as f:
        content = f.read()

    # Replace the temporary directory block with persistent, skip‑existing logic
    old_block = """    with tempfile.TemporaryDirectory() as tmpdir:
        for phrase, label in label_map.items():
            for i in range(args.samples):
                # Random augmentation parameters
                length_scale = random.uniform(0.9, 1.1)
                noise_scale  = random.uniform(0.0, 0.05)
                noise_w      = random.uniform(0.0, 0.05)

                out_path = os.path.join(tmpdir, f"{label}_{i}.wav")"""

    new_block = """    audio_dir = "/models/audio_cache"
    os.makedirs(audio_dir, exist_ok=True)
    for phrase, label in label_map.items():
        for i in range(args.samples):
            out_path = os.path.join(audio_dir, f"{label}_{i}.wav")
            # Skip if audio already exists
            if os.path.exists(out_path):
                emb = extract_embedding(embed_model, out_path)
                if emb is not None:
                    X.append(emb)
                    y.append(label)
                continue

            # Random augmentation parameters
            length_scale = random.uniform(0.9, 1.1)
            noise_scale  = random.uniform(0.0, 0.05)
            noise_w      = random.uniform(0.0, 0.05)"""

    # Also adjust the success check part – keep as is, but ensure indentation
    # The original code after the out_path line has:
    #                 success = piper.synthesize(...)
    #                 if not success:
    #                     continue
    #                 emb = extract_embedding(...)
    # We need to keep those lines and the rest, but remove the trailing tmpdir context.
    # We'll replace the whole block up to the line that closes the tmpdir `with` block.

    # To make it safe, we'll just replace the exact old block with the new one.
    # Then we must also remove the line that later does `continue` and the rest, which still works
    # but we must remove the final part that writes `out_path` – actually we keep it as is,
    # just the indentation needs to be preserved. The new_block above changes the loop, but we haven't
    # inserted the actual synthesis call. So we need to include it.

    # Let's construct the full replacement snippet properly.
    replacement = """    audio_dir = "/models/audio_cache"
    os.makedirs(audio_dir, exist_ok=True)
    for phrase, label in label_map.items():
        for i in range(args.samples):
            out_path = os.path.join(audio_dir, f"{label}_{i}.wav")
            # Skip if audio already exists
            if os.path.exists(out_path):
                emb = extract_embedding(embed_model, out_path)
                if emb is not None:
                    X.append(emb)
                    y.append(label)
                continue

            # Random augmentation parameters
            length_scale = random.uniform(0.9, 1.1)
            noise_scale  = random.uniform(0.0, 0.05)
            noise_w      = random.uniform(0.0, 0.05)

            success = piper.synthesize(phrase, out_path,
                                       length_scale=length_scale,
                                       noise_scale=noise_scale,
                                       noise_w=noise_w)
            if not success:
                continue
            emb = extract_embedding(embed_model, out_path)
            if emb is not None:
                X.append(emb)
                y.append(label)"""

    # Now replace the whole old block (including the opening `with` line up to the end of that block)
    # We'll locate the line `with tempfile.TemporaryDirectory() as tmpdir:` and replace everything
    # from that line to the corresponding unindent (end of the `with` block). Simpler: find
    # the string and replace with our new block, then also delete the leftover lines (like the closing
    # of the `with` block). Since the old block ends with an unindent after the loop, we can just
    # replace the entire snippet.
    # A straightforward approach: replace the old block as it appears exactly in the original.
    # However, there might be slight variations. We'll do a search-and-replace of the specific text.

    if old_block in content:
        content = content.replace(old_block, replacement)
        # Remove the now‑unnecessary `with` closing indentation: the old block had the `with` line,
        # then the loops, then the end of the `with` block (the code unindents after the `if __name__...`).
        # Our replacement no longer uses `with`, so the extra indentation lines (like `continue` after
        # synthesis) are already included. No extra closing needed.
    else:
        # Fallback: just add a warning, maybe the script format changed
        print("Warning: Could not patch train.py automatically. Audio will not be cached.")
        return

    with open(train_py, "w") as f:
        f.write(content)
    print("Patched train.py to cache audio in /models/audio_cache/")

@app.local_entrypoint()
def main():
    print(f"Training with phrases={PHRASES_FILE}, lang={LANG}, voice={VOICE}, "
          f"samples={SAMPLES}, epochs={EPOCHS}, hidden_dim={HIDDEN_DIM}")
    train.remote()
    print("\nTraining submitted. Retrieve model with:")
    print("  modal volume get phrase-classifier-models trained_model.pt .")
    print("Audio files will remain in /models/audio_cache/ for future runs.")
