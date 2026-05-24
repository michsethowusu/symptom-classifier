#!/usr/bin/env python3
"""
Spark-TTS version: Twi voice-cloned audio generation (parallel GPU workers)

- For every phrase in symptoms_dataset.csv, generate one output WAV
  per reference audio file (i.e. N_ref files per phrase).
- Uses ref_text.csv for prompt_text (optional).
- Writes metadata.csv inside the output volume.
- Filenames: phrase_{global_idx:05d}_ref_{ref_idx:03d}.wav
"""

import os
import csv
import threading
import queue

import modal
import soundfile as sf
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
LOCAL_CSV_PATH = "symptoms_dataset.csv"          # local CSV with "twi_translation" column

VOLUME_MOUNT = "/data"
OUTPUT_VOLUME_DIR = "/data/audio_cache"          # generated WAVs + metadata.csv
REF_AUDIO_DIR = "/data/ref_audio"               # reference WAVs + ref_text.csv
REF_CSV_PATH = "/data/ref_audio/ref_text.csv"    # columns: ref_audio, transcription
MODEL_CACHE_DIR = "/data/model_cache"            # cached base + checkpoint + final model

MODAL_VOLUME_NAME = "spark-tts-audio"            # Modal volume name
NUM_WORKERS = 1                                  # parallel GPU workers
# =============================================================================

app = modal.App("spark-tts-twi-gen")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

# ── Docker image with Spark-TTS dependencies ───────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")                                    # only git needed for cloning
    .pip_install(
        "torch==2.5.1+cu121",
        "torchaudio==2.5.1+cu121",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "transformers", "soundfile", "numpy", "tqdm",
        "sentencepiece", "soxr", "einops", "omegaconf",
        "accelerate", "safetensors", "einx",
        "huggingface_hub",                           # for snapshot_download
    )
    # Clone Spark-TTS repository and install its requirements
    .run_commands(
        "git clone https://github.com/SparkAudio/Spark-TTS.git /root/Spark-TTS",
        "pip install -r /root/Spark-TTS/requirements.txt",
    )
)

# ── Worker function (will be called NUM_WORKERS times in parallel) ─────────
@app.function(
    image=image,
    cpu=2,                         # enough for I/O
    gpu="A10G",                    # 24 GB VRAM – sufficient for 0.5B model
    timeout=10 * 3600,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],  # HF_TOKEN inside
)
@modal.concurrent(max_inputs=NUM_WORKERS)
def generate_chunk(phrases: list[str], chunk_id: int, phrase_offset: int = 0):
    """
    Generate one WAV per (phrase, ref_audio) pair.

    Args:
        phrases:       List of Twi sentences for this worker.
        chunk_id:      Worker index (for logging).
        phrase_offset: Index of phrases[0] in the global phrase list.
                       Used to build a unique, stable global phrase index.
    """

    import sys
    sys.path.insert(0, "/root/Spark-TTS")
    from cli.SparkTTS import SparkTTS

    import shutil
    import torch
    from pathlib import Path
    from huggingface_hub import snapshot_download

    # ── 1. Prepare the final model directory (cached in volume) ──────────
    BASE_MODEL_ID = "unsloth/Spark-TTS-0.5B"
    CKPT_REPO = "ghananlpcommunity/spark-tts-twi-ewe-dagbani-checkpoints"
    CACHE_DIR = Path(MODEL_CACHE_DIR)
    FINAL_MODEL_DIR = CACHE_DIR / "final_model"

    def prepare_model():
        base_dir = CACHE_DIR / "base"
        if not (base_dir / "LLM").exists():
            print(f"Worker {chunk_id}: Downloading base model …")
            snapshot_download(BASE_MODEL_ID, local_dir=str(base_dir))

        ckpt_dir = CACHE_DIR / "checkpoint"
        if not (ckpt_dir / "model.safetensors").exists():
            print(f"Worker {chunk_id}: Downloading fine-tuned checkpoint …")
            snapshot_download(CKPT_REPO, local_dir=str(ckpt_dir))

        marker = FINAL_MODEL_DIR / ".ready"
        if not marker.exists():
            print(f"Worker {chunk_id}: Assembling final model directory …")
            if FINAL_MODEL_DIR.exists():
                shutil.rmtree(FINAL_MODEL_DIR)
            # Copy non-LLM parts from base
            for item in base_dir.iterdir():
                if item.name != "LLM":
                    dest = FINAL_MODEL_DIR / item.name
                    (shutil.copytree if item.is_dir() else shutil.copy2)(item, dest)
            # Merge LLM folder
            merged_llm = FINAL_MODEL_DIR / "LLM"
            merged_llm.mkdir(exist_ok=True)
            for f in (base_dir / "LLM").iterdir():
                if f.suffix != ".safetensors":
                    dest = merged_llm / f.name
                    (shutil.copytree(f, dest, dirs_exist_ok=True) if f.is_dir()
                     else shutil.copy2(f, dest))
            # Overwrite with fine-tuned safetensors
            shutil.copy2(ckpt_dir / "model.safetensors",
                         merged_llm / "model.safetensors")
            marker.touch()
            print(f"Worker {chunk_id}: Final model directory ready.")
        return str(FINAL_MODEL_DIR)

    model_dir = prepare_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tts = SparkTTS(model_dir, device)
    print(f"Worker {chunk_id}: Spark-TTS loaded on {device}")

    # ── 2. Load reference WAVs and transcripts ─────────────────────────
    ref_files = sorted([
        os.path.join(REF_AUDIO_DIR, f)
        for f in os.listdir(REF_AUDIO_DIR) if f.lower().endswith(".wav")
    ])
    if not ref_files:
        raise RuntimeError(f"No WAV files found in {REF_AUDIO_DIR}")

    ref_text_map = {}
    if os.path.exists(REF_CSV_PATH):
        with open(REF_CSV_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row["ref_audio"].strip()
                txt = row["transcription"].strip()
                full_path = os.path.join(REF_AUDIO_DIR, fname)
                if full_path in ref_files:
                    ref_text_map[full_path] = txt
        print(f"Worker {chunk_id}: {len(ref_text_map)} transcripts loaded from CSV.")
    else:
        print(f"Worker {chunk_id}: ref_text.csv not found — prompt_text will be None.")

    print(f"Worker {chunk_id}: {len(ref_files)} reference WAVs found.")

    # ── 3. Output directory & metadata CSV (shared across workers) ─────
    os.makedirs(OUTPUT_VOLUME_DIR, exist_ok=True)
    total_files = len(phrases) * len(ref_files)
    print(f"Worker {chunk_id}: {len(phrases)} phrases × {len(ref_files)} refs "
          f"(global offset {phrase_offset}) → {total_files} files.")

    metadata_path = os.path.join(OUTPUT_VOLUME_DIR, "metadata.csv")
    write_header = not os.path.exists(metadata_path)
    metadata_file = open(metadata_path, "a", newline="", encoding="utf-8")
    metadata_writer = csv.writer(metadata_file)
    if write_header:
        metadata_writer.writerow(["filename", "phrase", "ref_audio"])

    # ── 4. Background file saver ───────────────────────────────────────
    save_queue = queue.Queue(maxsize=4)
    stop_event = threading.Event()
    failed = []
    pbar = tqdm(total=total_files, desc=f"Chunk {chunk_id}", unit="file", position=chunk_id)

    def saver():
        while not stop_event.is_set() or not save_queue.empty():
            try:
                wav_path, audio_np, phrase, ref_audio = save_queue.get(timeout=0.5)
                sf.write(wav_path, audio_np, 16000)
                fname = os.path.basename(wav_path)
                metadata_writer.writerow([fname, phrase, os.path.basename(ref_audio)])
                metadata_file.flush()
                save_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                failed.append(("save_error", -1, str(e)))

    saver_thread = threading.Thread(target=saver, daemon=True)
    saver_thread.start()

    # ── 5. Generation loop ────────────────────────────────────────────
    try:
        for local_idx, phrase in enumerate(phrases):
            global_phrase_idx = phrase_offset + local_idx

            for ref_idx, prompt_wav in enumerate(ref_files):
                wav_name = f"phrase_{global_phrase_idx:05d}_ref_{ref_idx:03d}.wav"
                wav_path = os.path.join(OUTPUT_VOLUME_DIR, wav_name)

                # Resume: skip already existing files
                if os.path.exists(wav_path):
                    pbar.update(1)
                    continue

                try:
                    with torch.no_grad():
                        wav = tts.inference(
                            text=phrase,
                            prompt_speech_path=prompt_wav,
                            prompt_text=ref_text_map.get(prompt_wav, None),
                        )
                    save_queue.put((wav_path, wav, phrase, prompt_wav))
                except Exception as e:
                    tqdm.write(f"  ❌ Chunk {chunk_id}: {phrase[:30]}... ref {ref_idx}: {e}")
                    failed.append((global_phrase_idx, ref_idx, phrase))

                pbar.update(1)
    finally:
        stop_event.set()
        saver_thread.join()
        metadata_file.close()
        pbar.close()

    if failed:
        log_path = os.path.join(OUTPUT_VOLUME_DIR, f"failed_w{chunk_id:02d}.log")
        with open(log_path, "w") as f:
            for p_idx, r_idx, txt in failed:
                f.write(f"phrase {p_idx} ref {r_idx}: {txt}\n")
        print(f"Worker {chunk_id}: {len(failed)} failures logged.")

    print(f"Worker {chunk_id}: Done. Files in {OUTPUT_VOLUME_DIR}")


# ── Local helpers ───────────────────────────────────────────────────────────
def load_phrases():
    """Load phrases from the local CSV (twi_translation column)."""
    if not os.path.exists(LOCAL_CSV_PATH):
        print(f"Error: Local CSV not found at {LOCAL_CSV_PATH}")
        return []
    phrases = set()
    with open(LOCAL_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "twi_translation" not in reader.fieldnames:
            print("CSV must contain 'twi_translation' column.")
            return []
        for row in reader:
            text = row["twi_translation"].strip()
            if text and text != "TRANSLATION_FAILED":
                phrases.add(text)
    return sorted(phrases)


@app.local_entrypoint()
def main():
    phrases = load_phrases()
    if not phrases:
        print("No phrases found. Exiting.")
        return

    # Split phrases into roughly equal chunks for parallel workers
    chunk_size = max(1, len(phrases) // NUM_WORKERS)
    chunks = [phrases[i:i+chunk_size] for i in range(0, len(phrases), chunk_size)]
    offsets = [i * chunk_size for i in range(len(chunks))]

    print(f"Launching {len(chunks)} parallel workers for {len(phrases)} phrases...")
    futures = [
        generate_chunk.spawn(chunk, idx, offset)
        for idx, (chunk, offset) in enumerate(zip(chunks, offsets))
    ]

    print("Waiting for all workers to finish...")
    for f in futures:
        f.get()

    print("\n✅ All workers done!")
    print(f"\nDownload audio and metadata with:")
    print(f"  modal volume get {MODAL_VOLUME_NAME} /data/audio_cache .")
