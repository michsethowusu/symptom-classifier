#!/usr/bin/env python3
"""
Generate Twi voice‑cloned audio using VoxCPM – parallel GPU workers.
- Iterates through all reference WAVs in REF_AUDIO_DIR, generating one
  audio file per (phrase, ref_audio) pair — no SAMPLES_PER_PHRASE needed.
- Reads ref_text.csv for prompt_text to pass alongside each ref WAV.
- Writes a single metadata.csv in OUTPUT_VOLUME_DIR mapping WAVs to phrases.
- Filenames use a global phrase index so they are unique across workers and
  resumable even if NUM_WORKERS / chunk boundaries change between runs.
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
LOCAL_CSV_PATH = "symptoms_dataset.csv"

VOLUME_MOUNT = "/data"
OUTPUT_VOLUME_DIR = "/data/audio_cache"
REF_AUDIO_DIR = "/data/ref_audio"
REF_CSV_PATH = "/data/ref_audio/ref_text.csv"
MODELSCOPE_CACHE_DIR = "/data/modelscope_cache"
MODAL_VOLUME_NAME = "spark-tts-audio"
CPU_CORES = 1
NUM_WORKERS = 1
# =============================================================================

app = modal.App("voxcpm-twi-gen")
volume = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "sox", "libsox-fmt-all", "ffmpeg")
    .pip_install(
        "torch", "torchaudio", "soundfile", "numpy", "tqdm",
        "voxcpm", "modelscope", "funasr", "librosa", "scipy",
        "sentencepiece", "transformers", "tokenizers", "accelerate",
        "huggingface_hub",
    )
    .add_local_dir(".", remote_path="/local_project")
)

@app.function(
    image=image,
    cpu=CPU_CORES,
    gpu="A10G",
    timeout=10 * 3600,
    volumes={VOLUME_MOUNT: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
@modal.concurrent(max_inputs=NUM_WORKERS)
def generate_chunk(phrases: list[str], chunk_id: int, phrase_offset: int = 0):
    """Generate audio for a chunk of phrases using VoxCPM.

    Args:
        phrases:       The subset of phrases this worker is responsible for.
        chunk_id:      Worker index (used only for logging / progress bars).
        phrase_offset: Index of phrases[0] in the global phrase list.
                       Combined with the local enumerate index this gives a
                       globally unique, stable phrase index for every file,
                       so filenames never collide across workers and a resumed
                       run with a different chunk split still skips already-
                       finished files correctly.

    One output file is produced per (phrase, ref_audio) pair, named:
        phrase_{global_idx:05d}_ref_{ref_idx:03d}.wav
    """

    # ---------- 1. Set up modelscope cache & pre-download ZipEnhancer ----------
    os.environ["MODELSCOPE_CACHE"] = MODELSCOPE_CACHE_DIR
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.makedirs(MODELSCOPE_CACHE_DIR, exist_ok=True)

    try:
        from modelscope import snapshot_download
        snapshot_download("iic/speech_zipenhancer_ans_multiloss_16k_base")
        print(f"Worker {chunk_id}: ZipEnhancer ready.")
    except Exception as e:
        print(f"Worker {chunk_id}: ZipEnhancer download skipped ({e})")

    # ---------- 2. Load VoxCPM ----------
    from voxcpm import VoxCPM
    MODEL_ID = "ghananlpcommunity/voxcpm-twi-ewe-dagbani-full"
    print(f"Worker {chunk_id}: Loading VoxCPM from {MODEL_ID} ...")
    model = VoxCPM.from_pretrained(MODEL_ID)
    print(f"Worker {chunk_id}: Model loaded.")

    # ---------- 3. Load reference WAVs and transcripts ----------
    ref_files = sorted([
        os.path.join(REF_AUDIO_DIR, f)
        for f in os.listdir(REF_AUDIO_DIR) if f.lower().endswith(".wav")
    ])
    if not ref_files:
        raise RuntimeError(f"No WAV files found in {REF_AUDIO_DIR}")

    # Load transcripts from ref_text.csv (ref_audio, transcription columns)
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
        print(f"Worker {chunk_id}: ref_text.csv not found — prompt_text will be None for all refs.")

    print(f"Worker {chunk_id}: {len(ref_files)} reference WAVs found.")

    # ---------- 4. Single shared output directory ----------
    os.makedirs(OUTPUT_VOLUME_DIR, exist_ok=True)

    total_files = len(phrases) * len(ref_files)
    print(f"Worker {chunk_id}: {len(phrases)} phrases × {len(ref_files)} ref files "
          f"(global offset {phrase_offset}) → {total_files} files.")

    # ---------- 5. Shared metadata CSV (one file for all workers) ----------
    metadata_path = os.path.join(OUTPUT_VOLUME_DIR, "metadata.csv")
    write_header = not os.path.exists(metadata_path)
    metadata_file = open(metadata_path, "a", newline="", encoding="utf-8")
    metadata_writer = csv.writer(metadata_file)
    if write_header:
        metadata_writer.writerow(["filename", "phrase", "ref_audio"])

    # ---------- 6. Background saver ----------
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

    # ---------- 7. Generation loop ----------
    try:
        for local_idx, phrase in enumerate(phrases):
            # Global index: stable across re-runs regardless of chunk split
            global_phrase_idx = phrase_offset + local_idx

            for ref_idx, prompt_wav in enumerate(ref_files):
                # One file per (phrase, ref) pair — ref_idx encodes which voice
                wav_name = f"phrase_{global_phrase_idx:05d}_ref_{ref_idx:03d}.wav"
                wav_path = os.path.join(OUTPUT_VOLUME_DIR, wav_name)

                # Resume: skip files that already exist from a previous run
                if os.path.exists(wav_path):
                    pbar.update(1)
                    continue

                try:
                    wav = model.generate(
                        text=phrase,
                        prompt_wav_path=prompt_wav,
                        prompt_text=ref_text_map.get(prompt_wav, None),
                        cfg_value=2.0,
                        inference_timesteps=10,
                        normalize=True,
                        denoise=False,
                        retry_badcase=False,
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


def load_phrases():
    """Load phrases from the local CSV (on your computer)."""
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

    chunk_size = max(1, len(phrases) // NUM_WORKERS)
    chunks = [phrases[i:i+chunk_size] for i in range(0, len(phrases), chunk_size)]

    # Each chunk's starting offset in the global phrase list — gives every
    # file a unique, stable name regardless of NUM_WORKERS or chunk split.
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
