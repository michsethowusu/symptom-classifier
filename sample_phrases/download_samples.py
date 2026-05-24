#!/usr/bin/env python3
"""Download all reference WAVs + texts from the speaker dataset locally."""
import os
import csv
from datasets import load_dataset
from tqdm import tqdm
import soundfile as sf

DATASET_NAME = "michsethowusu/twi-speech-text-multispeaker-clean"
OUTPUT_DIR = "reference_audio"
CSV_PATH = os.path.join(OUTPUT_DIR, "reference_texts.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading {DATASET_NAME} ...")
ds = load_dataset(DATASET_NAME, split="train")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "text"])   # header

    for i, example in enumerate(tqdm(ds, desc="Saving WAVs + texts")):
        audio = example["audio"]["array"]
        sr = example["audio"]["sampling_rate"]
        text = example["text"]
        fname = f"ref_{i:05d}.wav"
        sf.write(os.path.join(OUTPUT_DIR, fname), audio, sr)
        writer.writerow([fname, text])

print(f"Done – {len(ds)} files + '{CSV_PATH}' created in '{OUTPUT_DIR}/'")
