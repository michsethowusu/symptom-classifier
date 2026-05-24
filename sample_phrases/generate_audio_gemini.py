#!/usr/bin/env python3
"""
Generate Twi speech audio using Gemini TTS.
- 15 samples per sentence, varied voices/pacing.
- Prompt given entirely in Twi to force Twi output.

Usage: python generate_audio_gemini.py
"""

import csv, os, time, base64, struct, random
import httpx
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
GEMINI_API_KEY = "AIzaSyA-PVd-HE7wwMUln6IdKvpkoPWf4hbkQyc"
INPUT_CSV      = "twi_symptom_translations.csv"
OUTPUT_DIR     = "./gemini_audio"
MODEL_TTS      = "gemini-3.1-flash-tts-preview"
SAMPLE_COUNT   = 5
DELAY_SECONDS  = 2.0
# ============================================================

ALL_VOICES = [
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam",
    "Aoede", "Autonoe", "Callirrhoe", "Charon", "Despina",
    "Enceladus", "Erinome", "Fenrir", "Gacrux", "Iapetus",
    "Kore", "Laomedeia", "Leda", "Orus", "Puck",
    "Pulcherrima", "Rasalgethi", "Sadachbia", "Sadaltager",
    "Schedar", "Sulafat", "Umbriel", "Vindemiatrix",
    "Zephyr", "Zubenelgenubi"
]

PACE_VARIANTS = [
    "Kasa wɔ abɔdeɛ mu, sɛnea ɛyɛ a ɛnyɛ ntɛmntɛm.",
    "Kasa ntɛm kakra, sɛnea nkɔmmɔdie a ɛkɔ so no teɛ.",
    "Kasa ntɛmntɛm a ɛyɛ anigyeɛ.",
    "Kasa brɛoo na ɛda hɔ pefee, sɛnea worekyerɛkyerɛ asɛm bi.",
    "Kasa wɔ ɔkwan a ɛyɛ komm na ɛda hɔ pefee, na ɛtwɛn kakra wɔ nsɛmfua mu.",
    "Kasa ntɛm nanso ma ɛda hɔ pefee wɔ ɔkwan nyinaa mu.",
]

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_TTS}:generateContent?key={GEMINI_API_KEY}"
)

# ---------- audio utilities (unchanged) ----------
def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    bits, rate = 16, 24000
    for part in mime_type.split(";"):
        part = part.strip()
        if part.lower().startswith("rate="):
            try: rate = int(part.split("=", 1)[1])
            except: pass
        elif part.startswith("audio/L"):
            try: bits = int(part.split("L", 1)[1])
            except: pass
    return {"bits_per_sample": bits, "rate": rate}

def pcm_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    p = parse_audio_mime_type(mime_type)
    bits, sr = p["bits_per_sample"], p["rate"]
    data_size = len(audio_data)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE", b"fmt ",
        16, 1, 1, sr, sr * (bits // 8), (bits // 8), bits,
        b"data", data_size,
    )
    return header + audio_data

def build_prompt(text: str, pace: str) -> str:
    """
    Instruction entirely in Twi – the model receives Twi commands
    and a Twi transcript, pushing it to speak Twi.
    """
    # Twi: "Speak in Twi language. Read this: [text]"
    return f"Kasa wɔ Twi kasa mu. {pace} Kenkan eyi: {text}"

def generate_speech(text: str, voice: str, pace: str) -> bytes | None:
    prompt = build_prompt(text, pace)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            },
        },
    }
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(API_URL, headers={"Content-Type": "application/json"}, json=payload)
        if resp.status_code != 200:
            err = resp.json().get("error", {}).get("message", f"HTTP {resp.status_code}")
            tqdm.write(f"  ⚠️  API error: {err}")
            return None
        data = resp.json()
        inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
        mime = inline.get("mime_type", "audio/L16;rate=24000")
        return pcm_to_wav(base64.b64decode(inline["data"]), mime)
    except Exception as e:
        tqdm.write(f"  ❌ Exception: {e}")
        return None

def load_unique_twi_sentences(csv_path: str) -> list[str]:
    sentences = set()
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("twi_translation", "").strip()
            if text and text != "TRANSLATION_FAILED":
                sentences.add(text)
    return sorted(sentences)

def main():
    print(f"Loading sentences from {INPUT_CSV} ...")
    sentences = load_unique_twi_sentences(INPUT_CSV)
    if not sentences:
        print("No valid Twi sentences found.")
        return

    total_files = len(sentences) * SAMPLE_COUNT
    print(f"Found {len(sentences)} unique sentences → {total_files} files.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    failed = []
    pbar = tqdm(total=total_files, desc="Generating", unit="file")

    for idx, sentence in enumerate(sentences, 1):
        for sample_num in range(1, SAMPLE_COUNT + 1):
            wav_name = f"sentence_{idx:04d}_sample_{sample_num:02d}.wav"
            wav_path = os.path.join(OUTPUT_DIR, wav_name)

            if os.path.exists(wav_path):
                pbar.update(1)
                continue

            voice = random.choice(ALL_VOICES)
            pace  = random.choice(PACE_VARIANTS)
            wav_data = generate_speech(sentence, voice, pace)

            if wav_data is None:
                failed.append((idx, sample_num, sentence))
            else:
                with open(wav_path, "wb") as f:
                    f.write(wav_data)
                tqdm.write(f"  ✅ {voice:20s} → {wav_name}")

            time.sleep(DELAY_SECONDS)
            pbar.update(1)

    pbar.close()
    if failed:
        print(f"\n⚠️  Failed for {len(failed)} samples:")
    succeeded = total_files - len(failed)
    print(f"\n✅ Done – {succeeded} / {total_files} files in '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    main()
