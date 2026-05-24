from google import genai
from google.genai import types
import csv
import time
import os

# ============================================================
# CONFIGURATION
# ============================================================
GEMINI_API_KEY = "AIzaSyA-PVd-HE7wwMUln6IdKvpkoPWf4hbkQyc"          # Replace or use environment variable
CSV_INPUT_PATH = "/home/owusus/Downloads/phrase_classifier/sample_phrases/symptoms_dataset.csv"
# Model to use – a Gemini Flash Lite preview with thinking
MODEL_NAME = "gemini-3.1-flash-lite"
# ============================================================

# Create the client (prefer environment variable for the key)
client = genai.Client(api_key=GEMINI_API_KEY)   # or os.environ.get("GEMINI_API_KEY")

PROMPT_TEMPLATE = """Give me up to 5 ways that each of these can be said in Twi. Give me a plain text list of sentences for all together in one plain text list. No headers or bullets.

{sentences}"""

BATCH_SIZE = 6   # sentences per API call


def load_symptoms_from_csv(csv_path: str) -> list[tuple[str, str, str]]:
    """
    Read the symptom dataset from a CSV file.
    Expected columns: body_part, sub_category, symptom_description
    Returns a list of tuples: (body_part, sub_category, description)
    """
    symptoms = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            body_part = row["body_part"].strip()
            sub_category = row["sub_category"].strip()
            description = row["symptom_description"].strip()
            symptoms.append((body_part, sub_category, description))
    return symptoms


def get_twi_translations(sentences: list[str]) -> list[list[str]]:
    quoted = "\n".join(f'"{s}"' for s in sentences)
    prompt = PROMPT_TEMPLATE.format(sentences=quoted)

    # Use the new client with high-thinking configuration
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1,                     # required when thinking is enabled
            thinking_config=types.ThinkingConfig(
                thinking_level="HIGH",         # "HIGH" for high thinking
                # Alternatively, use a budget: thinking_budget=8192,
            ),
        ),
    )

    raw = response.text.strip()

    # Split all lines, strip blanks
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

    # Group into chunks of up to 5 per source sentence
    result = []
    idx = 0
    for _ in sentences:
        chunk = []
        while idx < len(lines) and len(chunk) < 5:
            chunk.append(lines[idx])
            idx += 1
        result.append(chunk)

    return result


def main():
    # Load symptoms from CSV
    print(f"Loading symptoms from {CSV_INPUT_PATH} ...")
    symptoms = load_symptoms_from_csv(CSV_INPUT_PATH)
    total = len(symptoms)
    print(f"Loaded {total} symptom descriptions.")

    output_rows = []  # will hold dicts for final CSV

    # Process in batches
    for batch_start in range(0, total, BATCH_SIZE):
        batch = symptoms[batch_start: batch_start + BATCH_SIZE]
        sentences = [s[2] for s in batch]   # the description is the third element

        print(f"Processing {batch_start + 1}–{min(batch_start + BATCH_SIZE, total)} / {total} …")

        try:
            translations_per_sentence = get_twi_translations(sentences)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            translations_per_sentence = [[] for _ in sentences]

        for (body_part, sub_cat, original), twi_list in zip(batch, translations_per_sentence):
            for twi_sentence in twi_list:
                output_rows.append({
                    "body_part": body_part,
                    "sub_category": sub_cat,
                    "original_english": original,
                    "twi_translation": twi_sentence,
                })
            # If Gemini returned nothing, still write a placeholder row
            if not twi_list:
                output_rows.append({
                    "body_part": body_part,
                    "sub_category": sub_cat,
                    "original_english": original,
                    "twi_translation": "TRANSLATION_FAILED",
                })

        time.sleep(1)   # be polite to the API

    # Write output CSV
    output_path = "twi_symptom_translations.csv"
    fieldnames = ["body_part", "sub_category", "original_english", "twi_translation"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nDone! {len(output_rows)} rows written to '{output_path}'.")


if __name__ == "__main__":
    main()
