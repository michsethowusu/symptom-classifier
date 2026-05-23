# Phrase Classifier – Tiny Offline Command Recognition

Train a custom multi‑phrase classifier (wake‑word / command) using synthetic speech from Piper TTS.
Works offline, runs on CPU, and produces models <50 KB.

## How it works

1. Piper TTS generates a few hundred short audio clips of your phrases with random voice variations.
2. The frozen, pre‑trained speech embedding model from [openWakeWord](https://github.com/dscripka/openWakeWord) converts each clip to a fixed 96‑dimensional vector.
3. A tiny MLP classifier (96 → 256 → N_phrases) is trained on those vectors.
4. The classifier + label map are saved for offline inference.

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/michsethowusu/phrase-classifier.git
cd phrase-classifier
```



### 2. Install Python dependencies

bash

```
pip install -r requirements.txt
```



### 3. Get Piper TTS

You need the Piper executable and at least one voice model.

**Option A: Install via pip (Linux / macOS)**

bash

```
pip install piper-tts
```



Then download a voice (e.g., `en_US-lessac-medium`) from the [Piper releases](https://github.com/rhasspy/piper/releases) and place the `.onnx` + `.json` files into:

text

```
voices/<language>/<voice_name>.onnx
voices/<language>/<voice_name>.onnx.json
```



**Option B: Use the standalone binary**
Download the binary for your OS from [Piper GitHub](https://github.com/rhasspy/piper) and put it somewhere in your PATH, or set the `PIPER_PATH` environment variable.

**Voice directory example**:

text

```
voices/
├── en/
│   ├── en_US-lessac-medium.onnx
│   └── en_US-lessac-medium.onnx.json
└── twi/
    ├── twi_GH-akua-medium.onnx
    └── twi_GH-akua-medium.onnx.json
```



> The training script expects voices in `voices/<language>/`.
> You can also specify a custom voice path with `--voice-path`.

## Training

bash

```
python3 train.py \
  --phrases sample_phrases/twi-test.txt \
  --lang twi \
  --voice twi_GH-kofi-medium \
  --piper-path ./piper/piper \
  --samples 30 \
  --epochs 40 \
  --hidden-dim 256
```
OR


```
python3 train.py --phrases sample_phrases/twi-test.txt --lang twi --voice twi_GH-kofi-medium --piper-path ./piper/piper
```


**Arguments**:

- `--phrases` : comma‑separated list of phrases, or path to a `.txt` file (one phrase per line).
- `--lang` : language subfolder inside `voices/` (e.g., `en`).
- `--voice` : name of the voice model without extension (e.g., `en_US-lessac-medium`).
- `--samples` : number of synthetic samples per phrase (default: 15).
- `--output` : output model filename (default: `classifier.pt`).
- `--hidden-dim` : classifier hidden layer size (default: 128).

## Inference

python

```
from classifier import PhraseClassifier

model = PhraseClassifier("classifier.pt")
label = model.predict("recording.wav")
print(label)  # "stop"
```



Or use the built‑in `classifier.py` CLI:

bash

```
python classifier.py --model classifier.pt --wav test.wav
```



## How to add a new language

1. Download a Piper voice for that language.
2. Place the `.onnx` and `.json` files in `voices/<language>/`.
3. Train with `--lang <language> --voice <voice_name>`.

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) – speech embedding model
- [Piper TTS](https://github.com/rhasspy/piper) – high‑quality text‑to‑speech
