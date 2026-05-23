import subprocess
import os
import numpy as np
import librosa
import onnxruntime as ort
import openwakeword


class PiperSynthesizer:
    def __init__(self, piper_path=None, model_path=None, config_path=None):
        self.piper_path = piper_path or "piper"
        self.model_path = model_path
        self.config_path = config_path

    def synthesize(self, text, output_wav_path,
                   length_scale=1.0, noise_scale=0.0, noise_w=0.0):
        cmd = [
            self.piper_path,
            "--model", self.model_path,
            "--config", self.config_path,
            "--output_file", output_wav_path,
            "--length_scale", str(length_scale),
            "--noise_scale", str(noise_scale),
            "--noise_w", str(noise_w),
        ]
        try:
            result = subprocess.run(
                cmd, input=text, capture_output=True, text=True, timeout=15
            )
            if result.returncode == 0 and os.path.exists(output_wav_path):
                return True
        except subprocess.TimeoutExpired:
            pass
        return False


def load_embedding_model():
    """
    Load the speech embedding ONNX model bundled with openWakeWord.
    Returns (onnxruntime.InferenceSession, input_name).
    """
    model_path = os.path.join(
        os.path.dirname(openwakeword.__file__),
        "resources", "models", "embedding_model.onnx"
    )
    if not os.path.exists(model_path):
        openwakeword.utils.download_models()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Embedding model not found at {model_path}")

    session = ort.InferenceSession(model_path)

    # Print model input shape so we can confirm expectations
    inp = session.get_inputs()[0]
    print(f"Embedding model input — name: '{inp.name}', shape: {inp.shape}, type: {inp.type}")

    return session


def _compute_mel_spectrogram(audio, sr=16000, n_mels=32, n_fft=512,
                               hop_length=160, win_length=400):
    """
    Compute a log mel spectrogram matching openWakeWord's preprocessing:
      - 16 kHz mono audio
      - 32 mel bins
      - 25 ms windows, 10 ms hop
    Returns array of shape (time_frames, 32).
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        fmin=60.0,
        fmax=3800.0,
    )
    log_mel = librosa.power_to_db(mel, ref=1.0).T  # → (time_frames, 32)
    return log_mel.astype(np.float32)


def extract_embedding(session, wav_path, target_sr=16000,
                      frame_len=76, hop_frames=8):
    """
    Extract a speech embedding from a WAV file.

    The openWakeWord embedding model expects input of shape:
        (batch, frame_len, n_mels, 1)  →  e.g. (N, 76, 32, 1)

    We slide a window over the mel spectrogram and average the resulting
    per-frame embeddings into a single vector.

    Parameters:
        session    : onnxruntime InferenceSession from load_embedding_model()
        wav_path   : path to audio file
        target_sr  : sample rate (must be 16000)
        frame_len  : number of mel frames per window (76 ≈ 0.96 s)
        hop_frames : step between windows

    Returns:
        numpy array of shape (embedding_dim,), or None on error.
    """
    try:
        audio, _ = librosa.load(wav_path, sr=target_sr, mono=True)
        log_mel = _compute_mel_spectrogram(audio, sr=target_sr)  # (T, 32)

        T, n_mels = log_mel.shape
        input_name = session.get_inputs()[0].name

        # Pad if the clip is shorter than one window
        if T < frame_len:
            pad = np.zeros((frame_len - T, n_mels), dtype=np.float32)
            log_mel = np.concatenate([log_mel, pad], axis=0)
            T = frame_len

        # Slide window and collect embeddings
        embeddings = []
        for start in range(0, T - frame_len + 1, hop_frames):
            window = log_mel[start: start + frame_len]          # (76, 32)
            inp = window[np.newaxis, :, :, np.newaxis]          # (1, 76, 32, 1)
            out = session.run(None, {input_name: inp})          # [(1, emb_dim)]
            embeddings.append(out[0].flatten())

        if not embeddings:
            print(f"Warning: no windows extracted from {wav_path}")
            return None

        # Average across windows → single fixed-size embedding
        return np.mean(embeddings, axis=0).astype(np.float32)

    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return None
