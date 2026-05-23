#!/usr/bin/env python3
"""
Train a phrase classifier on a real speech dataset (e.g., Twi words).
Designed to run on Modal, but also works locally if you have enough RAM.
Usage (local): python train_real.py --max-classes 500
Modal: see bottom of file for example run.
"""

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset, Audio

# Add repo root for utility imports
sys.path.append(str(Path(__file__).parent))
from utils.data_generation import load_embedding_model, extract_embedding


class SimpleMLP(nn.Module):
    def __init__(self, num_classes, hidden_dim=512, dropout=0.2, num_layers=1):
        super().__init__()
        layers = []
        input_dim = 96
        for i in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def process_batch(batch, embed_session):
    """Extract embeddings for a batch of audio samples."""
    embeddings = []
    for audio_array, sr in zip(batch["audio"]["array"], batch["audio"]["sampling_rate"]):
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
        else:
            audio = audio_array.astype(np.float32)
        # Write to temp file (extract_embedding expects a path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            emb = extract_embedding(embed_session, f.name)
            os.unlink(f.name)
        if emb is None:
            emb = np.zeros(96, dtype=np.float32)  # fallback (won't be used if filtered)
        embeddings.append(emb)
    return {"embedding": embeddings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="michsethowusu/twi-words-speech-text-parallel-400k")
    parser.add_argument("--max-classes", type=int, default=500,
                        help="Number of most frequent words/phrases to keep")
    parser.add_argument("--min-samples", type=int, default=5,
                        help="Ignore classes with fewer samples")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--output", default="classifier_real.pt")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 1. Load dataset (streaming if possible to save memory)
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, split="train", streaming=False)  # full in memory
    # Keep only necessary columns
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # 2. Determine which words/phrases to include
    texts = dataset["text"]
    counter = Counter(texts)
    filtered = [(word, cnt) for word, cnt in counter.items() if cnt >= args.min_samples]
    filtered.sort(key=lambda x: x[1], reverse=True)
    if args.max_classes and args.max_classes < len(filtered):
        filtered = filtered[: args.max_classes]
    selected_words = [word for word, _ in filtered]
    label_map = {word: i for i, word in enumerate(selected_words)}
    print(f"Classes: {len(label_map)}")
    print(f"Total samples: {sum(cnt for _, cnt in filtered)}")

    # 3. Filter dataset to selected words
    keep_mask = [t in label_map for t in texts]
    dataset = dataset.select(np.where(keep_mask)[0])
    print(f"Filtered dataset size: {len(dataset)}")

    # 4. Extract embeddings (use dataset.map with multiprocessing)
    print("Loading embedding model...")
    embed_session = load_embedding_model()

    # We need to pass embed_session to the map function; use partial
    from functools import partial
    map_fn = partial(process_batch, embed_session=embed_session)
    # Use multiprocessing for speed (Modal supports many CPUs)
    dataset = dataset.map(
        map_fn,
        batched=True,
        batch_size=32,
        remove_columns=["audio", "text"],
        num_proc=os.cpu_count() // 2,  # adjust as needed
        desc="Extracting embeddings"
    )

    # 5. Build training arrays
    embeddings = np.array(dataset["embedding"], dtype=np.float32)
    labels = np.array([label_map[t] for t in dataset["text"]], dtype=np.int64)
    # Remove any zero-vector fallbacks (very rare)
    valid = ~np.all(np.isclose(embeddings, 0), axis=1)
    embeddings = embeddings[valid]
    labels = labels[valid]
    print(f"Valid embeddings: {len(embeddings)}")

    # 6. Train / val split
    indices = np.arange(len(embeddings))
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    X_train, y_train = embeddings[train_idx], labels[train_idx]
    X_val, y_val = embeddings[val_idx], labels[val_idx]

    # 7. Train classifier
    num_classes = len(label_map)
    model = SimpleMLP(num_classes, hidden_dim=args.hidden_dim,
                      num_layers=args.num_layers)
    device = torch.device(args.device)
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=256, shuffle=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for bx, by in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validate
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val).to(device))
            val_pred = val_logits.argmax(dim=1)
            acc = (val_pred == torch.tensor(y_val).to(device)).float().mean().item()
        print(f"Epoch {epoch+1}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.4f}")

    # 8. Save model + label map
    save_dict = {
        "state_dict": model.state_dict(),
        "labels": {str(i): w for w, i in label_map.items()},
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }
    torch.save(save_dict, args.output)
    print(f"Model saved to {args.output}")


# ── Modal entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()