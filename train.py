#!/usr/bin/env python3
"""
Train a tiny phrase classifier from synthetic Piper TTS data.
Usage: python train.py --phrases "hello,stop" --lang en --voice en_US-lessac-medium
"""
import argparse
import os
import json
import random
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.data_generation import PiperSynthesizer, extract_embedding, load_embedding_model

class SimpleMLP(nn.Module):
    def __init__(self, num_classes, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    def forward(self, x):
        return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrases", required=True, help="Comma-separated list or path to .txt file")
    parser.add_argument("--lang", required=True, help="Language folder under voices/ (e.g., en)")
    parser.add_argument("--voice", required=True, help="Voice name (no extension, e.g., en_US-lessac-medium)")
    parser.add_argument("--samples", type=int, default=15, help="Samples per phrase (default: 15)")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (default: 25)")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Classifier hidden size (default: 128)")
    parser.add_argument("--output", default="classifier.pt", help="Output model file")
    parser.add_argument("--piper-path", default=None, help="Path to piper executable (optional)")
    args = parser.parse_args()

    # Parse phrases
    if os.path.isfile(args.phrases):
        with open(args.phrases, "r") as f:
            phrases = [line.strip() for line in f if line.strip()]
    else:
        phrases = [p.strip() for p in args.phrases.split(",") if p.strip()]

    if len(phrases) < 2:
        print("Need at least 2 phrases.")
        return

    print(f"Phrases: {phrases}")
    print(f"Generating {args.samples} samples per phrase with Piper...")

    # Paths
    voice_dir = Path("voices") / args.lang
    model_path = voice_dir / f"{args.voice}.onnx"
    config_path = voice_dir / f"{args.voice}.onnx.json"

    if not model_path.exists() or not config_path.exists():
        print(f"Voice not found: {model_path} or {config_path}")
        print("Place Piper voice files in the voices/ directory tree.")
        return

    # Initialise Piper synthesizer
    piper = PiperSynthesizer(
        piper_path=args.piper_path,
        model_path=str(model_path),
        config_path=str(config_path)
    )

    # Load embedding model
    print("Loading speech embedding model...")
    embed_model = load_embedding_model()

    # Generate data and extract embeddings
    X, y = [], []
    label_map = {phrase: i for i, phrase in enumerate(phrases)}
    total = len(phrases) * args.samples

    with tempfile.TemporaryDirectory() as tmpdir:
        for phrase, label in label_map.items():
            for i in range(args.samples):
                # Random augmentation parameters
                length_scale = random.uniform(0.9, 1.1)
                noise_scale  = random.uniform(0.0, 0.05)
                noise_w      = random.uniform(0.0, 0.05)

                out_path = os.path.join(tmpdir, f"{label}_{i}.wav")
                success = piper.synthesize(phrase, out_path,
                                           length_scale=length_scale,
                                           noise_scale=noise_scale,
                                           noise_w=noise_w)
                if not success:
                    continue
                emb = extract_embedding(embed_model, out_path)
                if emb is not None:
                    X.append(emb)
                    y.append(label)

    if len(X) < 10:
        print("Not enough usable samples generated. Check Piper voice and audio format.")
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Simple train/val split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.9 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Train classifier
    num_classes = len(phrases)
    model = SimpleMLP(num_classes, hidden_dim=args.hidden_dim)
    device = torch.device("cpu")
    model.to(device)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(X_train)} samples, validating on {len(X_val)}...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.tensor(X_val))
            val_pred = val_logits.argmax(dim=1)
            acc = (val_pred == torch.tensor(y_val)).float().mean().item()
        print(f"Epoch {epoch+1:2d}/{args.epochs}  loss={total_loss/len(train_loader):.4f}  val_acc={acc:.3f}")

    # Save model + label map
    save_dict = {
        "state_dict": model.state_dict(),
        "labels": {str(v): k for k, v in label_map.items()},
        "hidden_dim": args.hidden_dim,
    }
    torch.save(save_dict, args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()
