import torch
import torch.nn as nn
import numpy as np
import argparse
from utils.data_generation import extract_embedding, load_embedding_model

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

class PhraseClassifier:
    def __init__(self, model_path):
        """
        Load a trained classifier.
        model_path: path to the .pt file saved by train.py
        """
        checkpoint = torch.load(model_path, map_location="cpu")
        self.labels = checkpoint["labels"]  # dict: index (str) -> phrase
        self.hidden_dim = checkpoint.get("hidden_dim", 128)
        self.num_classes = len(self.labels)

        # Build model
        self.model = SimpleMLP(self.num_classes, hidden_dim=self.hidden_dim)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        # Load embedding model once
        self.embed_model = load_embedding_model()

    def predict(self, wav_path):
        """
        Return the predicted phrase (str) from a WAV file.
        """
        emb = extract_embedding(self.embed_model, wav_path)
        if emb is None:
            return None
        with torch.no_grad():
            inp = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            logits = self.model(inp)
            pred_idx = logits.argmax(dim=1).item()
        return self.labels.get(str(pred_idx), "unknown")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to trained .pt file")
    parser.add_argument("--wav", required=True, help="WAV file to classify")
    args = parser.parse_args()

    clf = PhraseClassifier(args.model)
    phrase = clf.predict(args.wav)
    print(f"Predicted phrase: {phrase}")