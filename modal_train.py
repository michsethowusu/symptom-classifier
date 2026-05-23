import modal

stub = modal.Stub("train-twi-classifier")
image = (
    modal.Image.debian_slim()
    .pip_install(
        "datasets",
        "librosa",
        "soundfile",
        "torch",
        "tqdm",
        "openwakeword",
        "onnxruntime",
    )
    .pip_install("librosa")  # ensure latest
)

volume = modal.SharedVolume().persist("models")

@stub.function(image=image, cpu=8, shared_volumes={"/models": volume}, timeout=3600*4)
def train():
    import subprocess
    # Copy local code to /root (or mount)
    subprocess.run(["cp", "-r", "/phrase_classifier", "/root/"], check=True)
    os.chdir("/root/phrase_classifier")
    subprocess.run([
        "python", "train_real.py",
        "--max-classes", "1000",
        "--epochs", "40",
        "--hidden-dim", "1024",
        "--num-layers", "2",
        "--output", "/models/twi_1000words.pt"
    ], check=True)
    print("Training done! Model saved to shared volume /models/")

if __name__ == "__main__":
    with stub.run():
        train()