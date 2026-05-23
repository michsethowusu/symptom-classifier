from classifier import PhraseClassifier

# Load your trained model
clf = PhraseClassifier("classifier.pt")

# Predict from an audio file
result = clf.predict("test_hello.wav")
print(f"Heard: {result}")