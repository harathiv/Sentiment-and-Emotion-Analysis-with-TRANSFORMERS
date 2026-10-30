from transformers import pipeline

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

print("Loading sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
)

print("Loading emotion model...")
emotion_pipeline = pipeline(
    "text-classification",
    model=EMOTION_MODEL,
)

print("Both model identifiers are valid and loaded successfully.")

text = "I am excited to start this project!"

print("\nSentiment result:")
print(sentiment_pipeline(text))

print("\nEmotion result:")
print(emotion_pipeline(text))