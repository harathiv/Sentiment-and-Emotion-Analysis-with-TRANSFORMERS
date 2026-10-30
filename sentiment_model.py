from transformers import pipeline

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

sentiment_classifier = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
)


def analyze_sentiment(text):
    """Return the sentiment label and confidence score for valid text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    result = sentiment_classifier(text)[0]

    return {
        "label": result["label"],
        "score": result["score"],
    }