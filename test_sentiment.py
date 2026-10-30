from sentiment_model import analyze_sentiment
test_texts = [
    "i love the way she irritates me.",
    "The service was disappointing.",
]
for text in test_texts:
    result = analyze_sentiment(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']}")
    print(f"Confidence: {result['score']:.2%}")