from emotion_model import analyze_emotion

texts = [
    "I am so excited about the weekend!",
    "I feel terrified of losing my job.",
    "This was the best day of my life.",
    "I am very angry and disappointed.",
]

for text in texts:
    result = analyze_emotion(text, top_k=3)
    print(f"\nText: {text}")
    print(f"Emotion: {result['label']}")
    print(f"Confidence: {result['score']:.2%}")
    print("Top 3 emotions:")
    for entry in result["top_emotions"]:
        print(f"  - {entry['label']}: {entry['score']:.2%}")
