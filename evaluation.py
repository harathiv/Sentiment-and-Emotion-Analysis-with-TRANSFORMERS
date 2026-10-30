import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from transformers import pipeline

from emotion_model import EMOTION_MODEL
from sentiment_model import SENTIMENT_MODEL


def evaluate_sentiment_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    valid_labels = ["NEGATIVE", "POSITIVE"]
    labels_in_data = sorted({str(label).strip().upper() for label in df["label"]})
    invalid_labels = sorted(set(labels_in_data) - set(valid_labels))

    if invalid_labels:
        raise ValueError(
            "Sentiment dataset labels must match the model's binary output. "
            f"Unsupported labels: {invalid_labels}. Use labels from: {valid_labels}."
        )

    sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        actual_label = str(row["label"]).strip().upper()
        predicted_label = sentiment_pipe(text)[0]["label"].upper()

        y_true.append(actual_label)
        y_pred.append(predicted_label)

    return {
        "labels": valid_labels,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def evaluate_emotion_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    valid_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    labels_in_data = sorted({str(label).strip().lower() for label in df["label"]})
    invalid_labels = sorted(set(labels_in_data) - set(valid_labels))

    if invalid_labels:
        raise ValueError(
            "Emotion dataset labels do not match the model's expected seven classes. "
            f"Unsupported labels: {invalid_labels}. Use labels from: {valid_labels}."
        )

    emotion_pipe = pipeline("text-classification", model=EMOTION_MODEL, top_k=None)
    y_true = []
    y_pred = []

    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        actual_label = str(row["label"]).strip().lower()
        predicted = emotion_pipe(text)[0]
        predicted_label = max(predicted, key=lambda entry: entry["score"])["label"].lower()

        y_true.append(actual_label)
        y_pred.append(predicted_label)

    return {
        "labels": valid_labels,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def print_metrics(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    sentiment_csv = "sentiment_eval_sample.csv"
    if os.path.exists(sentiment_csv):
        print("Evaluating sentiment model on local CSV...")
        sentiment_result = evaluate_sentiment_from_csv(sentiment_csv)
        print_metrics(sentiment_result["y_true"], sentiment_result["y_pred"], sentiment_result["labels"])
    else:
        print(
            "\nNo compatible sentiment CSV found. "
            "Create sentiment_eval_sample.csv with columns: text,label "
            "and labels from: NEGATIVE, POSITIVE."
        )

    emotion_csv = "emotion_eval_sample.csv"
    if os.path.exists(emotion_csv):
        print("\nEvaluating emotion model on compatible CSV...")
        emotion_result = evaluate_emotion_from_csv(emotion_csv)
        print_metrics(emotion_result["y_true"], emotion_result["y_pred"], emotion_result["labels"])
    else:
        print(
            "\nNo compatible emotion CSV found. "
            "Create emotion_eval_sample.csv with columns: text,label "
            "and labels from: anger, disgust, fear, joy, neutral, sadness, surprise."
        )
