# Sentiment and Emotion Analysis Using Transformers

This project analyzes text in two ways:

- Sentiment classification: positive or negative
- Emotion classification: one of anger, disgust, fear, joy, neutral, sadness, or surprise

The app uses pretrained Hugging Face models and a Streamlit interface.

## Features

- Sentiment analysis with DistilBERT
- Emotion prediction with a DistilRoBERTa model
- Reusable model modules
- Evaluation with accuracy, precision, recall, F1-score, confusion matrix, and classification report
- Local Streamlit app for interactive text input

## Project structure

- `app.py` — Streamlit UI
- `sentiment_model.py` — sentiment analysis module
- `emotion_model.py` — emotion classification module
- `evaluation.py` — evaluation script
- `requirements.txt` — Python dependencies
- `.gitignore` — ignores local environment files

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the app

```powershell
streamlit run app.py
```

## Run evaluation

```powershell
python evaluation.py
```

## Notes

- The sentiment model is a binary classifier for positive vs. negative sentiment.
- The emotion model is trained for seven classes; evaluation datasets must match those exact labels.
- If a dataset does not match the model output labels, it must be remapped or replaced with a compatible dataset before metrics are calculated.
- This project uses local CSV evaluation files for reproducibility and compatibility with current Hugging Face library versions.

## Example

Input:

> I am thrilled to start this new project.

Output:

- Sentiment: POSITIVE
- Emotion: JOY

## Limitations

- The models are pretrained and may still misclassify sarcasm, slang, or subtle emotional language.
- Performance depends on the quality and compatibility of the evaluation dataset.
- Models do not understand all context equally well across domains.
