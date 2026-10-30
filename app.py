import pandas as pd
import streamlit as st

from emotion_model import analyze_emotion
from sentiment_model import analyze_sentiment

st.set_page_config(page_title="Sentiment and Emotion Analysis", layout="wide")

st.markdown(
    """
    <style>
        :root {
            --bg: #0f766e;
            --bg-deep: #0b4f5a;
            --panel: rgba(15, 118, 110, 0.42);
            --panel-strong: rgba(9, 76, 83, 0.7);
            --mint: #a7f3d0;
            --mint-strong: #6ee7b7;
            --white: #ffffff;
            --text-soft: #d1fae5;
            --border: rgba(167, 243, 208, 0.35);
        }

        .stApp {
            background: linear-gradient(135deg, var(--bg) 0%, var(--bg-deep) 100%);
            color: var(--white);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, h4 {
            color: var(--mint) !important;
        }

        p, label, .stMetricLabel, .stMetricValue, .stDataFrame, .stTextArea textarea {
            color: var(--white) !important;
        }

        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--border);
            color: var(--white);
        }

        .stTextArea textarea::placeholder,
        .stTextInput input::placeholder {
            color: var(--text-soft) !important;
        }

        .stButton > button {
            background: linear-gradient(135deg, var(--mint-strong), #2dd4bf);
            color: #083344;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 700;
            box-shadow: 0 6px 16px rgba(16, 185, 129, 0.25);
        }

        .stButton > button:hover {
            background: linear-gradient(135deg, #86efac, var(--mint));
            color: #062b3b;
        }

        div[data-testid="metric-container"] {
            background: rgba(255,255,255,0.08);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 16px rgba(6, 78, 93, 0.18);
        }

        .stDataFrame, .stExpander {
            background: rgba(255,255,255,0.06);
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        .stAppHeader, .stHeader {
            background: transparent;
        }

        .css-1d391kg, .css-1v0mbdj {
            color: var(--white) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sentiment and Emotion Analysis")
st.write("Analyze a sentence to see whether it is positive or negative and which emotion is strongest.")

text = st.text_area(
    "Enter some text",
    height=180,
    placeholder="Type a sentence here...",
)

if st.button("Analyze"):
    if not text or not text.strip():
        st.error("Please enter some text before analyzing.")
    else:
        sentiment = analyze_sentiment(text)
        emotion = analyze_emotion(text, top_k=5)

        st.subheader("Results")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentiment", sentiment["label"], f"{sentiment['score']:.2%}")

        with col2:
            st.metric("Emotion", emotion["label"], f"{emotion['score']:.2%}")

        st.write(f"Original text: {text}")

        emotion_scores = pd.DataFrame(emotion["top_emotions"])
        emotion_scores["label"] = emotion_scores["label"].str.upper()
        st.bar_chart(emotion_scores.set_index("label")["score"])

        with st.expander("Full emotion probability distribution"):
            st.dataframe(emotion["all_scores"], use_container_width=True)
