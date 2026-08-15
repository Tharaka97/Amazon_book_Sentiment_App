import os
import re
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
MODEL_PATH = "best_cnn_book_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
MAX_LEN = 250


# ---------------------------------------------------------
# Text preprocessing
# IMPORTANT: Keep this identical to the cleaning function
# used when the model was trained.
# ---------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------
# Load trained model and tokenizer once
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place it in the same folder as app.py."
        )

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer file not found: {TOKENIZER_PATH}. "
            "Place it in the same folder as app.py."
        )

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict_sentiment(review, model, tokenizer):
    cleaned_review = clean_text(review)

    if not cleaned_review:
        return None

    sequence = tokenizer.texts_to_sequences([cleaned_review])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    probability_positive = float(
        model.predict(padded_sequence, verbose=0)[0][0]
    )

    if probability_positive >= 0.5:
        sentiment = "Positive Review"
        confidence = probability_positive
    else:
        sentiment = "Negative Review"
        confidence = 1.0 - probability_positive

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_probability": probability_positive,
        "cleaned_review": cleaned_review
    }


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------
st.set_page_config(
    page_title="Amazon Book Review Sentiment Analyzer",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Amazon Book Review Sentiment Analyzer")

st.write(
    "Enter a book review below. The trained CNN model will classify "
    "the review as positive or negative."
)

try:
    model, tokenizer = load_resources()
except Exception as error:
    st.error(f"Unable to load the model resources: {error}")
    st.stop()

review_text = st.text_area(
    "Enter a book review",
    height=180,
    placeholder=(
        "Example: This book was beautifully written and I enjoyed "
        "every chapter."
    )
)

analyze = st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True
)

if analyze:
    if not review_text.strip():
        st.warning("Please enter a book review before analysing.")
    else:
        result = predict_sentiment(
            review_text,
            model,
            tokenizer
        )

        if result is None:
            st.warning(
                "The review did not contain enough valid text to analyse."
            )
        else:
            if result["sentiment"] == "Positive Review":
                st.success(f"Prediction: {result['sentiment']}")
            else:
                st.error(f"Prediction: {result['sentiment']}")

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            with st.expander("Prediction details"):
                st.write(
                    f"Positive probability: "
                    f"{result['positive_probability'] * 100:.2f}%"
                )
                st.write(
                    f"Negative probability: "
                    f"{(1 - result['positive_probability']) * 100:.2f}%"
                )
                st.caption(
                    "Classification threshold: 0.50 "
                    "(≥ 0.50 = Positive, < 0.50 = Negative)"
                )

st.divider()

st.caption(
    "ISY503 Intelligent Systems – Amazon Book Review "
    "Sentiment Analysis Project"
)

st.info(
    "Model note: This system performs binary sentiment classification. "
    "Mixed opinions, sarcasm, unusual wording and reviews that differ "
    "from the training data may be classified incorrectly."
)
