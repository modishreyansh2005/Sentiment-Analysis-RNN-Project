import streamlit as st
import pickle
import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import pad_sequences
import tensorflow as tf

load_model = tf.keras.models.load_model
pad_sequences = tf.keras.utils.pad_sequences


# Load model
model = load_model("sentiment_model.h5", compile=False)

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 50

# UI
st.title("😀 Sentiment Analysis App")
st.write("Enter text to analyze sentiment")

text = st.text_area("Your Text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=max_len)

        pred = model.predict(padded)
        label = np.argmax(pred, axis=1)[0]

        sentiments = ["Negative 😞", "Neutral 😐", "Positive 😀"]

        st.subheader("Result:")
        st.success(sentiments[label])