import streamlit as st
import torch
from sent_analysis_encoder import vocab, predict_sentiment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("sentiment_transformer_full.pth", map_location=device)
model.eval()

st.title("🎭 Sentiment Analysis Transformer")
st.write("Enter your review below, and the model will predict its sentiment.")

text_input = st.text_area("Review:", "The movie was surprisingly good!")

if st.button("Predict"):
    if text_input.strip():
        prediction = predict_sentiment(text_input, model, vocab)
        st.success(f"Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text.")
