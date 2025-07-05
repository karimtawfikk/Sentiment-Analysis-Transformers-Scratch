import streamlit as st
import torch
from sent_analysis_encoder import SentimentTransformer, vocab, predict_sentiment

embed_dim = 128
num_heads = 4
ff_dim = 256
num_layers = 5
num_classes = 2
max_len = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SentimentTransformer(
    vocab_size=len(vocab),
    max_len=max_len,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ff_dim=ff_dim,
    num_layers=num_layers,
    num_classes=num_classes
).to(device)

model.load_state_dict(torch.load("sentiment_transformer.pth", map_location=device))
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
