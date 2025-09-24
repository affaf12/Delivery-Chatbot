import streamlit as st
import pandas as pd
from transformers import pipeline

# -------------------------
# Load your dataset
# -------------------------
df = pd.read_csv("Zomato Dataset.csv")  # replace with your delivery dataset file
data_text = df.to_string(index=False)

# -------------------------
# Load model (small demo model)
# -------------------------
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")

st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get AI-powered answers.")

# Answer output box (top)
answer_placeholder = st.empty()

# Question input box
question = st.text_area("🔍 Enter your question:", placeholder="e.g. Which delivery person has the highest rating?")

# Submit button
if st.button("🚀 Ask Question"):
    if question.strip():
        prompt = f"Here is some delivery data:\n{data_text}\n\nQuestion: {question}\nAnswer:"
        result = generator(prompt, max_length=200)[0]['generated_text']
        clean_answer = result.split("Answer:")[-1].strip()
        answer_placeholder.text_area("🤖 AI Response", value=clean_answer, height=150)
    else:
        st.warning("⚠️ Please enter a question.")
