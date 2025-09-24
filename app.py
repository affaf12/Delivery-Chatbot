import streamlit as st
import pandas as pd
from transformers import pipeline

# -------------------------
# Load dataset
# -------------------------
df = pd.read_csv("company_esg_financial_dataset.csv")  # replace with your dataset
data_text = df.to_string(index=False)

# -------------------------
# Load model (tiny demo model for now)
# -------------------------
generator = pipeline("text-generation", model="sshleifer/tiny-gpt2")

# -------------------------
# Streamlit Layout
# -------------------------
st.set_page_config(page_title="🚚 Delivery Data Chatbot", layout="wide")

st.title("🚚 Delivery Data Chatbot")
st.markdown("Ask questions about your delivery dataset and get clean, AI-powered answers.")

# -------------------------
# Layout: Sidebar + Main Panel
# -------------------------
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📝 Options")

    with st.expander("📊 Delivery Performance"):
        st.markdown("""
        - Which delivery person is the fastest on average?  
        - What is the average delivery time per city?  
        - How do multiple deliveries affect delivery time?  
        - Which vehicle type is most efficient for deliveries?  
        """)

    with st.expander("👥 Customer & Order Insights"):
        st.markdown("""
        - What types of orders take the longest to deliver?  
        - Does order time affect delivery speed?  
        - Are deliveries slower during festivals?  
        """)

    with st.expander("🌍 Environment & External Factors"):
        st.markdown("""
        - How does traffic density impact delivery time?  
        - Do weather conditions affect delivery speed?  
        - How do restaurant vs. delivery locations affect time?  
        """)

    with st.expander("🚴 Delivery Personnel Metrics"):
        st.markdown("""
        - Who has the highest ratings and fastest deliveries?  
        - Does the age of the delivery person affect speed?  
        - Does vehicle condition impact delivery time?  
        """)

    with st.expander("📍 Geospatial Insights"):
        st.markdown("""
        - Which areas have the highest delays?  
        - What is the correlation between distance and time?  
        """)

    st.markdown("---\n💡 Tip: Try typing your own custom questions!")

with col2:
    st.subheader("🤖 AI Response")
    answer_placeholder = st.empty()

    st.subheader("🔍 Enter your question")
    question = st.text_area(
        "",
        placeholder="e.g. Which delivery person has the highest rating?",
        height=80
    )

    if st.button("🚀 Ask Question"):
        if question.strip():
            prompt = f"Here is some delivery data:\n{data_text}\n\nQuestion: {question}\nAnswer:"
            result = generator(prompt, max_length=200)[0]['generated_text']
            clean_answer = result.split("Answer:")[-1].strip()
            answer_placeholder.text_area("🤖 AI Response", value=clean_answer, height=150)
        else:
            st.warning("⚠️ Please enter a question.")
