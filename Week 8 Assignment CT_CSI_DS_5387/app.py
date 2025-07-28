import streamlit as st
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="Loan Approval RAG Chatbot")
st.title("🤖 Loan Approval RAG Q&A Chatbot")

pipeline = RAGPipeline(data_path="data/Training Dataset.csv")

query = st.text_input("Ask your loan-related question:")

if query:
    with st.spinner("Thinking..."):
        answer = pipeline.generate_answer(query)
        st.markdown("### Answer")
        st.write(answer)
