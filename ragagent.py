import os
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 1. System Setup
load_dotenv()

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


API_KEY = os.getenv("PINECONE_API_KEY")

if not API_KEY:
    st.error("PINECONE_API_KEY not found in .env")
    st.stop()

pc = Pinecone(api_key=API_KEY)

# 2. Sidebar Gradient
st.markdown("""
<style>
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, #0F172A, #1E3A8A);
}
section[data-testid="stSidebar"] *{
    color: white;
}
</style>
""", unsafe_allow_html=True)

# 3. Sidebar Navigation
with st.sidebar:

    st.markdown("""

    ### 📚 Machine Learning Guide

    Welcome! 👋

    Select a lecture below and ask anything from your uploaded ML book.

    ---
    """)

    topic = st.radio(
        "📖 Choose a Lecture",
        [
            "🟢 L01 • Introduction",
            "🔵 L02 • Supervised Learning",
            "🌳 L03 • Decision Trees",
            "⚖️ L04 • Trade-offs",
            "🚫 L05 • Overfitting"
        ]
    )
    

# 4. Main Operational View
st.title("📚 ML Book RAG Assistant")
st.info(f"📖 Current Lecture : **{topic}**")
query = st.text_input(
    "🔍 Ask your question",
    placeholder="Example: What is Decision Tree?"
)


# 5. Action Request Processing
if st.button("🚀 Ask RAG Assistant", type="primary") and query:
    with st.spinner("Thinking..."):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150
        )

        chunks = splitter.split_text(
            f"Topic Scope: {topic}. Question: {query}"
        )

        final_prompt = " ".join(chunks)

        try:
            assistant = pc.assistant("ragagent")

            response = assistant.chat(
                messages=[
                    {
                        "role": "user",
                        "content": final_prompt
                    }
                ]
            )

            st.markdown("### 🤖 Answer")
            st.write(response.message.content)

        except Exception as e:
            st.error(str(e))