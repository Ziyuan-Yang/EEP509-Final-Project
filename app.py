import streamlit as st
from openai import OpenAI
from agents_part3_local import Head_Agent

def load_key(path="key.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

st.title("Multi-Agent RAG System (Part3)")

api_key = load_key("key.txt")
client = OpenAI(api_key=api_key)
head = Head_Agent(client, "local_vectorstore.pkl")

if "history" not in st.session_state:
    st.session_state["history"] = ""

query = st.text_input("Ask a question:")

if st.button("Submit") and query.strip():
    answer, path = head(query, history=st.session_state["history"])

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Agent Path")
    st.write(" → ".join(path))

    st.session_state["history"] += f"User: {query}\nAssistant: {answer}\n"