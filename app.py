import streamlit as st
from src.rag_core import rag_system

st.set_page_config(page_title="NCERT Physics Helper", layout="wide")

st.title("⚡ NCERT Physics Helper")
st.write("I can answer questions on electrostatics. Fire away!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_system.get_answer(prompt)
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
