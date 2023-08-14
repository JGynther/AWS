import streamlit as st
from helpers import StreamHandler


st.title("SageMaker LLM Demo")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        container = st.empty()
        handler = StreamHandler(container)
        full_text = handler.stream_iterate_tokens(prompt)
        del handler

    st.session_state.messages.append({"role": "assistant", "content": full_text})
