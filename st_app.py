import os
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import StreamlitCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler

from mdagent import MDAgent

load_dotenv()


st_callback = StreamlitCallbackHandler(st.container())


# Streamlit app
st.title("MDAgent")

# for now I'm just going to allow  pdb and cif files - we can add more later
uploaded_files = st.file_uploader(
    "Upload a .pdb or .cif file", type=["pdb", "cif"], accept_multiple_files=True
)
files: List[str] = []
# write file to disk
if uploaded_files:
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())

    st.write("Files successfully uploaded!")
    uploaded_file = [os.path.join(os.getcwd(), file.name) for file in uploaded_files]
else:
    uploaded_file = []

mdagent = MDAgent(uploaded_files=uploaded_file)


def generate_response(prompt):
    result = mdagent.run(prompt)
    return result


# make new container to store scratch
scratch = st.empty()
scratch.write(
    """Hi! I am MDAgent, your MD automation assistant.
              How can I help you today?"""
)


# This allows streaming of llm tokens
class TokenStreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container

    def on_llm_new_token(self, token, **kwargs):
        self.container.write("".join(token))


token_st_callback = TokenStreamlitCallbackHandler(scratch)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = mdagent.run(prompt, callbacks=[st_callback, token_st_callback])
        st.write(response)
