# Issue Query Bot
import streamlit as st

st.title("Monster Query Bot")

st.subheader("Example Query:\n Which Monsters found in sea?")

st.subheader("Example Query:\n What Monsters can eat metal?")

# Loading Finding Issue Text
# Embedding via API
# Embedding Prompt
# Cos Similarity
# 

import os
import math
import pickle

import numpy as np

from mistralai import Mistral

EMBEDDINGS_LENGTH = 281

CURRENT_PATH = os.getcwd()

EMBEDDINGS_FILE = r"C:\Projects\Project 031 PHBot\src\embeddings.pkl"
MM_FILE = r"C:\Projects\Project 031 PHBot\src\MM_parsed.txt"

EMBEDDINGS_FILE = os.path.join(CURRENT_PATH, "embeddings.pkl")
MM_FILE  = os.path.join(CURRENT_PATH, "MM_parsed.txt")

API_KEY = st.secrets["api_key"]
MODEL = "open-mistral-7b"

TOP_N_ANSWER = 1

def mistral_completion(text):
    # Return Text Completion by API call
    client = Mistral(api_key=API_KEY)
    messages = [{"role": "user", "content": text}]
    chat_response = client.chat.complete(model=MODEL,messages=messages)
    return chat_response.choices[0].message.content

def cos_sim(vector1, vector2):
    # Compute dot product of the two vectors
    dot_product = sum(i*j for i,j in zip(vector1, vector2))
    # Compute magnitude of each vector
    magnitude1 = math.sqrt(sum(i**2 for i in vector1))
    magnitude2 = math.sqrt(sum(i**2 for i in vector2))
    # Compute cosine similarity
    return(dot_product / (magnitude1 * magnitude2))


def top_n_similar(n, vectors, one_vector):
    cos_scores = np.apply_along_axis(cos_sim, axis=1, arr=vectors, vector2=one_vector)
    return np.array(cos_scores).argsort()[-n:][::-1]

# Loading Embedding Files
with open(EMBEDDINGS_FILE, 'rb') as f:
    finding_embeddings = pickle.load(f)

# Loading Monster File
with open(MM_FILE, "r", encoding="utf-8") as file:
    lines = [line for line in file]

# Load Mistral Client
client = Mistral(api_key=API_KEY)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# create a clear button in the sidebar
if st.sidebar.button('Clear'):
    # clear the placeholder
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    embeddings_batch_response = client.embeddings.create(model="mistral-embed",inputs=prompt)
    prompt_embedding = embeddings_batch_response.data[0].embedding
    top_n_idx = top_n_similar(TOP_N_ANSWER, finding_embeddings, prompt_embedding)
    for i in range(TOP_N_ANSWER):
        retrieval = lines[top_n_idx[i]]
        messages = f"{prompt} based on the following text {retrieval}"
        completion = mistral_completion(messages)
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(completion)
    # Add assistant response to chat history
    # st.session_state.messages.append({"role": "assistant", "content": response})