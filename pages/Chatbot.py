# API Call Bot
# Adding Memory
import streamlit as st

st.title("Chatbot")

from mistralai import Mistral

# API_KEY = st.secrets["api_key"]
MODEL = "open-mistral-7b"
API_KEY = "a3NT4OluyAY7SVN9VLVGR9K2UFTU0akj"

def mistral_completion(text):
    # Return Text Completion by API call
    client = Mistral(api_key=API_KEY)
    messages = [{"role": "user", "content": text}]
    chat_response = client.chat.complete(model=MODEL,messages=messages)
    return chat_response.choices[0].message.content


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
    response = mistral_completion(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})