# DND PHB Demo
# FrontEnd Demo

# 2 Features
# Echo Bot
# User Query PHB

# TODO
# Chunking PHB, Embedded as Pickle with SBERT
# Using SBERT to Embed User Query and Give Similar Answer


import streamlit as st

# https://llm-examples.streamlit.app/Langchain_Quickstart
# https://github.com/streamlit/llm-examples
# https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    st.write("# Welcome to D&D Demo 👋")

    # create a clear button in the sidebar
    if st.sidebar.button('Clear'):
        # clear the placeholder
        st.session_state.messages = []

    st.markdown(
        """
        DnD Platform built specifically for Dnd Players
        **👈 Select a demo from the sidebar** to see some examples
    """
    )


if __name__ == "__main__":
    run()
