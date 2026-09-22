"""
Gym FAQ Web Application Frontend.
Wraps the high-performing Hybrid RRF retrieval engine in a browser-based UI.

Usage:
    streamlit run app.py
"""
import os
import streamlit as st
# hybrid backend class from chatbot.py
from chatbot import HybridFaqChatbot

# Set up browser window tab branding configurations
st.set_page_config(page_title="Gym FAQ AI Assistant", page_icon="🏋️‍♂️", layout="centered")

@st.cache_resource
def load_chatbot_backend():
    """
    Instantiates the search index mapping layers and Qwen model exactly once.
    Streamlit caches this instance so page refreshes do not trigger slow CPU reloads.
    """
    return HybridFaqChatbot()


# Initialize the cached engine
try:
    bot = load_chatbot_backend()
except Exception as e:
    st.error(f"Failed to initialize search indexes: {e}")
    st.stop()

# Render visual header layout elements
st.title("🏋️‍♂️ Gym FAQ Assistant")
st.caption("Grounded strictly in verified WHO, NHS, CDC, and USADA health frameworks.")

# Initialize the session state tracking dictionary array if it does not exist yet
if "messages" not in st.session_state:
    st.session_state.messages = []

# Continuously loop through and draw previous chat histories on screen refreshes
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If sources were logged for this specific exchange, render them as clean expandable blocks
        if "evidence" in message and message["evidence"]:
            with st.expander("📚 View Verified Source Context"):
                for i, (docid, p) in enumerate(message["evidence"], 1):
                    st.markdown(f"**[{i}] {docid}** *({p.source})*")
                    st.caption(f"\"{p.passage}\"")

# Render active user chat interaction text block input box
if user_query := st.chat_input("Ask a question (e.g., Why do my knees crack when I squat?)"):
    
    # 1. Instantly display user input message layout on screen
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Process query via the hybrid RRF engine blocks and track timeline execution
    with st.chat_message("assistant"):
        with st.spinner("Analyzing verified sources and generating answer..."):
            answer, evidence = bot.answer(user_query)
            
            # Draw the final generated plain-language response text block
            st.markdown(answer)
            
            # If valid passages were successfully retrieved, list them out as dropdown cards
            if evidence:
                with st.expander("📚 View Verified Source Context"):
                    for i, (docid, p) in enumerate(evidence, 1):
                        st.markdown(f"**[{i}] {docid}** *({p.source})*")
                        st.caption(f"\"{p.passage}\"")
                        
        # Save structural logs into history arrays to preserve chat state tracking
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "evidence": evidence
        })
