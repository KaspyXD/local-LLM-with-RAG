import streamlit as st
import os

from langchain_ollama import OllamaLLM
from chromadb import Client
from chromadb.config import Settings
from llm import getStreamingChain

st.title("Local LLM with RAG 📚")

# Set embedding model and LLM model explicitly
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2-vision"

if "llm" not in st.session_state:
    st.session_state["llm"] = OllamaLLM(model=LLM_MODEL)

# Load existing ChromaDB
if "db" not in st.session_state:
    try:
        with st.spinner("Connecting to ChromaDB..."):
            chroma_client = Client(
                Settings(
                    chroma_api_impl="chromadb.api.fastapi.FastAPI",
                    chroma_server_host="http://localhost",
                    chroma_server_port=800
                )
            )
            st.session_state["db"] = chroma_client.get_or_create_collection(name="default")
        st.success("ChromaDB connected successfully!")
    except Exception as e:
        st.error(f"Failed to connect to ChromaDB: {e}")

# Display an input box for queries
query = st.text_input("Enter your query below:")

if query:
    with st.spinner("Processing your query..."):
        try:
            # Ensure all arguments are passed correctly
            db = st.session_state.get("db")
            llm = st.session_state.get("llm")

            if db and llm:
                # Stream response
                stream = getStreamingChain(query, llm, db)
                response = "".join(stream)
                st.markdown(f"**Response:**\n{response}")
            else:
                st.error("Database or LLM is not initialized correctly.")
        except Exception as e:
            st.error(f"Error processing query: {e}")
