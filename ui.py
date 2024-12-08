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
        with st.spinner("Loading ChromaDB..."):
            chroma_client = Client(
                Settings(
                    persist_directory="C:\\Users\\Stefan\\Documents\\Coding\\Project\\Carb-Wizard\\ChromaDB"
                )
            )
            st.session_state["db"] = chroma_client.get_or_create_collection(name="default")
        st.success("ChromaDB loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load ChromaDB: {e}")

# Display an input box for queries
query = st.text_input("Enter your query below:")

if query:
    with st.spinner("Processing your query..."):
        try:
            stream = getStreamingChain(query, st.session_state["llm"], st.session_state["db"])
            response = "".join(stream)
            st.markdown(f"**Response:**\n{response}")
        except Exception as e:
            st.error(f"Error processing query: {e}")
