import streamlit as st
from openai import OpenAI
import pypdf
from io import BytesIO
import os
from dotenv import load_dotenv # Use dotenv for local development
import markdown

# --- Langchain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
# Langchain message types (ensure compatibility across versions)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

# --- Configuration ---
# Load environment variables (optional, good practice for local dev)
load_dotenv()

# --- API Key Management ---
# Recommended: Use Environment Variables or Streamlit Secrets
# Example using environment variables:
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["LANGSMITH_TRACING"]='true'
os.environ["LANGSMITH_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]="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"
os.environ["LANGSMITH_PROJECT"]="3 Way Match Agent"