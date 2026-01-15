import streamlit as st


import vertexai
import os
import json
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, Content, Part
from vertexai.preview import rag
from dotenv import load_dotenv

# --- 1. SETUP UI (MUST BE FIRST) ---
st.set_page_config(page_title="Gemini RAG Tester", page_icon="🤖")

# --- 2. CONFIGURATION ---
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RAW_CORPUS_ID = os.getenv("CORPUS_ID")
CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAW_CORPUS_ID}"

# --- 3. AUTHENTICATION ---
try:
    raw_creds = st.secrets["gcp_service_account"]
    
    # If it's a string, parse it as JSON. If it's a dict, use it directly.
    if isinstance(raw_creds, str):
        creds_info = json.loads(raw_creds)
    else:
        creds_info = dict(raw_creds)
    
    # Clean the private key to ensure Base64 multiple-of-4 alignment
    if "private_key" in creds_info:
        creds_info["private_key"] = creds_info["private_key"].replace("\\n", "\n").strip()

    credentials = service_account.Credentials.from_service_account_info(creds_info)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"❌ Auth Error: {e}")
    st.stop()

# --- 4. INITIALIZE RAG TOOL ---
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)],
            similarity_top_k=3,
        ),
    )
)

GUIDED_SYSTEM_PROMPT = "Role: Agri Venture Studio Coach. Language: English."

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# --- 5. CHAT UI ---
st.title("🤖 Vertex AI RAG Chat")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your Co-Engineer coach..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        history = [Content(role="user" if m["role"] == "user" else "model", 
                   parts=[Part.from_text(m["content"])]) for m in st.session_state.messages[:-1]]
        chat = model.start_chat(history=history)
        response = chat.send_message(prompt)
        st.markdown(response.text)
        st.session_state.messages.append({"role": "assistant", "content": response.text})