import streamlit as st
import vertexai
import os
from google.oauth2 import service_account
from vertexai.generative_models import GenerativeModel, Tool, Content, Part
from vertexai.preview import rag
from dotenv import load_dotenv

# --- 1. SETUP UI (Must be first Streamlit command) ---
st.set_page_config(page_title="Gemini RAG Tester", page_icon="🤖")

# --- 2. CONFIGURATION ---
load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RAW_CORPUS_ID = os.getenv("CORPUS_ID")
CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAW_CORPUS_ID}"

# --- 3. AUTHENTICATION ---
try:
    # Convert AttrDict to standard dict to satisfy google-auth
    creds_info = dict(st.secrets["gcp_service_account"])
    
    # Fix Base64 padding/newline issues on the string value
    if "private_key" in creds_info:
        # Replace literal \n and strip stray whitespace
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

GUIDED_SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach (Agri Venture Studio).
Language: ALWAYS respond in English.
Style: sharp, peer-to-peer, collaborative. 

CORE BEHAVIOR:
1. Don't ask generic questions.
2. Use the Co-Engineered framework to offer specific directions.
3. Every response ends with a "Pivot Question".
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

def get_vertex_history():
    vertex_history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        vertex_history.append(
            Content(role=role, parts=[Part.from_text(msg["content"])])
        )
    return vertex_history

# --- 5. CHAT UI ---
st.title("🤖 Vertex AI RAG Chat")
st.caption(f"Connected to Corpus: `{RAW_CORPUS_ID}`")

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
        chat = model.start_chat(history=get_vertex_history()[:-1])
        response = chat.send_message(prompt)
        answer = response.text
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})