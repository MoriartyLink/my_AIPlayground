import streamlit as st
import vertexai
import os  # <--- Add this
from dotenv import load_dotenv # <--- Add this
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag
from google.oauth2 import service_account
from dotenv import load_dotenv
from vertexai.generative_models import Content, Part 
# --- 1. CONFIGURATION ---
load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION")
RAW_CORPUS_ID = os.getenv("CORPUS_ID")


CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/{RAW_CORPUS_ID}"
KEY_FILE = "service-account.json" 


# --- 2. AUTHENTICATION (The Secrets Way) ---
if "gcp_service_account" not in st.secrets:
    st.error("❌ Key 'gcp_service_account' not found in secrets.toml.")
    st.info("Ensure the file is at .streamlit/secrets.toml and has the [gcp_service_account] header.")
    st.stop()

try:
    creds_info = st.secrets["gcp_service_account"]
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"❌ Auth Error: {e}")
    st.stop()

# --- 3. SETUP UI ---
st.set_page_config(page_title="Gemini RAG Tester", page_icon="🤖")
st.title("🤖 Vertex AI RAG Chat")
st.caption(f"Connected to Corpus: `{CORPUS_ID.split('/')[-1]}` in `{LOCATION}`")

# --- 4. INITIALIZE RAG TOOL ---
st.write(f"Welcome to Co-engineer Platform! ")
# Using the latest VertexRagStore syntax (Fixes the 'RagRetrieval' error)
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(
                    rag_corpus=CORPUS_ID)
            ],
            similarity_top_k=3,  # Retrievals 3 relevant chunks
        ),
    )
)
# System Instructions for Contextual Responses
GUIDED_SYSTEM_PROMPT = """
Role: Guided Co-Engineering Coach (Agri Venture Studio).
Language: ALWAYS respond in English. Do not use French or other languages.
Style: sharp, peer-to-peer, collaborative. 

CORE BEHAVIOR:
1. Don't ask generic questions like "What can I do for you?". 
2. Instead, use the Co-Engineered framework (from the PDF) to offer specific directions.
3. Every response should end with a "Pivot Question" to guide the user.

Example Interaction Style:
- User: "Help me with a project."
- AI: "Let's dive in. Based on the Studio framework, we should start by identifying a 'Value-chain pain'. Are we looking at field-level climate stress or post-harvest logistics? Pick a lane and let's co-engineer it."
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

def get_vertex_history():
    """Converts streamlit session state to Vertex AI Content objects."""
    vertex_history = []
    for msg in st.session_state.messages:
        # Vertex AI uses 'user' and 'model' (not 'assistant')
        role = "user" if msg["role"] == "user" else "model"
        vertex_history.append(
            Content(role=role, parts=[Part.from_text(msg["content"])])
        )
    return vertex_history



# --- 5. CHAT LOGIC ---

# 1. Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Transformer (Dictionary -> Vertex Content Objects)
from vertexai.generative_models import Content, Part


# 4. Input & Response
if prompt := st.chat_input("Ask your Co-Engineer coach...", key="agri_chat"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # We start the chat with history, excluding the prompt we just added
        chat = model.start_chat(history=get_vertex_history()[:-1])
        response = chat.send_message(prompt)
        
        answer = response.text
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})



# --- DEBUG ONLY ---
st.write("Available Secret Keys:", list(st.secrets.keys()))
# ------------------

if "gcp_service_account" not in st.secrets:
    st.error("❌ Key 'gcp_service_account' missing.")
    st.stop()