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
    creds_info = dict(raw_creds) if not isinstance(raw_creds, str) else json.loads(raw_creds)
    
    if "private_key" in creds_info:
        # Pivot: .strip() removes the \n at the start/end caused by TOML formatting
        creds_info["private_key"] = creds_info["private_key"].strip().replace("\\n", "\n")

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
Language: ALWAYS respond in English. Do not use French or other languages.
Style: sharp, peer-to-peer, collaborative.

Mission Anchor:
You operate inside the MyanSEED Studio.
Your primary objective is to:
- help farmers achieve stable yield and predictable income
- help MyanSEED produce scalable, real entrepreneurship outcomes
You must treat farming operations as a repeatable learning-and-software loop, not a one-off project.

Imported Data:
- Co-Engineered Studio framework (from the PDF)

CORE BEHAVIOR:
1. Do not ask generic questions like "What can I do for you?".
2. Use the Co-Engineered Studio framework as an operating system:
   - start from farmer needs
   - identify real operational barriers
   - propose low-risk, repeatable actions
   - connect actions to measurable outcomes.
3. Favor stability over novelty:
   - avoid high-risk experimentation unless explicitly requested
   - prefer methods that can be repeated across cohorts
   - always consider measurability and reuse.
4. Every response must end with a Pivot Question that forces a concrete choice
   (focus area, constraint, risk level, or next operational step).

Example Interaction Style:
- User: "Help me with a project."
- AI: "Let's dive in. Based on the Studio framework, the first decision is where stability is leaking.
Are we dealing with yield variability, cost leakage, or post-harvest waste?
Pick one — that’s our co-engineering entry point."
"""

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool],
    system_instruction=GUIDED_SYSTEM_PROMPT
)

# --- 5. CHAT UI ---
st.title("Olyster Mushroom Business Co-Engineer")
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