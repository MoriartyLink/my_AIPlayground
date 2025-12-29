import streamlit as st
import vertexai
import json
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
PROJECT_ID = "my-project-482605"
LOCATION = "europe-west1" 
CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/4611686018427387904"
KEY_FILE = "service-account.json" # The file you just downloaded

# --- 2. AUTHENTICATION (The Explicit Way) ---
try:
    # We load the credentials directly from your JSON file
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
    
    # Initialize Vertex AI with those specific credentials
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"Failed to load service-account.json: {e}")
    st.stop()

# --- 3. SETUP UI ---
st.set_page_config(page_title="Gemini RAG Tester", page_icon="🤖")
st.title("🤖 Vertex AI RAG Chat")
st.markdown(f"**Status:** Connected to `{LOCATION}`")

# --- 4. INITIALIZE RAG TOOL ---
# Note: Using the 2025 'Retrieval' + 'VertexRagStore' syntax
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[rag.RagResource(rag_corpus=CORPUS_ID)],
            similarity_top_k=3,
        ),
    )
)

model = GenerativeModel(
    model_name="gemini-2.0-flash",
    tools=[rag_retrieval_tool]
)

# --- 5. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response = model.generate_content(prompt)
                answer = response.text
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Grounding Metadata (Sources)
                if hasattr(response.candidates[0], 'grounding_metadata') and \
                   response.candidates[0].grounding_metadata.grounding_chunks:
                    with st.expander("📚 Source Documents"):
                        for i, chunk in enumerate(response.candidates[0].grounding_metadata.grounding_chunks):
                            title = chunk.web_content.title if chunk.web_content else "RAG Document"
                            st.info(f"Source {i+1}: {title}")
            except Exception as e:
                st.error(f"RAG Error: {str(e)}")
