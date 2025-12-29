import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Tool
from vertexai.preview import rag
from google.oauth2 import service_account

# --- 1. CONFIGURATION ---
# Replace these with your actual details
PROJECT_ID = ""
LOCATION = "europe-west1" 
# Your specific Corpus ID
CORPUS_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/ragCorpora/1111111" #add your CORPUS_ID
KEY_FILE = "service-account.json" 

# --- 2. AUTHENTICATION ---
# We load credentials explicitly to avoid system permission errors
try:
    credentials = service_account.Credentials.from_service_account_file(KEY_FILE)
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"❌ Auth Error: Could not load '{KEY_FILE}'. Make sure it is in this folder.")
    st.stop()

# --- 3. SETUP UI ---
st.set_page_config(page_title="Gemini RAG Tester", page_icon="🤖")
st.title("🤖 Vertex AI RAG Chat")
st.caption(f"Connected to Corpus: `{CORPUS_ID.split('/')[-1]}` in `{LOCATION}`")

# --- 4. INITIALIZE RAG TOOL ---
# Using the latest VertexRagStore syntax (Fixes the 'RagRetrieval' error)
rag_retrieval_tool = Tool.from_retrieval(
    retrieval=rag.Retrieval(
        source=rag.VertexRagStore(
            rag_resources=[
                rag.RagResource(rag_corpus=CORPUS_ID)
            ],
            similarity_top_k=3,  # Retrievals 3 relevant chunks
        ),
    )
)

# Initialize Model (Using Gemini 2.0 to fix 404 errors)
model = GenerativeModel(
    model_name="gemini-2.0-flash-001", 
    tools=[rag_retrieval_tool]
)

# --- 5. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle Input
if prompt := st.chat_input("Ask about your documents..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                response = model.generate_content(prompt)
                answer = response.text
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Show Sources (Citations)
                if response.candidates[0].grounding_metadata.grounding_chunks:
                    with st.expander("📚 View Sources"):
                        for i, chunk in enumerate(response.candidates[0].grounding_metadata.grounding_chunks):
                            title = chunk.web_content.title if chunk.web_content else f"Chunk {i+1}"
                            uri = chunk.web_content.uri if chunk.web_content else "Internal RAG DB"
                            st.info(f"**{title}**\n\nLocation: `{uri}`")
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
