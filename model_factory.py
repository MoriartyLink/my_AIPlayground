import vertexai
from vertexai.generative_models import GenerativeModel

def get_model(project_id, location, credentials, corpus_id, system_instruction):
    vertexai.init(project=project_id, location=location, credentials=credentials)
    
    # We move the Tool initialization here
    from vertexai.preview import rag
    from vertexai.generative_models import Tool
    
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=corpus_id)],
                similarity_top_k=3,
            ),
        )
    )

    return GenerativeModel(
        model_name="gemini-2.0-flash",
        tools=[rag_retrieval_tool],
        system_instruction=system_instruction
    )
