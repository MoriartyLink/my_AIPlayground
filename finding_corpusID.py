from vertexai.preview import rag
import vertexai

# Use your project details and the region where you finally got it to work
PROJECT_ID = "my-project-482605"
LOCATION = "europe-west1" # Or whichever region you switched to

vertexai.init(project=PROJECT_ID, location=LOCATION)

# This lists all your corpora
for corpus in rag.list_corpora():
    print(f"Display Name: {corpus.display_name}")
    print(f"CORPUS_ID to use: {corpus.name}")
    print("-" * 30)
