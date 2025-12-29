
#  Vertex AI RAG Chat

A lightweight chat interface to talk to your private documents (PDFs, Google Drive) using **Google Gemini 2.0** and **Vertex AI RAG Engine**.

##  Prerequisites

1.  **Google Cloud Project:** With "Vertex AI API" enabled.
2.  **RAG Corpus:** Created in Vertex AI Studio (Region: `europe-west1`).
3.  **Service Account Key:** A `service-account.json` file inside the project folder.
    * *Required Roles:* `Vertex AI User`, `Storage Object Viewer`.
4.  **Python 3.13+** (Installed on Manjaro via `pacman`).

##  Installation

Open your terminal in the project folder:

```bash
# 1. Create a virtual environment (keeps your system clean)
python -m venv venv

# 2. Activate the bubble (Fish Shell)
source venv/bin/activate.fish
# If using Bash/Zsh: source venv/bin/activate

# 3. Install dependencies
pip install --upgrade google-cloud-aiplatform streamlit google-auth

```

##  Configuration

1. Place your downloaded key file named `service-account.json` in the root folder.
2. Open `chat_test.py` and update these lines with your actual IDs:

```python
PROJECT_ID = "your-project-id"
LOCATION = "europe-west1"
CORPUS_ID = "projects/your-project/locations/europe-west1/ragCorpora/your-corpus-id"

```

##  Usage

Run the app locally:

```bash
streamlit run chat_test.py

```

*The interface will open in your browser at `http://localhost:8501`.*

##  Tech Stack

* **App Framework:** [Streamlit](https://streamlit.io/)
* **AI Model:** Gemini 2.0 Flash
* **Vector Search:** Vertex AI RAG Store
* **Auth:** Google Service Account (ADC)

---

**⚠️ Security Note:** Never commit `service-account.json` to GitHub. Add it to your `.gitignore`.
