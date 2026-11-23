# AI Meeting Assistant (100% Free Streamlit Cloud Edition)

Lightweight, 100% free-to-deploy AI Meeting Assistant powered by:
- **Groq Whisper API** (free English & Hinglish transcription)
- **Mistral AI** (`mistral-small-latest` + `mistral-embed` for summaries, extraction, and RAG Q&A)
- **Streamlit Community Cloud** (100% free hosting, no credit card)

## Deploy for Free on Streamlit Community Cloud
1. Push this folder to a **public GitHub repository**.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Click **Create app** → select your repo, branch (`main`), and main file (`app.py`).
4. Click **Advanced settings** → paste your keys into **Secrets**:
   ```toml
   MISTRAL_API_KEY = "your_mistral_key_here"
   GROQ_API_KEY = "your_groq_key_here"
   ```
5. Click **Deploy!**
