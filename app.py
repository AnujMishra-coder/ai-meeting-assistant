import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Load Streamlit Cloud secrets into environment variables if present
for key in ("MISTRAL_API_KEY", "GROQ_API_KEY", "YOUTUBE_COOKIES"):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = st.secrets[key]

from utils.audio_processor import process_input, get_youtube_transcript
from core.transcriber import transcribe_all
from core.analyzer import (
    generate_title,
    summarize,
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain

st.set_page_config(page_title="AI Meeting Assistant", page_icon="🎙️", layout="wide")
st.title("🎙️ AI Meeting Assistant")
st.caption("Upload an audio/video file or paste a YouTube URL for summaries, action items, decisions, and RAG Q&A.")

if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    uploaded_file = st.file_uploader(
        "Upload Audio / Video",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
    )
with col2:
    youtube_url = st.text_input("Or Paste YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
with col3:
    language = st.radio("Language", ["english", "hinglish"])

if st.button("Analyze Meeting", type="primary", use_container_width=True):
    source = None
    temp_path = None

    if uploaded_file is not None:
        suffix = os.path.splitext(uploaded_file.name)[1] or ".mp3"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name
        source = temp_path
    elif youtube_url.strip():
        source = youtube_url.strip()

    if not source:
        st.error("Please upload an audio/video file or paste a YouTube URL.")
    else:
        try:
            with st.status("Analyzing meeting...", expanded=True) as status:
                transcript = None

                if source.startswith(("http://", "https://")):
                    st.write("1/5 Fetching YouTube captions...")
                    transcript = get_youtube_transcript(source)

                if not transcript:
                    st.write("1/5 Processing audio...")
                    chunks = process_input(source)

                    st.write("2/5 Transcribing with Groq Whisper...")
                    transcript = transcribe_all(chunks, language=language)
                else:
                    st.write("2/5 Captions fetched directly (skipped audio download).")

                if not transcript:
                    raise RuntimeError("Could not extract any speech from the audio.")

                st.write("3/5 Generating title & summary...")
                title = generate_title(transcript)
                summary = summarize(transcript)

                st.write("4/5 Extracting action items, decisions & questions...")
                actions = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                st.write("5/5 Building RAG search index...")
                rag_chain = build_rag_chain(transcript)

                status.update(label="Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "summary": summary,
                "actions": actions,
                "decisions": decisions,
                "questions": questions,
                "transcript": transcript,
                "rag_chain": rag_chain,
            }
            st.session_state.messages = []
        except Exception as e:
            if "403" in str(e):
                st.error(
                    "YouTube blocked the cloud server from downloading this video (HTTP 403). "
                    "Please download the audio locally and use the 'Upload Audio / Video' box instead."
                )
            else:
                st.error(f"Error: {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

if st.session_state.result:
    res = st.session_state.result
    st.subheader(f"📌 {res['title']}")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📋 Summary",
        "✅ Action Items",
        "🔑 Key Decisions",
        "❓ Open Questions",
        "📝 Full Transcript",
        "💬 Chat with Meeting",
    ])

    with tab1:
        st.markdown(res["summary"])
    with tab2:
        st.markdown(res["actions"])
    with tab3:
        st.markdown(res["decisions"])
    with tab4:
        st.markdown(res["questions"])
    with tab5:
        st.text_area("Transcript", res["transcript"], height=350)
    with tab6:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask anything about the meeting..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching transcript..."):
                    answer = res["rag_chain"].invoke(prompt)
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
