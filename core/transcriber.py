import os
import requests

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_TRANSLATE_URL = "https://api.groq.com/openai/v1/audio/translations"


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    is_hinglish = language.lower() == "hinglish"
    url = GROQ_TRANSLATE_URL if is_hinglish else GROQ_TRANSCRIBE_URL
    model = "whisper-large-v3" if is_hinglish else "whisper-large-v3-turbo"

    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": model, "response_format": "json"}
    if not is_hinglish:
        data["language"] = "en"

    with open(chunk_path, "rb") as f:
        files = {"file": (os.path.basename(chunk_path), f, "audio/mpeg")}
        response = requests.post(url, headers=headers, files=files, data=data, timeout=120)

    if not response.ok:
        raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

    return response.json().get("text", "").strip()


def transcribe_all(chunks: list, language: str = "english") -> str:
    transcripts = []
    for chunk in chunks:
        try:
            transcripts.append(transcribe_chunk(chunk, language=language))
        finally:
            if os.path.exists(chunk):
                os.remove(chunk)
    return " ".join(transcripts).strip()
