import os
import re
import tempfile
import yt_dlp
from pydub import AudioSegment
from youtube_transcript_api import YouTubeTranscriptApi

DOWNLOAD_DIR = tempfile.gettempdir()


def get_youtube_transcript(url: str) -> str | None:
    """Fetch YouTube captions directly (avoids cloud IP 403 download errors)."""
    match = re.search(r"(?:v=|youtu\.be/|shorts/)([a-zA-Z0-9_-]{11})", url)
    if not match:
        return None

    video_id = match.group(1)
    try:
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            fetched = api.fetch(video_id, languages=["en", "hi"])
            return " ".join(snippet.text for snippet in fetched).strip()
        else:
            fetched = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "hi"])
            return " ".join(item["text"] for item in fetched).strip()
    except Exception:
        return None


def download_youtube_audio(url: str) -> str:
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "force_ipv4": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }
        ],
        "quiet": True,
        "noplaylist": True,
    }

    cookies_content = os.getenv("YOUTUBE_COOKIES")
    cookie_path = None
    if cookies_content:
        cookie_path = os.path.join(DOWNLOAD_DIR, "yt_cookies.txt")
        with open(cookie_path, "w", encoding="utf-8") as f:
            f.write(cookies_content.strip())
        ydl_opts["cookiefile"] = cookie_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return os.path.join(DOWNLOAD_DIR, f"{info['id']}.mp3")
    finally:
        if cookie_path and os.path.exists(cookie_path):
            os.remove(cookie_path)


def convert_to_mp3(input_path: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR, f"{os.path.splitext(os.path.basename(input_path))[0]}_conv.mp3"
    )
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="mp3", bitrate="64k")
    return output_path


def chunk_audio(mp3_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_file(mp3_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{mp3_path}_chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3", bitrate="64k")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        audio_path = download_youtube_audio(source)
    else:
        audio_path = convert_to_mp3(source)
    return chunk_audio(audio_path)
