import os
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-20b"


def _call_groq(system_prompt: str, user_text: str, temperature: float = 0.2) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }

    for attempt in range(4):
        time.sleep(1)
        resp = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=120)
        if resp.status_code == 429 and attempt < 3:
            time.sleep(5 * (attempt + 1))
            continue
        if not resp.ok:
            raise RuntimeError(f"Groq Chat API error {resp.status_code}: {resp.text}")
        return resp.json()["choices"][0]["message"]["content"].strip()

    raise RuntimeError("Groq Chat API rate limit exceeded.")


def generate_overview(transcript: str) -> tuple[str, str]:
    if len(transcript) <= 25000:
        raw = _call_groq(
            "You are an expert meeting summarizer. Return two exact sections separated by '===SUMMARY===':\n\n"
            "1. First section: A short professional meeting title (max 8 words). Only the title.\n\n"
            "===SUMMARY===\n"
            "2. Second section: A clear, professional meeting summary in concise bullet points.",
            transcript,
            temperature=0.3,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=300)
        chunks = splitter.split_text(transcript)
        partials = [
            _call_groq("Summarize this portion of a meeting transcript concisely.", c, 0.3)
            for c in chunks
        ]
        combined_summary = _call_groq(
            "Combine these partial summaries into one final professional meeting summary in bullet points.",
            "\n\n".join(partials),
            temperature=0.3,
        )
        title = _call_groq(
            "Based on the meeting summary, generate a short professional meeting title "
            "(max 8 words). Only return the title, nothing else.",
            combined_summary[:2500],
            temperature=0.3,
        )
        return title.strip().strip("#*\" ") or "Meeting Summary", combined_summary.strip()

    if "===SUMMARY===" in raw:
        title_part, summary_part = raw.split("===SUMMARY===", 1)
        return (
            title_part.strip().strip("#*\" ") or "Meeting Summary",
            summary_part.strip() or "No summary generated.",
        )

    return "Meeting Summary", raw.strip()


def extract_all_insights(transcript: str) -> dict[str, str]:
    raw = _call_groq(
        "You are an expert meeting analyst. Analyze the transcript and return three exact sections "
        "separated by '===DECISIONS===' and '===QUESTIONS===':\n\n"
        "1. First section: All action items. For each provide:\n"
        "- Task description\n- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n"
        "Format as a numbered list. If none found say 'No action items found.'\n\n"
        "===DECISIONS===\n"
        "2. Second section: All key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'\n\n"
        "===QUESTIONS===\n"
        "3. Third section: All unresolved questions or topics needing follow-up. "
        "Format as a numbered list. If none found say 'No open questions found.'",
        transcript,
    )

    decisions_marker = "===DECISIONS==="
    questions_marker = "===QUESTIONS==="

    actions_part = raw
    decisions_part = "No key decisions found."
    questions_part = "No open questions found."

    if decisions_marker in raw:
        actions_part, rest = raw.split(decisions_marker, 1)
        if questions_marker in rest:
            decisions_part, questions_part = rest.split(questions_marker, 1)
        else:
            decisions_part = rest

    return {
        "actions": actions_part.strip() or "No action items found.",
        "decisions": decisions_part.strip() or "No key decisions found.",
        "questions": questions_part.strip() or "No open questions found.",
    }
