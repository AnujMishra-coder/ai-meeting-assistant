import os
import time
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm(temperature: float = 0.2):
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=temperature,
        max_retries=0,
    )


def _run_prompt(system_prompt: str, text: str, temperature: float = 0.2) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    chain = prompt | get_llm(temperature) | StrOutputParser()

    for attempt in range(4):
        try:
            time.sleep(2)
            return chain.invoke({"text": text})
        except Exception as e:
            if "429" in str(e) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def generate_title(transcript: str) -> str:
    return _run_prompt(
        "Based on the meeting transcript, generate a short professional meeting title "
        "(max 8 words). Only return the title, nothing else.",
        transcript[:2500],
        temperature=0.3,
    )


def summarize(transcript: str) -> str:
    if len(transcript) <= 25000:
        return _run_prompt(
            "You are an expert meeting summarizer. Write a clear, professional "
            "meeting summary in concise bullet points.",
            transcript,
            temperature=0.3,
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=12000, chunk_overlap=300)
    chunks = splitter.split_text(transcript)
    partials = [
        _run_prompt("Summarize this portion of a meeting transcript concisely.", c, 0.3)
        for c in chunks
    ]
    return _run_prompt(
        "Combine these partial summaries into one final professional meeting summary in bullet points.",
        "\n\n".join(partials),
        temperature=0.3,
    )


def extract_all_insights(transcript: str) -> dict[str, str]:
    raw = _run_prompt(
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
