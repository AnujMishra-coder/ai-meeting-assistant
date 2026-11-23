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
    )


def _run_prompt(system_prompt: str, text: str, temperature: float = 0.2) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    chain = prompt | get_llm(temperature) | StrOutputParser()
    time.sleep(1)  # Avoid free-tier rate limits
    return chain.invoke({"text": text})


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


def extract_action_items(transcript: str) -> str:
    return _run_prompt(
        "You are an expert meeting analyst. From the meeting transcript, extract all action items. "
        "For each provide:\n- Task description\n- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'",
        transcript,
    )


def extract_key_decisions(transcript: str) -> str:
    return _run_prompt(
        "You are an expert meeting analyst. Extract all key decisions made. "
        "Format as a numbered list. If none found say 'No key decisions found.'",
        transcript,
    )


def extract_questions(transcript: str) -> str:
    return _run_prompt(
        "From the meeting transcript, extract all unresolved questions or topics needing follow-up. "
        "Format as a numbered list. If none found say 'No open questions found.'",
        transcript,
    )
