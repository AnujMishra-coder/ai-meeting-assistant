from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.analyzer import _call_groq


class MeetingRAGAssistant:
    def __init__(self, transcript: str):
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        self.chunks = splitter.split_text(transcript) or [transcript]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.chunks)

    def invoke(self, question: str) -> str:
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:4]
        context = (
            "\n\n".join(self.chunks[i] for i in top_indices if scores[i] > 0)
            or "\n\n".join(self.chunks[:4])
        )

        system_prompt = (
            "You are an expert meeting assistant. Answer the user's question "
            "based ONLY on the meeting transcript context provided below.\n\n"
            "If the answer is not found in the context, say:\n"
            '"I could not find this information in the meeting transcript."\n\n'
            "Always be concise and precise. If quoting someone, mention it clearly.\n\n"
            f"Context from meeting transcript:\n{context}"
        )
        return _call_groq(system_prompt, question, temperature=0.3)


def build_rag_chain(transcript: str) -> MeetingRAGAssistant:
    return MeetingRAGAssistant(transcript)
