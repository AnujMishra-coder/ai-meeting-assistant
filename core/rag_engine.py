from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.analyzer import _call_groq


class MeetingRAGAssistant:
    def __init__(self, transcript: str):
        clean_text = (transcript or "").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        self.chunks = splitter.split_text(clean_text) or ([clean_text] if clean_text else [""])
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self._has_vocab = False

        try:
            self.matrix = self.vectorizer.fit_transform(self.chunks)
            self._has_vocab = True
        except ValueError:
            self.matrix = None

    def invoke(self, question: str) -> str:
        if not question or not question.strip():
            return "Please enter a question about the meeting."

        if self._has_vocab and self.matrix is not None:
            q_vec = self.vectorizer.transform([question])
            scores = cosine_similarity(q_vec, self.matrix).flatten()
            ranked = sorted(
                ((idx, float(score)) for idx, score in enumerate(scores) if score > 0),
                key=lambda item: item[1],
                reverse=True,
            )[:4]
            context = "\n\n".join(self.chunks[idx] for idx, _ in ranked) or "\n\n".join(self.chunks[:4])
        else:
            context = "\n\n".join(self.chunks[:4])

        system_prompt = (
            "You are an expert meeting assistant. Answer the user's question "
            "based ONLY on the meeting transcript context provided below.\n\n"
            "If the answer is not found in the context, say:\n"
            '"I could not find this information in the meeting transcript."\n\n'
            "Always be concise and precise. If quoting someone, mention it clearly.\n\n"
            f"Context from meeting transcript:\n{context}"
        )

        try:
            return _call_groq(system_prompt, question.strip(), temperature=0.3)
        except Exception as exc:
            return f"Unable to answer right now due to an API error: {exc}"


def build_rag_chain(transcript: str) -> MeetingRAGAssistant:
    return MeetingRAGAssistant(transcript)
