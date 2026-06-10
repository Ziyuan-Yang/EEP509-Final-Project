import pickle
import math
from typing import List, Dict, Tuple, Any, Optional
from openai import OpenAI

CHAT_MODEL = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-small"


def cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def load_local_vectorstore(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        vs = pickle.load(f)

    if isinstance(vs, dict) and "chunks" in vs and "vectors" in vs:
        chunks = vs["chunks"]
        vectors = vs["vectors"]

        try:
            chunks_list = list(chunks)
            vectors_list = list(vectors)
        except Exception as e:
            raise ValueError(f"chunks/vectors must be iterable sequences. Error: {e}")

        if len(chunks_list) != len(vectors_list):
            raise ValueError(f"chunks and vectors length mismatch: {len(chunks_list)} vs {len(vectors_list)}")

        items: List[Dict[str, Any]] = []
        for c, v in zip(chunks_list, vectors_list):
            text = ""
            page_number = None

            if isinstance(c, str):
                text = c
            elif isinstance(c, dict):
                text = c.get("text") or c.get("chunk") or c.get("content") or ""
                page_number = c.get("page_number") or c.get("page") or c.get("page_num")
            else:
                text = str(c)

            try:
                emb = v.tolist() if hasattr(v, "tolist") else list(v)
            except Exception:
                emb = v  # last resort; cosine() will fail if not iterable

            items.append({"text": text, "embedding": emb, "page_number": page_number})

        return items

    if isinstance(vs, list):
        items = []
        for it in vs:
            if not isinstance(it, dict):
                continue
            emb = it.get("embedding") or it.get("vector") or it.get("vectors")
            if emb is None:
                continue
            txt = it.get("text") or it.get("chunk") or it.get("content") or ""
            pg = it.get("page_number") or it.get("page") or it.get("page_num")

            try:
                emb2 = emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception:
                emb2 = emb

            items.append({"text": txt, "embedding": emb2, "page_number": pg})

        if not items:
            raise ValueError("Vectorstore list format found, but no valid items with embeddings.")
        return items

    raise ValueError("Unsupported local_vectorstore.pkl format. Expected dict(chunks,vectors) or list(items).")


class Obnoxious_Agent:
    def __init__(self, client: OpenAI) -> None:
        self.client = client
        self.prompt = (
            "You are a classifier. Decide if the user's message is obnoxious (rude/insulting). "
            "Return ONLY 'Yes' or 'No'."
        )

    def extract_action(self, response: str) -> bool:
        t = (response or "").strip().lower()
        return t.startswith("yes")

    def check_query(self, query: str) -> bool:
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"User message: {query}\nAnswer (Yes/No):"},
            ],
            temperature=0,
        )
        output = resp.choices[0].message.content
        return self.extract_action(output)


class Context_Rewriter_Agent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.system_prompt = (
            "You are a context resolution agent in a multi-agent system.\n"
            "Your ONLY task is to rewrite the user's latest message into a fully self-contained question.\n\n"
            "Constraints:\n"
            "- Do NOT answer.\n"
            "- Do NOT add new information.\n"
            "- If already self-contained, return unchanged.\n"
            "- Output ONLY the rewritten query."
        )

    def rephrase(self, user_history: str, latest_query: str) -> str:
        if not user_history or not user_history.strip():
            return latest_query.strip()

        user_message = (
            "Conversation History:\n"
            f"{user_history}\n\n"
            "Latest User Message:\n"
            f"{latest_query}\n\n"
            "Rewrite into a fully self-contained query:"
        )

        try:
            resp = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
            rewritten = (resp.choices[0].message.content or "").strip()
            return rewritten if rewritten else latest_query.strip()
        except Exception:
            return latest_query.strip()


class Query_Agent:
    def __init__(self, local_vectorstore_path: str, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.vs = load_local_vectorstore(local_vectorstore_path)

        self.prompt = (
            "You are a strict classifier for a Machine Learning textbook assistant.\n"
            "Determine whether the user's query is relevant to a Machine Learning textbook.\n"
            "Relevant examples: logistic regression, overfitting, KNN, EM, loss functions, bias-variance.\n"
            "Irrelevant examples: cooking, sports, travel, celebrities, weather.\n\n"
            "Output ONLY one word: RELEVANT or IRRELEVANT."
        )

    def extract_action(self, response: str) -> bool:
        text = (response or "").strip().upper()
        if "IRRELEVANT" in text:
            return False
        if "RELEVANT" in text:
            return True
        return False

    def _embed(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=[text.replace("\n", " ")],
        ).data[0].embedding
        return emb

    def query_vector_store(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Query: {query}\nAnswer:"},
            ],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip()
        if not self.extract_action(out):
            return []

        qv = self._embed(query)

        scored = []
        for item in self.vs:
            sim = cosine(qv, item["embedding"])
            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        docs = []
        for sim, item in scored[:k]:
            docs.append(
                {
                    "text": item.get("text", ""),
                    "page_number": item.get("page_number", None),
                    "score": float(sim),
                }
            )
        return docs

class Relevant_Documents_Agent:
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.prompt = (
            "You are a strict relevance judge.\n"
            "Given the user query and retrieved textbook snippets, decide if the snippets are relevant.\n"
            "Return ONLY 'Yes' or 'No'."
        )

    def get_relevance(self, query: str, docs: List[Dict[str, Any]]) -> str:
        snippet = "\n---\n".join([(d.get("text", "")[:300] or "") for d in docs]) if docs else ""
        convo = f"User query: {query}\n\nRetrieved snippets:\n{snippet}\n\nAre the snippets relevant?"

        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": convo},
            ],
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        return "Yes" if out.startswith("yes") else "No"

class Answering_Agent:
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.system_prompt = (
            "You are a helpful assistant answering questions using ONLY the provided textbook context. "
            "If the context is insufficient, say you cannot find it in the book. "
            "Answer in English."
        )
        self.refusal_text = (
            "This query is not relevant to the context of this book. "
            "I would be happy to answer the question based on the book's context."
        )

    def generate_response(self, query: str, docs: List[Dict[str, Any]], conv_history: str, k: int = 5) -> str:
        if not docs:
            return self.refusal_text

        top_docs = docs[:k]
        context_blocks = []
        for i, d in enumerate(top_docs, start=1):
            page = d.get("page_number", "unknown")
            text = d.get("text", "")
            context_blocks.append(f"[Doc {i} | Page {page}]\n{text}")

        context_text = "\n\n".join(context_blocks)

        user_message = (
            f"Conversation History:\n{conv_history}\n\n"
            f"Textbook Context:\n{context_text}\n\n"
            f"User Question:\n{query}\n\n"
            "Instructions:\n"
            "- Answer using only the Textbook Context.\n"
            "- If the answer is not in the context, say you cannot find it in the provided text.\n"
            "- Mention page number(s) when helpful.\n"
        )

        resp = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,
        )
        reply = (resp.choices[0].message.content or "").strip()
        return reply if reply else "I could not generate a response from the provided context."



class Head_Agent:
    """
    Main controller. Returns (final_answer, agent_path_list).
    """
    def __init__(self, openai_client: OpenAI, local_vectorstore_path: str) -> None:
        self.client = openai_client
        self.local_vectorstore_path = local_vectorstore_path

        self.obnox = Obnoxious_Agent(self.client)
        self.rewriter = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.local_vectorstore_path, self.client)
        self.rel_docs_agent = Relevant_Documents_Agent(self.client)
        self.answer_agent = Answering_Agent(self.client)

    def __call__(self, user_input: str, history: str = "") -> Tuple[str, List[str]]:
        agent_path: List[str] = []

        agent_path.append("Obnoxious_Agent")
        if self.obnox.check_query(user_input):
            return (
                "I’m here to help, but I can’t respond to rude or insulting requests. Please rephrase politely.",
                agent_path,
            )

        agent_path.append("Context_Rewriter_Agent")
        rewritten = self.rewriter.rephrase(history, user_input)

        agent_path.append("Query_Agent")
        docs = self.query_agent.query_vector_store(rewritten, k=5)

        agent_path.append("Relevant_Documents_Agent")
        if docs:
            rel = self.rel_docs_agent.get_relevance(rewritten, docs)
            if rel == "No":
                docs = []

        agent_path.append("Answering_Agent")
        answer = self.answer_agent.generate_response(rewritten, docs, conv_history=history, k=5)

        return answer, agent_path