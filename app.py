import os
import threading
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv()


@dataclass(frozen=True)
class BrainItem:
    text: str


class InMemoryBrain:
    """
    In-memory "RAG brain" for demo purposes.

    Stores:
      - items: list[BrainItem]
      - embeddings: np.ndarray shape (n_items, dim)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: list[BrainItem] = []
        self._embeddings: np.ndarray | None = None

    def clear_and_ingest(self, lines: list[str], embeddings: np.ndarray) -> None:
        with self._lock:
            self._items = [BrainItem(text=line) for line in lines]
            self._embeddings = embeddings.astype(np.float32, copy=False)

    def is_ready(self) -> bool:
        with self._lock:
            return bool(self._items) and self._embeddings is not None

    def stats(self) -> dict:
        with self._lock:
            n = len(self._items)
            dim = int(self._embeddings.shape[1]) if self._embeddings is not None and n else 0
            return {"items": n, "embedding_dim": dim}

    def top_k(self, query_embedding: np.ndarray, k: int = 3) -> list[str]:
        with self._lock:
            if not self._items or self._embeddings is None:
                return []
            embeddings = self._embeddings
            items = self._items

        # query_embedding expected shape (1, dim)
        sims = cosine_similarity(query_embedding, embeddings)[0]  # shape (n,)
        k = max(1, min(int(k), len(items)))
        top_idx = np.argsort(-sims)[:k]
        return [items[i].text for i in top_idx]


app = Flask(__name__)

# Global in-memory storage (demo only)
brain = InMemoryBrain()

# Load embedding model once at startup.
embedder = SentenceTransformer("all-MiniLM-L6-v2")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file provided (expected form field 'file')."}), 400

    file = request.files["file"]
    filename = (file.filename or "").strip()
    if not filename.lower().endswith(".txt"):
        return jsonify({"ok": False, "error": "Only .txt files are supported."}), 400

    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return jsonify({"ok": False, "error": "The uploaded file had no non-empty lines to ingest."}), 400

    # Generate embeddings for each line (each non-empty line is a rule/Q&A).
    embeddings = embedder.encode(
        lines,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    if embeddings.ndim != 2 or embeddings.shape[0] != len(lines):
        return jsonify({"ok": False, "error": "Embedding generation failed unexpectedly."}), 500

    brain.clear_and_ingest(lines=lines, embeddings=embeddings)
    return jsonify({"ok": True, "ingested_lines": len(lines), "brain": brain.stats()})


@app.post("/api/draft")
def api_draft():
    payload = request.get_json(silent=True) or {}
    user_prompt = (payload.get("prompt") or "").strip()
    message_type = (payload.get("message_type") or "").strip()
    tone = (payload.get("tone") or "").strip()

    if not user_prompt:
        return jsonify({"ok": False, "error": "Missing required field: prompt"}), 400
    if message_type not in {"Email", "WhatsApp Message"}:
        return jsonify({"ok": False, "error": "message_type must be one of: Email, WhatsApp Message"}), 400
    if tone not in {"Formal", "Informal"}:
        return jsonify({"ok": False, "error": "tone must be one of: Formal, Informal"}), 400
    if not brain.is_ready():
        return jsonify({"ok": False, "error": "Brain is not trained yet. Upload a .txt file first."}), 400

    query_emb = embedder.encode(
        [user_prompt],
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    retrieved_chunks = brain.top_k(query_embedding=query_emb, k=3)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "OPENAI_API_KEY is not set. Add it to your environment (see .env.example).",
                }
            ),
            500,
        )

    client = OpenAI(api_key=api_key)

    retrieved_text = "\n".join(f"- {c}" for c in retrieved_chunks)
    llm_prompt = (
        "You are an expert communication assistant. "
        f"Draft a {tone} {message_type} based on this user request: {user_prompt}. "
        "You MUST strictly follow these retrieved user rules and facts:\n"
        f"{retrieved_text}\n"
        "If drafting a WhatsApp message, keep it concise and natural for mobile chat."
        "The reply should be in the same language as the user request."
        "Only respond with the document text, no other text or explanation."
        "Do not add any additional information or commentary."
        "if you do not know the answer, say so. say it is not in my brain i have to train it first."
        "Do not use any emojis or special characters."
    )

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You follow instructions precisely and keep outputs clear."},
            {"role": "user", "content": llm_prompt},
        ],
        temperature=0.5,
    )

    drafted = (completion.choices[0].message.content or "").strip()
    return jsonify({"ok": True, "draft": drafted, "retrieved_chunks": retrieved_chunks})


if __name__ == "__main__":
    # Local dev server (Railway/Docker uses gunicorn; see Procfile / Dockerfile).
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)

