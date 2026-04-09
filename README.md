# Flask In-Memory RAG Demo

Working Flask prototype demonstrating Retrieval-Augmented Generation (RAG) **in memory**:

- Upload a `.txt` file where **each non-empty line** is a rule / fact / Q&A
- Generate local embeddings using `sentence-transformers`
- Retrieve top matches with cosine similarity (`scikit-learn`)
- Draft a personalized message via OpenAI (`gpt-4o-mini`)

## Project structure

- `app.py`: Flask backend + in-memory vector store
- `templates/index.html`: Tailwind + vanilla JS single-page UI
- `requirements.txt`: Python dependencies
- `.env.example`: environment variable template
- `Dockerfile`: production container for Railway
- `Procfile`: optional process declaration (Railway can use Dockerfile)

## Run locally (recommended)

1) Create a virtualenv and install dependencies:

```bash
cd ai_assist/flask-rag-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Set your OpenAI key:

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY
```

3) Start the app:

```bash
python app.py
```

Open `http://localhost:5000`.

## Run with Docker

```bash
cd ai_assist/flask-rag-demo
docker build -t flask-rag-demo .
docker run --rm -p 8080:8080 -e OPENAI_API_KEY="YOUR_KEY" flask-rag-demo
```

Open `http://localhost:8080`.

## Notes for demos

- The “brain” is **in memory** and resets on server restart.
- For a real product: persist vectors (Postgres/pgvector, Pinecone, Weaviate, etc.), add chunking, metadata, and auth.

