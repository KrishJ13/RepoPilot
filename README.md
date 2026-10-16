# RepoPilot — V1

Summarise a public GitHub README (V0), or index repository sources and ask questions
with cited evidence (V1). A small learning project using FastAPI, Streamlit,
Sentence Transformers and Qdrant, with ordinary Python functions for the RAG flow.

```text
V0: URL → README → OpenAI → validated summary
V1 index: URL → documents → chunks → local embeddings → Qdrant
V1 ask: question → embedding → repository-scoped search → OpenAI → cited answer
```

Read the preserved [V0 guide](docs/V0_GUIDE.md), the new [V1 guide](docs/V1_GUIDE.md),
and the [measured V1 results](docs/V1_RESULTS.md).

## Run with Docker

Install Docker with Compose, then from the repository root:

```sh
cp -n .env.example .env  # Keep an existing configured .env
# Edit .env: set OPENAI_API_KEY to your own OpenAI API key.
docker compose up --build
```

If port 8000 is occupied, use `API_PORT=8001 docker compose up --build`;
the API docs then move to http://localhost:8001/docs. This session left V1 running
with that alternate API port, preserving the existing local port-8000 process.

Open http://localhost:8501. Paste `https://github.com/pallets/flask` and click **Analyse**.
For V1, scroll to **Repository Q&A**, index `https://github.com/encode/httpx`,
then ask a question and inspect its evidence. First indexing downloads the local
embedding model. Qdrant/model caches persist in Docker volumes.
The API explorer is at http://localhost:8000/docs. Stop with `docker compose down`.
Live analysis sends the public README to OpenAI and uses your API credits.
The default model is `gpt-4o-mini`; change `OPENAI_MODEL` if needed to a Responses API
model that supports structured output and `temperature`.

## Run locally (Python 3.12+)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp -n .env.example .env  # Keep an existing configured .env
# Edit .env and add your API key.
docker compose up -d qdrant
uvicorn repopilot.api:app --reload
```

In a second terminal, from the same root:

```sh
source .venv/bin/activate
streamlit run repopilot/ui.py
```

API example (with the server running):

```sh
curl -X POST http://localhost:8000/api/v1/repositories/summary \
  -H 'Content-Type: application/json' \
  -d '{"repository_url":"https://github.com/pallets/flask"}'
```

## Verify and evaluate

```sh
python -m pytest -q
# No key: exercise the evaluation file format with a clearly labelled fake response.
python -m evals.run --demo --output evals/results/my-demo.json
# With your key: capture a real model response to the fixed synthetic README.
python -m evals.run --output evals/results/my-live-baseline.json
```

Each evaluation captures one summary for ten manual review questions. Review the
saved summary against expected facts; fill in `correct`, `unsupported_claims` and
notes. Existing result files cannot be overwritten. This is a V0 summarizer,
not an arbitrary-question answering system. Tokens and elapsed time are returned;
cost and answer quality are not guessed.

## V1 endpoints and evaluation

- `POST /api/v1/repositories/index`: URL → bounded source snapshot → chunk/embed/store.
- `POST /api/v1/repositories/ask`: URL + question + optional K/type → cited answer.

```sh
python -m evals.v1_evaluate --output evals/results/my-v1-retrieval
# Optional: also run 120 live answer requests for the four V1 experiments.
python -m evals.v1_evaluate --live --output evals/results/my-v1-live
QDRANT_TEST_URL=http://localhost:6333 python -m pytest -q tests/test_v1.py
```

The 24-question synthetic baseline retrieved expected evidence at Recall@5=1.0.
RAG answered 24/24 expected facts versus 0/24 with the deliberately minimal README.
This demonstrates expanded evidence access on a controlled fixture, not production
accuracy. Raw outputs, manual review and cost/latency limitations are linked in
[the results](docs/V1_RESULTS.md).

## Scope and limitations

- Public repositories only; optional `GITHUB_TOKEN` helps with rate limits.
- V0 supplies only metadata and the first 20,000 README characters. Missing README
  falls back to metadata, with an explicit warning and unknown run instructions.
- JSON/schema validity does not prove factual truth. Check generated commands against sources.
- Repository text is treated as untrusted input; the LLM has no tools or GitHub write access.
  Prompt instructions alone do not prove injection resistance.
- V1 indexes at most eight docs, three releases and five issues plus README/metadata.
  Repository-scoped search and citation-ID validation are enforced. Citations still
  need factual-support review; vector similarity does not prove correctness.
- No V2 tools/agents or issue triage yet. No reranking or automatic index refresh.
- Intended for local learning; no authentication or public deployment setup.

Verified state and known limitations are in [CURRENT_STATE.md](CURRENT_STATE.md).
