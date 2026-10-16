# RepoPilot — V0

Paste a public GitHub repository URL to get a structured summary of its README.
This is a small learning project: direct LLM prompting, Pydantic, FastAPI and Streamlit.

```text
Browser → Streamlit → FastAPI → GitHub metadata + README
                            → OpenAI → Pydantic validation → summary
```

Read [the V0 learning and run guide](docs/V0_GUIDE.md) for the concepts and code walkthrough.

## Run with Docker

Install Docker with Compose, then from the repository root:

```sh
cp .env.example .env
# Edit .env: set OPENAI_API_KEY to your own OpenAI API key.
docker compose up --build
```

Open http://localhost:8501. Paste `https://github.com/pallets/flask` and click **Analyse**.
The API explorer is at http://localhost:8000/docs. Stop with `docker compose down`.
Live analysis sends the public README to OpenAI and uses your API credits.
The default model is `gpt-4o-mini`; change `OPENAI_MODEL` if needed to a Responses API
model that supports structured output and `temperature`.

## Run locally (Python 3.12+)

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Edit .env and add your API key.
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

## Scope and limitations

- Public repositories only; optional `GITHUB_TOKEN` helps with rate limits.
- Only metadata and the first 20,000 README characters are supplied. Missing README
  falls back to metadata, with an explicit warning and unknown run instructions.
- JSON/schema validity does not prove factual truth. Check generated commands against sources.
- Repository text is treated as untrusted input; the LLM has no tools or GitHub write access.
  Prompt instructions alone do not prove injection resistance.
- No embeddings, Qdrant, RAG, citations per claim, or issue triage yet: those belong to V1/V2.
- Intended for local learning; no authentication or public deployment setup.

Verified evidence and remaining live checks are in [CURRENT_STATE.md](CURRENT_STATE.md).
