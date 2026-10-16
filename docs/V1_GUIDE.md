# V1 — learn retrieval-augmented generation

V0 sends one README to the model. V1 first searches a larger set of repository
sources, then asks the model to answer from the selected passages. This is
**retrieval-augmented generation (RAG)**. The language model still generates the
answer; the retriever selects its evidence. Neither step retrains the language model.

The [V0 guide](V0_GUIDE.md) is preserved. This guide explains only what V1 adds.

## Run and use V1

Your existing `.env` still supplies the OpenAI key. You do not need a paid Hugging
Face account or an OpenAI embedding API: the small embedding model runs locally.

```sh
# From the repository root; requires Docker with Compose.
docker compose up --build
```

If your local Uvicorn already occupies port 8000, leave it running and use:

```sh
API_PORT=8001 docker compose up --build
```

This session verified the second command's layout: UI at http://localhost:8501,
container API docs at http://localhost:8001/docs, Qdrant at http://localhost:6333/dashboard.
The API port normally defaults to 8000. `UI_PORT` can also override 8501 if needed.
Changing a host API port does not change the UI container's internal `http://api:8000` address.

1. Open the UI. The original **Analyse** form still gives the V0 README summary.
2. In **V1 · Repository Q&A**, enter `https://github.com/encode/httpx`.
3. Click **Index repository**. First use downloads the embedding model (roughly
   90 MB of weights); allow a few minutes on a slow connection. Later requests reuse it.
4. Ask “How do I configure connection pool limits?” or “How do I install HTTPX?”
5. Read the answer, open its source buttons, and expand the retrieved evidence.
6. Try changing K or the source-type filter. The answer can change when evidence changes.

Local Python alternative:

```sh
source .venv/bin/activate
pip install -r requirements-dev.txt
docker compose up -d qdrant
uvicorn repopilot.api:app --reload
# Second terminal, same root and activated environment:
streamlit run repopilot/ui.py
```

Python 3.12+ is required; the tested runtime is 3.13. A Linux machine without a GPU
can install the CPU-only PyTorch wheel before the requirements, as in the Dockerfile.
The first install is larger than V0 because local neural inference needs PyTorch.

API requests, using whichever port you started:

```sh
curl -X POST http://localhost:8001/api/v1/repositories/index \
  -H 'Content-Type: application/json' \
  -d '{"repository_url":"https://github.com/encode/httpx"}'

curl -X POST http://localhost:8001/api/v1/repositories/ask \
  -H 'Content-Type: application/json' \
  -d '{"repository_url":"https://github.com/encode/httpx","question":"How do I configure connection pool limits?","top_k":5}'
```

The response includes claims, supporting chunk IDs, source passages, cosine scores,
model/prompt version, token counts and elapsed time. Indexing does not call the LLM.
Asking a question does use your OpenAI credits.

## 1. Documents: collect useful evidence

Goal: answer questions whose evidence is outside the README. The simplest next
step is to collect more text. `repopilot/ingestion.py` fetches one root README,
up to eight docs, three releases, five recent issues, and repository metadata.
These limits keep the first version understandable and affordable. It is not a
complete mirror of the repository. The index response reports selection/size limits.

A `Document` contains text plus repository, type, title, source URL and metadata.
For example, issue metadata retains its number; release metadata retains its tag.
This is **provenance**: knowing where text came from after it has been transformed.
GitHub issue listings also include pull requests, so the code explicitly excludes PRs.

```python
commit = get(f"/repos/{repository}/commits/{branch}")["sha"]
```

The actual code URL-encodes the branch. It resolves a commit before reading files,
so file links point to that snapshot rather than a changing branch. Issues/releases
are sampled live and retain timestamps; they are not immutable Git snapshots.
Files over 40,000 bytes are skipped with a warning. A failed required network call
fails indexing visibly. Re-index when sources change; nothing refreshes automatically.

Checkpoint: inspect the Index response's document count and warnings. Explain back:
why keep source URLs and timestamps instead of saving only the text?

## 2. Chunks: fit useful pieces into model inputs

The naive approach embeds an entire long document. But the embedding model has
a finite input length: this model accepts 256 tokens. Passing more text can quietly
truncate the end. A vector that never saw the answer cannot reliably retrieve it.

`repopilot/chunking.py` first splits at Markdown/reStructuredText headings, then
splits oversized sections into at most 180 embedding-token pieces. This leaves room
for the model's special tokens. These are the embedding tokenizer's tokens, not
necessarily the same tokens used by the hosted language model.

```python
offsets = tokenizer(section, add_special_tokens=False,
                    return_offsets_mapping=True)["offset_mapping"]
```

Offsets identify character positions in the original text. We cut at those
positions to preserve code/punctuation instead of decoding a lossy token string.
Every chunk retains its document's metadata, an index and a stable content-derived
ID. There is no overlap in this first implementation: a fact crossing a boundary
can still be harder to retrieve. That is a real limitation to measure later.

Checkpoint: the chunk test checks size, identity and provenance. Explain back:
what can go wrong if a paragraph contains more tokens than the embedding model accepts?

## 3. Embeddings: text becomes numbers

An embedding model maps text to a fixed-length list of numbers, a **vector**.
It is trained so related text tends to have related vectors. It does not produce
an answer or a source citation. Both document chunks and questions must use the
same model; otherwise their coordinates may mean different things.

In `repopilot/embeddings.py`:

```python
model.encode(texts, normalize_embeddings=True)
```

Input: a list of strings. Output: one 384-dimensional vector per string.
Normalization makes each vector length one. Cosine similarity compares direction;
for normalized vectors it equals their dot product. A higher score means more
similar according to this model, not a probability that an answer is correct.

We use one small CPU model, `all-MiniLM-L6-v2`, pinned to a downloaded revision.
`lru_cache` keeps the loaded model in the Python process; it avoids reloading weights
on each request. This is model-object reuse, not persistent conversational memory.
The disk cache avoids downloading weights again after a process restart.

Checkpoint: compare retrieval for “install” and “set up the package”. Explain back:
why can a semantic search find related words that do not exactly match the question?

## 4. Qdrant: store and search vectors

A naive search computes cosine similarity against every saved vector in a Python
list. Qdrant supplies a vector database: vectors plus **payloads** (chunk text and
metadata), persistent storage and nearest-neighbour search.

In `repopilot/store.py`, a Qdrant point stores:

```python
models.PointStruct(id=chunk.chunk_id, vector=vector,
                   payload=chunk.model_dump())
```

At query time, `query_points` asks for the K nearest vectors. K is the evidence
count, bounded to 1–10 in the API. We always send a repository filter and optionally
a document-type filter. A collection alias also identifies one repository. These
boundaries prevent another indexed repository's passages entering the answer.
This is local application isolation, not multi-user authentication/authorization.

Re-indexing uploads a new collection and then switches a stable **alias**, a name
pointing to the current collection. Qdrant switches that name atomically, so a
failed embedding/upload cannot replace the previous complete index. Queries see
an old or new completed generation. A lost network acknowledgement at publication
can leave completion uncertain; the error says so rather than claiming a rollback.
Old generations are removed after success; interrupted staging collections can
remain until manually cleaned up. This is deliberately small local-app machinery.

Compose's `qdrant_data` volume stores the index; `embedding_cache` stores model
weights. `docker compose down` preserves both. Avoid `down -v` unless you intend
to delete those saved volumes.

Checkpoint: the same replacement/isolation tests run with Qdrant's local engine or
real Docker server. Explain back: why upload first and switch the alias afterward?

## 5. RAG and citations: retrieve before generating

`repopilot/rag.py` shows the whole request as ordinary function calls:

```python
query_vector = vectors_for([request.question])[0]
evidence = VectorStore().search(repository, query_vector, request.top_k)
result = answer_question(request.question, evidence)
```

The real function also accepts an optional type filter. The model now sees a small
selected evidence set rather than the entire corpus. It may miss an answer if
retrieval misses the right passage; generating better prose cannot repair missing
evidence. Inspect the evidence expander before blaming the language model.

`repopilot/answers.py` requests structured claims, each with supporting chunk IDs.
Python rejects missing or unknown citations and allows only one correction attempt.
The UI resolves source URLs from stored chunks, never from model-generated links.
Missing evidence produces visible uncertainty; a filter returning no chunks abstains
without calling the hosted model. Repository/issue text remains untrusted data.

A valid ID proves that the source was supplied. It does **not** prove that the
source supports the claim. A model can cite a real but irrelevant passage. The
code validates reference integrity; the evaluation separately inspects factual
support. There are no GitHub write tools or agents in V1.

Checkpoint: ask an installation question with a metadata-only filter and inspect
whether evidence is adequate. Explain back: what is the difference between a valid
citation and a supporting citation?

## 6. Evaluation: distinguish retrieval from answer quality

```sh
python -m pytest -q
# Run the storage contracts against the real Qdrant service:
QDRANT_TEST_URL=http://localhost:6333 python -m pytest -q tests/test_v1.py
# Real local embeddings + Qdrant; no hosted-model cost:
python -m evals.v1_evaluate --output evals/results/my-v1-retrieval
# Also run paired live answer experiments (120 model calls across 24 cases):
python -m evals.v1_evaluate --live --output evals/results/my-v1-live
```

The runner creates a new evidence directory and refuses to overwrite an old one.
Its dataset is explicitly synthetic; fixture GitHub URLs are identifiers, not real
sources. It has 24 questions with expected facts and relevant-document labels.
For each chunk configuration, the evaluator records actual relevant chunk IDs.

**Recall@K = relevant chunks retrieved / relevant chunks available.** It asks whether
the right evidence was retrieved, not whether the generated answer is correct.
The runner compares 60/180-token chunks, K=3/5/10, two prompts, and a direct-README
baseline on the same questions/model. The direct baseline gets the fixture README;
RAG also has the fixture documentation/releases/issues. This tests the value of
expanded evidence access, not a universal superiority of vectors over full context.

Raw model outputs and manual-review slots are saved in `answers.jsonl`. Review:
correct answer? citation actually supports it? unsupported claims? Then inspect
`docs/V1_RESULTS.md` for this session's measured results and limitations. A high
score on this tiny controlled dataset is not a production accuracy estimate.
Generation timings use already-retrieved context; cold download/indexing time is
not included. The live API's elapsed time includes retrieval and generation.
Cost estimates use recorded usage and documented rates, not guessed billing totals.

The real HTTPX snapshot and answers are saved separately under `evals/results/v1`.
To capture another public-source smoke result in a new folder:

```sh
python -m evals.v1_smoke --output evals/results/my-httpx-smoke
```

## Where to read the code

Follow `ingestion.py` → `chunking.py` → `embeddings.py` → `store.py` first, then
`rag.py` → `answers.py` → API/UI. `rag_models.py` defines their shared data shapes.
These are functions and one small storage class, without an orchestration framework.

V1's deliberate limitations: bounded source selection, one embedding model, simple
chunks/no overlap, no reranking, no incremental refresh, no semantic truth validator,
and local single-user deployment. V2 can add typed read-only investigation tools
and a bounded agent only after you review and understand this version.

Official references: [Sentence Transformers embeddings](https://www.sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html),
[Qdrant quickstart](https://qdrant.tech/documentation/quickstart/),
[Qdrant filtering](https://qdrant.tech/documentation/search/filtering/),
[OpenAI structured output](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses).
