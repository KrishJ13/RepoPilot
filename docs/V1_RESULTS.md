# V1 measured results — 2026-10-16

This is a small learning baseline, not a production benchmark. Code is uncommitted
over `f308455`; source and environment are recorded with the run evidence.

## What ran

- Live OpenAI `gpt-4o-mini`, temperature 0, one request per question/variant.
- Pinned `all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, 384 normalized dimensions.
- 24 hand-authored questions, one synthetic repository, 25 source documents.
  Fixture URLs are identifiers, not real public sources. The known facts are
  deliberately in documentation/releases/issues, not the minimal README.
- Real local Qdrant engine for benchmark queries; real Docker Qdrant 1.19.0 for
  separate storage contracts and the authentic HTTPX end-to-end check.
- All 120 answer outputs and evidence are retained. Codex reviewed them against
  expected facts and cited passages; there was no independent second reviewer.

The direct baseline gets only the fixture README. RAG gets selected passages from
the expanded source set. This measures the benefit of access to that evidence,
not vectors versus a full-corpus prompt given identical information.

## Retrieval and chunk size (EXP-002)

| Embedding-token budget | Chunks | Questions | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| 60 | 26 | 24 | 1.000 | 1.000 |
| 180 (default) | 25 | 24 | 1.000 | 1.000 |

Labels are recorded as actual relevant chunk IDs for each configuration. Recall
uses the intersection of retrieved and relevant IDs, divided by relevant-ID count,
then averages over questions. The short fixture passages barely exercise chunk
size, so this result is not evidence that chunk size never matters. Keep the
180-token baseline; do not add complexity without a harder measured workload.

## Answer comparisons (EXP-001, EXP-003, EXP-004)

| Variant | Correct expected fact | Responses with unsupported claims | Supporting citations on answers with claims | Mean generation seconds | Input / output tokens | Estimated list-rate USD |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| Direct README, prompt 1 | 0/24 | 0/24 | 3/3 | 1.261 | 6,772 / 802 | 0.001497 |
| RAG K=3, prompt 1 | 24/24 | 0/24 | 24/24 | 1.291 | 12,310 / 1,501 | 0.002747 |
| RAG K=5, prompt 1 | 24/24 | 0/24 | 24/24 | 1.294 | 16,427 / 1,505 | 0.003367 |
| RAG K=10, prompt 1 | 24/24 | 0/24 | 24/24 | 1.227 | 27,221 / 1,504 | 0.004986 |
| RAG K=5, prompt 2 | 24/24 | 0/24 | 24/24 | 1.300 | 16,427 / 1,440 | 0.003328 |

All variants returned valid structured responses. The direct baseline correctly
expressed missing evidence on every case; three responses also contained supported
but non-answering statements. Its 0/24 measures required-fact answerability, not
unsafe behavior. Supporting-citation rates exclude claim-free abstentions.
All RAG claims were inspected against their supplied passages; citation IDs were
also checked by application code. Some answers include extra relevant supported facts.

Generation timings exclude embedding/retrieval, model download and indexing.
Contexts were prepared before model calls. These are single sequential trials;
small timing differences are not speed-improvement evidence. API elapsed timings
in the real smoke include retrieval and generation.

Costs estimate published `gpt-4o-mini` rates as of this date: $0.15/M input tokens
and $0.60/M output tokens, ignoring cache discounts, taxes and local compute.
They are not a billing receipt. [Pricing source](https://developers.openai.com/api/docs/models/gpt-4o-mini).
The five benchmark variants together are about $0.016 at these undiscounted rates.

K=3 was adequate on this tiny fixture and used fewer tokens. K=10 and prompt 2
showed no quality gain. Keep prompt 1 and K=5 as the simple general demo defaults;
more diverse repository questions are needed before changing retrieval defaults.

## Authentic repository and V0 checks

The real HTTPX snapshot contains 18 documents and 195 chunks: README, eight chosen
docs, metadata, three releases and five issues. File sources point to a commit.
Documentation selection is explicitly bounded (8 of 23 files); this is not whole-
repository coverage. The live installation answer and a Docker API question about
connection pool limits returned supporting source chunks. These are smoke examples,
not 24 independently measured authentic repository tasks.

A newly funded live V0 summary also ran against the original synthetic notes README.
Manual review of its ten expected-fact checks found the installation command omitted
(9/10 checks met). This V0 summary baseline is preserved separately; the paired V1
experiment uses a question-answer schema so the same questions can be compared fairly.

## Evidence and reproduction

- `evals/v1/dataset.json`: versioned synthetic questions, sources and expected facts.
- `evals/results/v1/experiment-1/`: raw answers, per-case retrieval labels/results,
  manual `reviews.json`, settings, dataset copy and aggregated `metrics.json`.
- `evals/results/v1/httpx-snapshot.json`, `httpx-live-answer.json`,
  `container-live-answer.json`: real source snapshot and answers.
- `evals/results/v0-live-baseline.json`, `v0-live-review.json`: retained V0 baseline.
- `evals/results/v1/environment.txt`: installed packages for this machine.

Run `python -m evals.v1_evaluate --live --output evals/results/new-run` to reproduce
requests in a fresh folder. Review answers into a separate `reviews.json` using
the saved review format, then run `python -m evals.v1_report --run evals/results/new-run`.
Report creation refuses to overwrite an existing metrics file. Model generation
can vary even at temperature 0; the hosted model alias is recorded, not guaranteed
immutable. The embedding revision and fixture sources are pinned.

No broad accuracy, injection-resistance, scalability or V2 agent claim follows
from these results. Next evaluation improvement: label diverse real-repository
questions including missing/ambiguous evidence, rather than optimizing this fixture.
