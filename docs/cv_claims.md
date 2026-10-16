# RepoPilot evidence and claim boundaries

2026-10-16: V1 implemented; V2 not implemented.

Supported: built a local FastAPI/Streamlit repository assistant using Sentence
Transformers embeddings and Qdrant to retrieve repository-scoped evidence and
return structured cited answers. Real HTTPX ingestion/Q&A and local tests are saved.

Measured narrowly: 24 hand-authored synthetic questions retrieved expected evidence
at Recall@5=1.0, with 24/24 expected-fact RAG answers in this fixture. See
[V1 results](V1_RESULTS.md) for evidence, baseline design and limitations.
Do not present this as authentic repository accuracy or broad production quality.

Not supported: production scale/latency, general hallucination elimination,
prompt-injection resistance, tool-using agent or issue-triage capability.
No placeholder X% or future V2 claim should be presented as implemented.
