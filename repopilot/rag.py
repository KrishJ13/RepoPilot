"""The whole V1 pipeline, written as a short sequence of ordinary functions."""

from time import perf_counter

from repopilot.answers import answer_question
from repopilot.chunking import chunk_documents
from repopilot.embeddings import MODEL_NAME, embed, embedding_model
from repopilot.ingestion import ingest
from repopilot.rag_models import AskRequest, IndexResponse
from repopilot.store import VectorStore, repository_key


class EmbeddingUnavailable(RuntimeError):
    pass


def vectors_for(texts):
    try:
        return embed(texts)
    except Exception as error:
        raise EmbeddingUnavailable("Embedding model could not run. Check the first-download connection and model cache.") from error


def index_repository(url: str) -> IndexResponse:
    documents, warnings = ingest(url)
    try:
        model = embedding_model()
        chunks = chunk_documents(documents, model.tokenizer)
    except Exception as error:
        raise EmbeddingUnavailable("Could not load the embedding tokenizer/model.") from error
    vectors = vectors_for([chunk.text for chunk in chunks])
    repository = repository_key(url)
    VectorStore().replace(repository, chunks, vectors)
    return IndexResponse(repository=repository, documents=len(documents), chunks=len(chunks),
                         warnings=warnings, embedding_model=MODEL_NAME)


def ask_repository(request: AskRequest):
    started = perf_counter()
    query_vector = vectors_for([request.question])[0]
    evidence = VectorStore().search(repository_key(request.repository_url), query_vector,
                                    request.top_k, request.document_type)
    result = answer_question(request.question, evidence)
    result.elapsed_seconds = round(perf_counter() - started, 3)
    return result
