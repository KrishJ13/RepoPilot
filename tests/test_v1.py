"""Small checks for V1's main guarantees, using Qdrant's real local engine."""

import base64
import json
import os
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from repopilot import api, ingestion, rag
from repopilot.answers import answer_question, validate_citations
from repopilot.chunking import chunk_documents
from repopilot.rag_models import AskRequest, Chunk, CitedClaim, Document, Evidence, GroundedAnswer
from repopilot.store import VectorStore, StoreUnavailable, alias_for


@pytest.fixture
def tokenizer():
    tiny = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tiny.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tiny)


def document(repository="tests/a", text="# Install\npip install demo\n# Run\npython -m demo"):
    return Document(document_id="doc", repository=repository, document_type="documentation", title="Guide",
                    text=text, source_url="https://github.com/tests/a/blob/abc/docs/guide.md")


def make_chunk(repository, text="Evidence", index=0):
    return Chunk(**document(repository, text).model_dump(), chunk_id=uuid4().hex, chunk_index=index)


@pytest.fixture
def store(monkeypatch):
    # Set QDRANT_TEST_URL to run these SAME contracts against the Docker server.
    client = QdrantClient(url=os.environ["QDRANT_TEST_URL"]) if os.getenv("QDRANT_TEST_URL") else QdrantClient(":memory:")
    store = VectorStore(client)
    created = set()
    original_create = client.create_collection
    def track_create(collection_name, **kwargs):
        created.add(collection_name)
        return original_create(collection_name, **kwargs)
    monkeypatch.setattr(client, "create_collection", track_create)
    yield store
    # Remove only this test's generated collections, never pre-existing app data.
    for item in client.get_aliases().aliases:
        if item.collection_name in created:
            from qdrant_client import models
            client.update_collection_aliases([models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=item.alias_name))])
    for item in client.get_collections().collections:
        if item.name in created:
            client.delete_collection(item.name)
    client.close()


def test_chunks_keep_text_provenance_and_token_bound(tokenizer):
    source = document(text="# Install\n" + "word " * 40 + "\n# Run\npython -m demo")
    chunks = chunk_documents([source], tokenizer, max_tokens=8)
    assert len(chunks) > 3
    assert all(len(tokenizer.encode(chunk.text, add_special_tokens=False)) <= 8 for chunk in chunks)
    assert all(chunk.source_url == source.source_url and chunk.document_id == "doc" for chunk in chunks)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunk_documents([source], tokenizer, 8)]


def test_qdrant_isolation_filter_and_replacement(store):
    a, b = "tests/" + uuid4().hex, "tests/" + uuid4().hex
    old, other = make_chunk(a), make_chunk(b, "Wrong repository")
    store.replace(a, [old], [[1.0, 0.0]])
    store.replace(b, [other], [[1.0, 0.0]])
    assert [x.chunk.chunk_id for x in store.search(a, [1.0, 0.0])] == [old.chunk_id]
    assert store.search(a, [1.0, 0.0], document_type="issue") == []
    new = make_chunk(a, "Updated content")
    store.replace(a, [new], [[0.9, 0.1]])
    assert [x.chunk.chunk_id for x in store.search(a, [1.0, 0.0])] == [new.chunk_id]


def test_failed_upload_preserves_old_index(store, monkeypatch):
    repository = "tests/" + uuid4().hex
    old = make_chunk(repository)
    store.replace(repository, [old], [[1.0, 0.0]])
    monkeypatch.setattr(store.client, "upsert", Mock(side_effect=RuntimeError("service failed")))
    with pytest.raises(StoreUnavailable):
        store.replace(repository, [make_chunk(repository)], [[0.0, 1.0]])
    assert store.search(repository, [1.0, 0.0])[0].chunk.chunk_id == old.chunk_id


def test_unindexed_repository_fails_visibly(store):
    with pytest.raises(ValueError, match="not indexed"):
        store.search("tests/" + uuid4().hex, [1.0, 0.0])


def test_citations_and_uncertainty():
    evidence = [Evidence(chunk=make_chunk("tests/a"), score=1)]
    answer = GroundedAnswer(claims=[CitedClaim(text="Invented source", chunk_ids=["fake"])], insufficient_evidence=False, missing_information=[])
    with pytest.raises(ValueError, match="citations"):
        validate_citations(answer, evidence)
    result = answer_question("Unknown?", [])
    assert result.answer.insufficient_evidence and result.answer.missing_information
    assert result.input_tokens == 0


def test_invalid_citations_are_retried_then_rejected():
    output = json.dumps({"claims": [{"text": "Unsupported", "chunk_ids": ["invented"]}], "insufficient_evidence": False, "missing_information": []})
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text=output, usage=None))))
    with pytest.raises(ValueError, match="two attempts"):
        answer_question("Question?", [Evidence(chunk=make_chunk("tests/a"), score=1)], client)
    assert client.responses.create.call_count == 2


def test_index_and_ask_api_flow(store, tokenizer, monkeypatch):
    repository = "tests/" + uuid4().hex
    monkeypatch.setattr(rag, "ingest", lambda url: ([document(repository)], []))
    monkeypatch.setattr(rag, "embedding_model", lambda: SimpleNamespace(tokenizer=tokenizer))
    monkeypatch.setattr(rag, "vectors_for", lambda texts: [[1.0, 0.0] for _ in texts])
    monkeypatch.setattr(rag, "VectorStore", lambda: store)
    def respond(question, evidence):
        return answer_question(question, evidence, SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
            output_text=json.dumps({"claims": [{"text": "Install demo", "chunk_ids": [evidence[0].chunk.chunk_id]}],
                                    "insufficient_evidence": False, "missing_information": []}), usage=None))))
    monkeypatch.setattr(rag, "answer_question", respond)
    client = TestClient(api.app)
    assert client.post("/api/v1/repositories/index", json={"repository_url": "https://github.com/" + repository}).status_code == 200
    response = client.post("/api/v1/repositories/ask", json={"repository_url": "https://github.com/" + repository, "question": "Install?"})
    assert response.status_code == 200
    assert response.json()["answer"]["claims"][0]["chunk_ids"]
    assert client.post("/api/v1/repositories/ask", json={"repository_url": "https://github.com/" + repository, "question": "?", "top_k": 99}).status_code == 422


def test_ingestion_pins_files_and_excludes_prs(monkeypatch):
    def respond(request):
        path = request.url.path
        if path.endswith("/commits/main"):
            body = {"sha": "commit123"}
        elif path.endswith("/git/trees/commit123"):
            body = {"tree": [{"type": "blob", "path": "README.md", "sha": "blob1", "size": 10}, {"type": "blob", "path": "docs/start.md", "sha": "blob2", "size": 10}]}
        elif "/git/blobs/" in path:
            body = {"content": base64.b64encode(b"# Demo\nPublic docs").decode()}
        elif path.endswith("/releases"):
            body = []
        elif path.endswith("/issues"):
            body = [{"pull_request": {}, "title": "Not an issue"}]
        else:
            body = {"full_name": "tests/a", "default_branch": "main", "html_url": "https://github.com/tests/a"}
        return httpx.Response(200, json=body)
    original = httpx.Client
    monkeypatch.setattr(ingestion.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    docs, warnings = ingestion.ingest("https://github.com/tests/a")
    assert {d.document_type for d in docs} == {"readme", "documentation", "metadata"}
    assert all("/blob/commit123/" in d.source_url for d in docs if d.document_type != "metadata")


def test_ui_renders_cited_answer(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    chunk = make_chunk('tests/a')
    payload = {'answer': {'claims': [{'text': 'A supported answer', 'chunk_ids': [chunk.chunk_id]}],
                          'insufficient_evidence': False, 'missing_information': []},
               'evidence': [{'chunk': chunk.model_dump(), 'score': 0.8}], 'model': 'test-model',
               'elapsed_seconds': 0, 'input_tokens': 10, 'output_tokens': 5}
    monkeypatch.setattr(httpx, 'post', lambda *args, **kwargs: httpx.Response(200, json=payload))
    page = AppTest.from_file(Path('repopilot/ui.py').resolve()).run()
    page.button[2].click().run()  # Analyse, Index repository, Ask
    assert not page.exception
    assert any(item.value == 'A supported answer' for item in page.markdown)


def test_qdrant_failure_is_visible_at_api(monkeypatch):
    monkeypatch.setattr(api, 'index_repository', Mock(side_effect=StoreUnavailable('Qdrant indexing failed; completion could not be confirmed. Retry indexing.')))
    response = TestClient(api.app).post('/api/v1/repositories/index', json={'repository_url':'https://github.com/tests/a'})
    assert response.status_code == 503
    assert 'Qdrant' in response.json()['detail']
