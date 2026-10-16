"""Core checks: URL → GitHub bytes → prompt → validated response → API/UI."""

import base64
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAIError, RateLimitError
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from evals.run import demo_client
from repopilot import api, github
from repopilot.llm import summarise
from repopilot.models import RepositorySummary


@pytest.fixture
def repository():
    return json.loads(open("evals/repository.json").read())


def test_github_url():
    assert github.parse_repository_url("https://github.com/pallets/flask.git/") == ("pallets", "flask")


@pytest.mark.parametrize("url", ["http://github.com/a/b", "https://evil.test/a/b", "https://github.com/a/b/issues", "https://github.com/a/.."])
def test_invalid_url(url):
    with pytest.raises(ValueError):
        github.parse_repository_url(url)


def test_github_readme_decoding(monkeypatch):
    real_client = httpx.Client
    def respond(request):
        if request.url.path.endswith("/readme"):
            return httpx.Response(200, json={"content": base64.b64encode(b"# Hello").decode(), "html_url": "https://github.com/a/b/blob/main/README.md"})
        return httpx.Response(200, json={"full_name": "a/b", "html_url": "https://github.com/a/b", "language": "Python", "description": "Demo"})
    monkeypatch.setattr(github.httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs))
    result = github.fetch_repository("https://github.com/a/b")
    assert result["readme"] == "# Hello"
    assert result["name"] == "a/b"


def test_schema_and_mocked_llm(repository):
    result = summarise(repository, client=demo_client())
    assert result.summary.name == repository["name"]
    assert result.llm_calls == 1
    with pytest.raises(ValidationError):
        RepositorySummary.model_validate_json('{"name":"incomplete"}')


def test_correction_is_bounded(repository):
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=SimpleNamespace(output_text="not JSON", usage=None))))
    with pytest.raises(ValueError, match="two attempts"):
        summarise(repository, client=client)
    assert client.responses.create.call_count == 2


def test_second_attempt_can_correct(repository):
    good = demo_client().responses.create()
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(side_effect=[SimpleNamespace(output_text="{}", usage=None), good])))
    assert summarise(repository, client=client).llm_calls == 2


def test_missing_context_is_visible(repository):
    repository["readme"] = ""
    result = summarise(repository, client=demo_client())
    assert result.summary.how_to_run == "Not stated in the supplied README."
    assert any("README missing" in text for text in result.summary.limitations)


def test_api_flow_and_input_validation(repository, monkeypatch):
    monkeypatch.setattr(api, "fetch_repository", lambda url: repository)
    monkeypatch.setattr(api, "summarise", lambda repo: summarise(repo, client=demo_client()))
    client = TestClient(api.app)
    response = client.post("/api/v1/repositories/summary", json={"repository_url": repository["repository_url"]})
    assert response.status_code == 200
    assert response.json()["summary"]["technologies"] == ["Python"]
    assert client.post("/api/v1/repositories/summary", json={"repository_url": "https://evil.test/a/b"}).status_code == 422


def test_api_provider_error_is_visible(repository, monkeypatch):
    monkeypatch.setattr(api, "fetch_repository", lambda url: repository)
    monkeypatch.setattr(api, "summarise", Mock(side_effect=OpenAIError("private diagnostic")))
    response = TestClient(api.app).post("/api/v1/repositories/summary", json={"repository_url": repository["repository_url"]})
    assert response.status_code == 502
    assert "private diagnostic" not in response.text


@pytest.mark.parametrize("code,error_type,status,expected", [
    ("credit_balance_exhausted", "insufficient_quota", 503, "credits/quota are exhausted"),
    ("insufficient_quota", "insufficient_quota", 503, "credits/quota are exhausted"),
    ("rate_limit_exceeded", "requests", 429, "Wait briefly"),
])
def test_api_distinguishes_quota_from_rate_limit(repository, monkeypatch, code, error_type, status, expected):
    import httpx2
    error = RateLimitError(
        "private diagnostic must not escape",
        response=httpx2.Response(429, request=httpx2.Request("POST", "https://api.openai.com/v1/responses")),
        body={"code": code, "type": error_type},
    )
    monkeypatch.setattr(api, "fetch_repository", lambda url: repository)
    monkeypatch.setattr(api, "summarise", Mock(side_effect=error))
    response = TestClient(api.app).post("/api/v1/repositories/summary", json={"repository_url": repository["repository_url"]})
    assert response.status_code == status
    assert expected in response.json()["detail"]
    assert "private diagnostic" not in response.text


def test_ui_displays_api_summary(repository, monkeypatch):
    result = summarise(repository, client=demo_client()).model_dump()
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: httpx.Response(200, json=result))
    page = AppTest.from_file(Path("repopilot/ui.py").resolve()).run()
    page.button[0].click().run()
    assert not page.exception
    assert page.subheader[0].value == repository["name"]


def test_real_sdk_serialization(repository):
    """Exercise the installed SDK's HTTP contract without calling OpenAI."""
    import httpx2
    from openai import OpenAI

    def respond(request):
        body = json.loads(request.content)
        assert body["text"]["format"]["schema"]["additionalProperties"] is False
        assert body["input"][0]["role"] == "system"
        assert "UNTRUSTED" in body["input"][1]["content"]
        return httpx2.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 0,
            "status": "completed", "model": "gpt-4o-mini",
            "output": [{"id": "msg_fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": demo_client().responses.create().output_text, "annotations": []}]}],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        })
    with OpenAI(api_key="test-placeholder", max_retries=0, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        result = summarise(repository, client=client)
    assert result.input_tokens == 100
    assert result.output_tokens == 50
