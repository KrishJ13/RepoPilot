"""The API coordinates fetching and summarising; the UI only calls this API."""

from time import perf_counter
from contextlib import contextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAIError, RateLimitError

from repopilot.github import fetch_repository
from repopilot.llm import summarise
from repopilot.models import SummaryRequest, SummaryResponse
from repopilot.rag_models import AskRequest, AskResponse, IndexResponse
from repopilot.rag import index_repository, ask_repository, EmbeddingUnavailable
from repopilot.store import StoreUnavailable

load_dotenv()
app = FastAPI(title="RepoPilot V1", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@contextmanager
def upstream_errors():
    """Use the same safe service-error messages for all three endpoints."""
    try:
        yield
    except (StoreUnavailable, EmbeddingUnavailable) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except httpx.HTTPStatusError as error:
        message = "GitHub rejected the request. Check the URL, token or rate limit."
        raise HTTPException(status_code=502, detail=message) from error
    except RateLimitError as error:
        # OpenAI uses HTTP 429 for both exhausted credits and temporary rate limits.
        # Inspect identifiers, not the raw message (which may contain secrets).
        if error.code in {"insufficient_quota", "credit_balance_exhausted"} or error.type == "insufficient_quota":
            message = "OpenAI API credits/quota are exhausted. Check API billing and project limits, then try again."
            raise HTTPException(status_code=503, detail=message) from error
        raise HTTPException(status_code=429, detail="OpenAI request rate limit reached. Wait briefly and try again.") from error
    except httpx.RequestError as error:
        raise HTTPException(status_code=502, detail="Cannot connect to GitHub. Check connectivity and try again.") from error
    except OpenAIError as error:
        # Don't echo upstream exceptions: they may contain request/credential details.
        raise HTTPException(status_code=502, detail="OpenAI request failed. Check the API key, model access and connectivity.") from error


@app.post("/api/v1/repositories/summary", response_model=SummaryResponse)
def repository_summary(request: SummaryRequest):
    started = perf_counter()
    with upstream_errors():
        result = summarise(fetch_repository(request.repository_url))
        result.elapsed_seconds = round(perf_counter() - started, 3)
        return result


@app.post("/api/v1/repositories/index", response_model=IndexResponse)
def repository_index(request: SummaryRequest):
    with upstream_errors():
        return index_repository(request.repository_url)


@app.post("/api/v1/repositories/ask", response_model=AskResponse)
def repository_ask(request: AskRequest):
    with upstream_errors():
        return ask_repository(request)
