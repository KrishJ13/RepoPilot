"""The data that flows through ingestion, retrieval and cited answers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from repopilot.models import SummaryRequest


class Document(BaseModel):
    document_id: str
    repository: str
    document_type: str
    title: str
    text: str
    source_url: str
    metadata: dict = Field(default_factory=dict)


class Chunk(Document):
    chunk_id: str
    chunk_index: int


class IndexResponse(BaseModel):
    repository: str
    documents: int
    chunks: int
    warnings: list[str]
    embedding_model: str


class AskRequest(SummaryRequest):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    document_type: Literal["readme", "documentation", "release", "issue", "metadata"] | None = None


class CitedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    text: str
    chunk_ids: list[str]


class GroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    claims: list[CitedClaim]
    insufficient_evidence: bool
    missing_information: list[str]


class Evidence(BaseModel):
    chunk: Chunk
    score: float


class AskResponse(BaseModel):
    answer: GroundedAnswer
    evidence: list[Evidence]
    model: str
    prompt_version: str
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_seconds: float = 0
