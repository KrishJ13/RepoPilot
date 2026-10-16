"""Pydantic turns untrusted JSON into named, validated Python fields."""

from pydantic import BaseModel, ConfigDict, field_validator

from repopilot.github import parse_repository_url


class SummaryRequest(BaseModel):
    repository_url: str

    @field_validator("repository_url")
    @classmethod
    def public_github_url(cls, value: str) -> str:
        owner, repository = parse_repository_url(value)
        return f"https://github.com/{owner}/{repository}"


class RepositorySummary(BaseModel):
    # Reject unexpected fields instead of silently accepting a different shape.
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    purpose: str
    main_features: list[str]
    technologies: list[str]
    how_to_run: str
    limitations: list[str]


class SummaryResponse(BaseModel):
    summary: RepositorySummary
    repository_url: str
    readme_url: str | None
    readme_truncated: bool
    model: str
    prompt_version: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float
