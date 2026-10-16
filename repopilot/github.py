"""Only fetch GitHub's API; never fetch an arbitrary user-supplied host."""

import base64
import os
import re
from urllib.parse import urlsplit

import httpx

MAX_README_CHARACTERS = 20_000


def parse_repository_url(url: str) -> tuple[str, str]:
    parsed = urlsplit(url.strip())
    parts = parsed.path.strip("/").split("/")
    if (
        parsed.scheme != "https" or parsed.netloc != "github.com"
        or parsed.query or parsed.fragment or len(parts) != 2
        or not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in parts)
        or any(part in {".", ".."} for part in parts)
    ):
        raise ValueError("Use https://github.com/owner/repository (a public repository).")
    owner, name = parts
    name = name.removesuffix(".git")
    if not name:
        raise ValueError("Repository name is missing.")
    return owner, name


def fetch_repository(url: str) -> dict:
    owner, name = parse_repository_url(url)
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    # A context manager closes network connections when this request finishes.
    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=15) as client:
        metadata = client.get(f"/repos/{owner}/{name}")
        if metadata.status_code == 404:
            raise ValueError("Repository not found. V0 supports public repositories only.")
        metadata.raise_for_status()
        data = metadata.json()
        if data.get("private"):
            raise ValueError("V0 supports public repositories only.")
        readme_response = client.get(f"/repos/{owner}/{name}/readme")
        readme, readme_url = "", None
        if readme_response.status_code != 404:
            readme_response.raise_for_status()
            document = readme_response.json()
            # GitHub wraps the file's bytes in base64 inside its JSON response.
            readme = base64.b64decode(document["content"]).decode("utf-8", errors="replace")
            readme_url = document["html_url"]
    return {
        "name": data["full_name"],
        "description": data.get("description"),
        "language": data.get("language"),
        "repository_url": data["html_url"],
        "readme_url": readme_url,
        # Simple character clipping is our V0 baseline, not token-aware retrieval.
        "readme": readme[:MAX_README_CHARACTERS],
        "readme_truncated": len(readme) > MAX_README_CHARACTERS,
    }
