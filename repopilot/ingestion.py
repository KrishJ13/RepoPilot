"""A bounded public GitHub snapshot. V0's README-only fetch remains unchanged."""

import base64
import hashlib
import json
import os
from urllib.parse import quote

import httpx

from repopilot.github import parse_repository_url
from repopilot.rag_models import Document

MAX_DOCS = 8
MAX_FILE_BYTES = 40_000
MAX_ISSUES = 5
MAX_RELEASES = 3


def ingest(url: str) -> tuple[list[Document], list[str]]:
    owner, name = parse_repository_url(url)
    repository = f"{owner}/{name}".lower()
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.getenv("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    documents, warnings = [], []

    def add(kind, title, text, source, **metadata):
        if not text.strip():
            return
        # IDs describe source identity, not a model's invented references.
        document_id = hashlib.sha256(f"{repository}:{kind}:{source}".encode()).hexdigest()[:24]
        documents.append(Document(
            document_id=document_id, repository=repository, document_type=kind,
            title=title, text=text, source_url=source, metadata=metadata,
        ))

    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=20) as client:
        def get(path, **params):
            response = client.get(path, params=params)
            response.raise_for_status()  # A service failure must not look like complete indexing.
            return response.json()

        data = get(f"/repos/{repository}")
        if data.get("private"):
            raise ValueError("V1 supports public repositories only.")
        # Resolve a commit first so files and their source links share a stable snapshot.
        commit = get(f"/repos/{repository}/commits/{quote(data['default_branch'], safe='')}")["sha"]
        tree = get(f"/repos/{repository}/git/trees/{commit}", recursive="1")
        if tree.get("truncated"):
            raise ValueError("GitHub's file listing was truncated; this repository is too large for V1 ingestion.")
        files = [item for item in tree["tree"] if item["type"] == "blob"]
        readmes = [item for item in files if "/" not in item["path"] and item["path"].lower().startswith("readme")]
        docs = [item for item in files if item["path"].startswith(("docs/", "doc/")) and item["path"].endswith((".md", ".rst", ".txt"))]
        docs.sort(key=lambda item: (0 if any(word in item["path"].lower() for word in ["quickstart", "index", "getting-started"]) else 1, item["path"]))
        if len(docs) > MAX_DOCS:
            warnings.append(f"Selected {MAX_DOCS} of {len(docs)} documentation files (V1 limit).")
        if not readmes:
            warnings.append("No root README found; using other available sources.")
        for item in readmes[:1] + docs[:MAX_DOCS]:
            path = item["path"]
            if item.get("size", 0) > MAX_FILE_BYTES:
                warnings.append(f"Skipped {path}: exceeds {MAX_FILE_BYTES} bytes.")
                continue
            blob = get(f"/repos/{repository}/git/blobs/{item['sha']}")
            text = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
            add("readme" if item in readmes else "documentation", path, text,
                f"https://github.com/{repository}/blob/{commit}/{quote(path)}", path=path, commit=commit)
        add("metadata", data["full_name"], json.dumps({
            "name": data["full_name"], "description": data.get("description"),
            "language": data.get("language"), "topics": data.get("topics", []),
        }), data["html_url"], commit=commit)
        releases = get(f"/repos/{repository}/releases", per_page=MAX_RELEASES)
        for release in releases:
            body = release.get("body") or ""
            if len(body.encode()) > MAX_FILE_BYTES:
                warnings.append(f"Skipped oversized release {release['tag_name']}.")
                continue
            add("release", release.get("name") or release["tag_name"], body,
                release["html_url"], release_version=release["tag_name"], updated_at=release.get("published_at"))
        # The issues endpoint also includes PRs. Exclude them explicitly.
        issues = get(f"/repos/{repository}/issues", state="all", per_page=MAX_ISSUES * 2, sort="updated")
        for issue in [item for item in issues if "pull_request" not in item][:MAX_ISSUES]:
            text = issue["title"] + "\n" + (issue.get("body") or "")
            if len(text.encode()) > MAX_FILE_BYTES:
                warnings.append(f"Skipped oversized issue #{issue['number']}.")
                continue
            add("issue", issue["title"], text, issue["html_url"],
                issue_number=issue["number"], updated_at=issue["updated_at"])
    warnings.append("Bounded snapshot: at most 8 docs, 3 releases and 5 issues; issue/release records may change after indexing.")
    return documents, warnings
