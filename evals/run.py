"""Capture one summary, then inspect it against ten questions (not a Q&A API)."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

from repopilot.llm import summarise

FIXTURES = Path(__file__).parent


def demo_client():
    # Handwritten fake output exercises the plumbing without using money or a key.
    # It is explicitly not evidence that an LLM can answer these questions.
    summary = {
        "name": "learning/demo-notes", "purpose": "A command-line notes application",
        "main_features": ["Add notes", "List saved notes", "Store notes in a local JSON file"],
        "technologies": ["Python"],
        "how_to_run": "Install: pip install -e .\nRun: python -m demo_notes. Docker instructions are not stated.",
        "limitations": ["Single-user only", "No cloud synchronisation"],
    }
    return SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(
        output_text=json.dumps(summary), usage=None,
    )))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Use a handwritten fake LLM output; no model-quality measurement")
    parser.add_argument("--output", type=Path, required=True, help="New file path; existing evidence is never overwritten")
    args = parser.parse_args()
    load_dotenv()
    repository = json.loads((FIXTURES / "repository.json").read_text())
    questions = json.loads((FIXTURES / "questions.json").read_text())
    result = summarise(repository, client=demo_client() if args.demo else None)
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "handwritten-fake-output" if args.demo else "live-llm-on-synthetic-repository",
        "workload": "One synthetic README, ten manual review questions; not ten independent tasks",
        "source": repository, "response": result.model_dump(),
        "reviews": [dict(case, correct=None, unsupported_claims=None, notes="Awaiting manual review") for case in questions],
        "cost": None, "quality_score": None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:  # 'x' refuses to overwrite a saved baseline.
        json.dump(record, output, indent=2)
    print(f"Saved {args.output}. Review the ten questions; no quality score was assigned.")


if __name__ == "__main__":
    main()
