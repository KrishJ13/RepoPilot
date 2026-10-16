"""One real public-repository snapshot and cited answer (uses one LLM request)."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from repopilot.ingestion import ingest
from repopilot.chunking import chunk_documents
from repopilot.embeddings import embedding_model, embed
from repopilot.store import VectorStore
from repopilot.answers import answer_question


def main():
    load_dotenv('.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory')
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=False)
    documents, warnings = ingest('https://github.com/encode/httpx')
    (output/'httpx-snapshot.json').write_text(json.dumps({'documents':[d.model_dump() for d in documents], 'warnings':warnings},indent=2))
    print('Documents:', len(documents), [d.title for d in documents], flush=True)
    chunks=chunk_documents(documents,embedding_model().tokenizer)
    store=VectorStore()
    store.replace('encode/httpx',chunks,embed([c.text for c in chunks]))
    evidence=store.search('encode/httpx',embed(['How do I install HTTPX?'])[0])
    answer=answer_question('How do I install HTTPX?',evidence)
    with (output/'httpx-live-answer.json').open('x') as f:
        f.write(answer.model_dump_json(indent=2))
    print('Chunks:',len(chunks), flush=True)
    print(answer.answer.model_dump(), flush=True)


if __name__ == '__main__':
    main()
