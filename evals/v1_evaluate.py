"""Small V1 experiments: real embeddings/Qdrant, optional live model answers."""

import argparse
import json
import os
import platform
from pathlib import Path
from time import perf_counter
from datetime import datetime, timezone

from dotenv import load_dotenv
from qdrant_client import QdrantClient

from repopilot.answers import answer_question
from repopilot.chunking import chunk_documents
from repopilot.embeddings import MODEL_NAME, MODEL_REVISION, embedding_model, embed
from repopilot.rag_models import Chunk, Document, Evidence
from repopilot.store import VectorStore

DATASET = Path(__file__).parent / 'v1/dataset.json'


def recall(relevant_ids, retrieved_ids):
    return len(set(relevant_ids) & set(retrieved_ids)) / len(set(relevant_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; never overwrite prior runs')
    parser.add_argument('--live', action='store_true', help='Call the hosted LLM for paired answer experiments')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    load_dotenv('.env')
    dataset = json.loads(DATASET.read_text())
    documents = [Document.model_validate(item) for item in dataset['documents']]
    cases = dataset['cases']
    model = embedding_model()
    query_vectors = embed([case['question'] for case in cases])
    retrieval_records, contexts = [], {}
    for budget in [60, 180]:
        chunks = chunk_documents(documents, model.tokenizer, budget)
        vectors = embed([chunk.text for chunk in chunks])
        client = QdrantClient(':memory:')
        store = VectorStore(client)
        store.replace(documents[0].repository, chunks, vectors)
        for case, vector in zip(cases, query_vectors):
            relevant = [chunk.chunk_id for chunk in chunks if chunk.document_id in case['relevant_document_ids'] and case['expected_span'] in chunk.text]
            if not relevant:
                raise ValueError('Expected evidence was split/absent; label the new chunks before scoring: ' + case['id'])
            started = perf_counter()
            evidence = store.search(documents[0].repository, vector, 10)
            ids = [item.chunk.chunk_id for item in evidence]
            retrieval_records.append({
                'case_id':case['id'], 'chunk_tokens':budget, 'total_chunks':len(chunks),
                'relevant_chunk_ids':relevant, 'retrieved_chunk_ids':ids,
                'recall_at_5':recall(relevant,ids[:5]), 'recall_at_10':recall(relevant,ids),
                'search_seconds':round(perf_counter()-started,4),
            })
            if budget == 180:
                contexts[case['id']] = evidence
        client.close()
    retrieval_path = args.output / 'retrieval.json'
    retrieval_path.write_text(json.dumps(retrieval_records,indent=2))
    (args.output/'dataset.json').write_text(DATASET.read_text())
    metadata = {
        'created_at':datetime.now(timezone.utc).isoformat(), 'provenance':dataset['provenance'],
        'python':platform.python_version(), 'platform':platform.platform(),
        'embedding_model':MODEL_NAME, 'embedding_revision':MODEL_REVISION,
        'qdrant':'real in-process Qdrant client engine, cosine; server contracts tested separately',
        'model':os.getenv('OPENAI_MODEL','gpt-4o-mini'), 'temperature':0,
        'experiments':'EXP-001/002/003/004', 'quality_grading':'Manual review required; exact-text checks are not quality scores',
    }
    (args.output/'settings.json').write_text(json.dumps(metadata,indent=2))
    for budget in [60,180]:
        selected=[item for item in retrieval_records if item['chunk_tokens']==budget]
        print('Tokens',budget,'Recall@5',sum(item['recall_at_5'] for item in selected)/len(selected),
              'Recall@10',sum(item['recall_at_10'] for item in selected)/len(selected),flush=True)
    if not args.live:
        return
    readme = next(doc for doc in documents if doc.document_type=='readme')
    direct = [Evidence(chunk=Chunk(**readme.model_dump(),chunk_id='direct-readme',chunk_index=0),score=1)]
    # Same questions/model/schema. Only evidence selection, K, or prompt changes.
    variants=[('direct_readme',None,'v1-answer-1'), ('rag_k5',5,'v1-answer-1'),
              ('rag_k3',3,'v1-answer-1'), ('rag_k10',10,'v1-answer-1'), ('rag_prompt2',5,'v1-answer-2')]
    with (args.output/'answers.jsonl').open('x') as output:
        for label,k,prompt in variants:
            for case in cases:
                evidence = direct if k is None else contexts[case['id']][:k]
                try:
                    result=answer_question(case['question'], evidence, prompt_version=prompt)
                    record={'variant':label,'case_id':case['id'],'expected_answer':case['expected_answer'],
                            'response':result.model_dump(), 'correct':None, 'citations_support_claims':None,
                            'unsupported_claims':None, 'notes':'Awaiting manual review'}
                except Exception as error:
                    # Only an exception class, never a credential-bearing upstream message.
                    record={'variant':label,'case_id':case['id'],'error_type':type(error).__name__}
                output.write(json.dumps(record)+'\n')
                output.flush()  # Keep evidence even if a later request fails.
                print(label,case['id'],'saved',flush=True)


if __name__ == '__main__':
    main()
