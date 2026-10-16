"""Aggregate preserved retrieval results and explicit human/agent manual reviews."""

import argparse
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run/'answers.jsonl').read_text().splitlines()]
    reviews = json.loads((args.run/'reviews.json').read_text())
    review_map = {(item['variant'],item['case_id']):item for item in reviews['reviews']}
    metrics = {'review_method':reviews['method'], 'variants':{}, 'retrieval':{}}
    for label in sorted({item['variant'] for item in rows}):
        selected = [item for item in rows if item['variant']==label]
        valid = [item for item in selected if 'response' in item]
        reviewed = [review_map[(item['variant'],item['case_id'])] for item in selected]
        input_tokens=sum(item['response']['input_tokens'] for item in valid)
        output_tokens=sum(item['response']['output_tokens'] for item in valid)
        # Snapshot of published gpt-4o-mini list rates on 2026-10-16.
        # Ignore cache discounts/taxes: this is an estimate, never an invoice.
        cost=(input_tokens*.15+output_tokens*.60)/1_000_000 if all(item['response']['model']=='gpt-4o-mini' for item in valid) else None
        metrics['variants'][label]={
            'cases':len(selected), 'valid_responses':len(valid),
            'correct':sum(item['correct'] for item in reviewed),
            'responses_with_unsupported_claims':sum(item['unsupported_claims'] for item in reviewed),
            'responses_with_claims':sum(bool(item['response']['answer']['claims']) for item in valid),
            'responses_with_supporting_citations':sum(item['citations_support_claims'] is True for item in reviewed),
            'mean_generation_seconds':round(mean(item['response']['elapsed_seconds'] for item in valid),3),
            'input_tokens':input_tokens, 'output_tokens':output_tokens,
            'estimated_undiscounted_usd':round(cost,6) if cost is not None else None,
        }
    retrieval = json.loads((args.run/'retrieval.json').read_text())
    for size in sorted({item['chunk_tokens'] for item in retrieval}):
        selected=[item for item in retrieval if item['chunk_tokens']==size]
        metrics['retrieval'][str(size)]={
            'chunks':selected[0]['total_chunks'], 'cases':len(selected),
            'mean_recall_at_5':mean(item['recall_at_5'] for item in selected),
            'mean_recall_at_10':mean(item['recall_at_10'] for item in selected),
        }
    metrics['pricing']={'date':'2026-10-16','input_usd_per_million':.15,'output_usd_per_million':.60,
                        'source':'https://developers.openai.com/api/docs/models/gpt-4o-mini',
                        'limitation':'Undiscounted list-price estimate. Excludes local compute, cache discounts, taxes and failures without usage.'}
    with (args.run/'metrics.json').open('x') as output:
        json.dump(metrics,output,indent=2)
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
