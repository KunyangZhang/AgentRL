"""Audit paired outcomes; report sampling uncertainty without inventing significance."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random


def wilson(correct, n):
    z = 1.96
    p = correct/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def analyze(path):
    path = Path(path)
    stages = ('base', 'prompt-only', 'sft', 'grpo')
    rows = {s: [json.loads(line) for line in (path/f'{s}-test.jsonl').read_text().splitlines()] for s in stages}
    ids = [x['uid'] for x in rows['base']]
    assert len(ids) == len(set(ids))
    assert all([x['uid'] for x in batch] == ids for batch in rows.values())
    def bucket(row):
        if row['success']:return 'correct'
        if row['truncated']:return 'context_budget'
        if not row['valid_json']:return 'invalid_protocol'
        if not row['submitted']:return 'no_submission'
        if not row['executable']:return 'sql_execution_error'
        return 'wrong_result'
    result = {'n':len(ids), 'scope':'Single seed and filtered single-database execution metric; intervals do not capture training-seed uncertainty.', 'stages':{}}
    for stage,batch in rows.items():
        result['stages'][stage] = {
            'failure_buckets':dict(Counter(bucket(r) for r in batch)),
            'execution_match_95pct_wilson':wilson(sum(r['success'] for r in batch),len(batch)),
            'mean_input_tokens':sum(sum(len(t['prompt_ids']) for t in r['turns']) for r in batch)/len(batch),
            'mean_action_tokens':sum(sum(len(t['action_ids']) for t in r['turns']) for r in batch)/len(batch)}
    result['paired_comparisons'] = {}
    for previous,current in [('prompt-only','sft'),('sft','grpo')]:
        diffs = [int(b['success'])-int(a['success']) for a,b in zip(rows[previous],rows[current])]
        rng = random.Random(20260926)
        bootstrap = sorted(sum(rng.choices(diffs,k=len(diffs)))/len(diffs) for _ in range(10000))
        result['paired_comparisons'][f'{current}_minus_{previous}'] = {
            'difference':sum(diffs)/len(diffs), 'bootstrap_95pct':[bootstrap[249],bootstrap[9749]],
            'improved_questions':diffs.count(1), 'regressed_questions':diffs.count(-1)}
    updates = [json.loads(line) for line in (path/'grpo.jsonl').read_text().splitlines()]
    result['grpo_training'] = {'logged_updates':len(updates),
        'zero_reward_variance_groups':sum(x.get('reward_std',0)<1e-6 for x in updates),
        'nonzero_reward_variance_groups':sum(x.get('reward_std',0)>=1e-6 for x in updates),
        'sampled_trajectories':4*len(updates)}
    (path/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run')
    analyze(p.parse_args().run)
