"""Inventory reusable TensorBoard results for a six-method, 5,000-step RL study.
Run from repository root: python audit_matched_5000.py
Does not train or overwrite any experiments.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path.cwd()
OUT = ROOT / 'matched_5000_audit'
OUT.mkdir(exist_ok=True)
ENVIRONMENTS = ('hopper', 'halfcheetah', 'walker2d', 'ant')
SHIFTS = ('0.1', '0.5', '2.0', '5.0')
SEEDS = range(6)
METHODS = ('SAC_TARGET_ONLY', 'BC_SAC', 'RLPD', 'BC_PAR', 'EA_NTC_NORMALIZED', 'EA_NTC_UNNORMALIZED')
CHECKPOINTS = (1000, 2000, 3000, 4000, 5000)
RAW_TAGS = ('test/target_return', 'test/target return')
NORM_TAGS = ('test/target_normalized_score', 'test/target normalized score')
SKIP = {'.git', '.venv', 'venv', 'site-packages', 'node_modules', 'matched_5000_audit'}


def classify(parts):
    p = [x.lower().replace('-', '_') for x in parts]
    # EA-NTC variants require explicit evidence; generic EA-NTC logs remain unclassified.
    if any('ea_ntc' in x for x in p):
        if any('unnormalized' in x or 'un_normalized' in x for x in p):
            return 'EA_NTC_UNNORMALIZED'
        if any('normalized' in x for x in p):
            return 'EA_NTC_NORMALIZED'
        return None
    for token, label in [('sac_target_only', 'SAC_TARGET_ONLY'), ('bc_par', 'BC_PAR'),
                         ('bc_sac', 'BC_SAC'), ('rlpd', 'RLPD')]:
        if token in p:
            return label
    return None


def env_shift(parts):
    for p in parts:
        low = p.lower()
        for env in ENVIRONMENTS:
            m = re.search(r'(?<![a-z])' + env + r'[-_]friction(?:[-_]srcdatatype[-_][a-z]+)?[-_](0\.1|0\.5|2(?:\.0)?|5(?:\.0)?)(?:$|[^\d.])', low)
            if m:
                return env, str(float(m.group(1)))
    return None, None


def seed_from_parts(parts):
    for p in reversed(parts):
        m = re.fullmatch(r'(?:r|seed[_-]?)([0-2])', p.lower())
        if m:
            return int(m.group(1))
    return None


def load_eval(tb):
    acc = EventAccumulator(str(tb), size_guidance={'scalars': 0})
    acc.Reload()
    tags = set(acc.Tags().get('scalars', []))
    raw = next((t for t in RAW_TAGS if t in tags), None)
    norm = next((t for t in NORM_TAGS if t in tags), None)
    if not raw and not norm:
        return None
    chosen = raw or norm
    # A folder may contain multiple event files. Deduplicate by step; latest event wins.
    vals = {}
    for event in acc.Scalars(chosen):
        vals[int(event.step)] = float(event.value)
    return chosen, vals, raw is not None


rows = []
for tb in ROOT.rglob('tb'):
    if not tb.is_dir() or any(p.lower() in SKIP for p in tb.parts):
        continue
    if not list(tb.glob('events.out.tfevents.*')):
        continue
    method = classify(tb.parts)
    env, shift = env_shift(tb.parts)
    seed = seed_from_parts(tb.parts)
    if not method or env is None or seed is None or shift not in SHIFTS:
        continue
    try:
        result = load_eval(tb)
    except Exception as e:
        print('SKIPPED (unreadable):', tb, str(e))
        continue
    if not result:
        continue
    tag, vals, has_raw = result
    steps = sorted(vals)
    rows.append(dict(method=method, environment=env, shift=shift, seed=seed,
                     path=str(tb.relative_to(ROOT)), tag=tag, raw_return_available=has_raw,
                     checkpoints=';'.join(map(str, steps)), max_eval_step=max(steps) if steps else '',
                     complete_5000=all(s in vals for s in CHECKPOINTS),
                     eval_1000=vals.get(1000, ''), eval_2000=vals.get(2000, ''),
                     eval_3000=vals.get(3000, ''), eval_4000=vals.get(4000, ''),
                     eval_5000=vals.get(5000, '')))

fields = ['method', 'environment', 'shift', 'seed', 'path', 'tag', 'raw_return_available',
          'checkpoints', 'max_eval_step', 'complete_5000',
          'eval_1000', 'eval_2000', 'eval_3000', 'eval_4000', 'eval_5000']
with (OUT / 'all_candidate_runs.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(sorted(rows, key=lambda r: (r['method'], r['environment'], float(r['shift']), r['seed'], r['path'])))

by_key = defaultdict(list)
for row in rows:
    if row['complete_5000'] and row['raw_return_available']:
        by_key[(row['method'], row['environment'], row['shift'], row['seed'])].append(row)

coverage = []
missing = []
duplicates = []
for env in ENVIRONMENTS:
    for shift in SHIFTS:
        for seed in SEEDS:
            for method in METHODS:
                key = (method, env, shift, seed)
                candidates = by_key[key]
                status = 'MISSING' if not candidates else 'DUPLICATE_REVIEW' if len(candidates) > 1 else 'AVAILABLE'
                entry = dict(method=method, environment=env, shift=shift, seed=seed,
                             status=status, n_eligible=len(candidates),
                             paths=' | '.join(r['path'] for r in candidates))
                coverage.append(entry)
                if status == 'MISSING':
                    missing.append(entry)
                elif status == 'DUPLICATE_REVIEW':
                    duplicates.append(entry)

for filename, data in [('coverage_288_slots.csv', coverage), ('missing_runs.csv', missing),
                       ('duplicate_review.csv', duplicates)]:
    with (OUT / filename).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['method', 'environment', 'shift', 'seed',
                                          'status', 'n_eligible', 'paths'])
        w.writeheader()
        w.writerows(data)

print('TensorBoard candidate folders:', len(rows))
print('Unique fully comparable slots:', sum(r['status'] == 'AVAILABLE' for r in coverage))
print('Slots with duplicate candidates:', len(duplicates))
print('Missing slots:', len(missing))
print('\nMethod coverage (available / duplicate review / missing):')
for method in METHODS:
    subset = [r for r in coverage if r['method'] == method]
    print('{:<22} {:>3} / {:>3} / {:>3}'.format(method,
          sum(r['status'] == 'AVAILABLE' for r in subset),
          sum(r['status'] == 'DUPLICATE_REVIEW' for r in subset),
          sum(r['status'] == 'MISSING' for r in subset)))
print('\nSaved CSVs to:', OUT)
print('NOTE: Missing EA-NTC runs may exist in folders without explicit variant names.')
print('NOTE: Verify source datasets, code versions, and settings before reusing any run.')


