"""Read-only six-method RL experiment audit. Run from repository root.
Requires tensorboard. Outputs CSV reports in full_5000_audit/.
"""
import ast, csv, hashlib, json, re
from collections import defaultdict, Counter
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path.cwd(); OUT=ROOT/'full_5000_audit'; OUT.mkdir(exist_ok=True)
ENVS=('hopper','halfcheetah','walker2d','ant'); SHIFTS=('0.1','0.5','2.0','5.0'); SEEDS=range(3)
METHODS=('SAC_TARGET_ONLY','BC_SAC','RLPD','BC_PAR','EA_NTC_NORMALIZED','EA_NTC_UNNORMALIZED')
STEPS=(1000,2000,3000,4000,5000)
RAW=('test/target_return','test/target return'); NORM=('test/target_normalized_score','test/target normalized score')
SKIP={'.git','.venv','venv','site-packages','node_modules','full_5000_audit','matched_5000_audit'}

def canon_shift(x):
    try:
        v=float(x)
        return next((s for s in SHIFTS if float(s)==v),None)
    except (ValueError,TypeError): return None

def params_for(tb,seed):
    # Historical project writes r0log.txt alongside r0/tb, not log.txt.
    run_dir=tb.parent
    candidates=[run_dir.parent/(f'r{seed}log.txt'),run_dir/(f'r{seed}log.txt'),run_dir/'log.txt']
    for f in candidates:
        if f.is_file():
            try:
                lines=f.read_text(encoding='utf-8-sig',errors='replace').splitlines()
                data={}
                for line in lines:
                    st=line.strip()
                    if st.startswith('(') and st.endswith(')'):
                        try:
                            pair=ast.literal_eval(st)
                            if isinstance(pair,tuple) and len(pair)==2 and isinstance(pair[0],str):data[pair[0]]=pair[1]
                        except (ValueError,SyntaxError):pass
                return str(f.relative_to(ROOT)),data
            except OSError:pass
    return '',{}

def identify(tb):
    parts=[p.lower().replace('-','_') for p in tb.parts]
    joined='/'.join(parts)
    env=shift=None
    for p in tb.parts:
        m=re.search(r'(hopper|halfcheetah|walker2d|ant)[-_]friction(?:[-_]srcdatatype[-_][a-z]+)?[-_](0\.1|0\.5|2(?:\.0)?|5(?:\.0)?)(?:$|[^\d.])',p.lower())
        if m: env=m.group(1);shift=canon_shift(m.group(2));break
    seed=None
    for p in reversed(tb.parts):
        m=re.fullmatch(r'(?:r|seed[_-]?)([0-5])',p.lower())
        if m:seed=int(m.group(1));break
    if env is None or seed is None:return None
    logpath,config=params_for(tb,seed)
    policy=str(config.get('policy','')).lower().replace('-','_')
    if 'ea_ntc' in joined or 'ea_ntc' in policy:
        reduction=config.get('ea_bc_reduction')
        if reduction=='normalized':method='EA_NTC_NORMALIZED'
        elif reduction in ('mean','unnormalized'):method='EA_NTC_UNNORMALIZED'
        else:method='UNCLASSIFIED_EA_NTC'
    elif 'sac_target_only' in joined or policy=='sac_target_only':method='SAC_TARGET_ONLY'
    elif any(t in joined for t in ('bc_par','par_bc')) or policy in ('par','bc_par','par_bc'):method='BC_PAR'
    elif 'bc_sac' in joined or policy=='bc_sac':method='BC_SAC'
    elif 'rlpd' in joined or policy=='rlpd':method='RLPD'
    else:method='UNCLASSIFIED_METHOD'
    return method,env,shift,seed,logpath,config

def write_csv(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(rows)

def sig(config):
    # Only compare recorded configuration; omit run identifiers.
    omit={'seed','shift_level','env_name','device','save_path','log_dir','log_path'}
    return {k:v for k,v in config.items() if k not in omit}

runs=[];errors=[]
for tb in ROOT.rglob('tb'):
    if not tb.is_dir() or any(p.lower() in SKIP for p in tb.parts):continue
    events=sorted(tb.glob('events.out.tfevents.*'))
    if not events:continue
    info=identify(tb)
    if info is None:
        errors.append({'path':str(tb.relative_to(ROOT)),'reason':'could not parse environment or seed'});continue
    method,env,shift,seed,logpath,config=info
    try:
        acc=EventAccumulator(str(tb),size_guidance={'scalars':0});acc.Reload()
        tags=set(acc.Tags().get('scalars',[]))
        raw=next((x for x in RAW if x in tags),None)
        norm=next((x for x in NORM if x in tags),None)
        tag=raw or norm
        if not tag:
            errors.append({'path':str(tb.relative_to(ROOT)),'reason':'no target return/normalized score scalar'});continue
        vals={int(e.step):float(e.value) for e in acc.Scalars(tag)}
        checkpoints=all(x in vals for x in STEPS)
        fingerprints=[hashlib.sha256(f.read_bytes()).hexdigest() for f in events]
        config_sig=sig(config)
        runs.append(dict(method=method,environment=env,shift=shift,seed=seed,
                         path=str(tb.relative_to(ROOT)),log_path=logpath,
                         reduction=str(config.get('ea_bc_reduction','NOT_RECORDED')),
                         max_step=config.get('max_step','NOT_RECORDED'),eval_freq=config.get('eval_freq','NOT_RECORDED'),
                         eval_episode=config.get('eval_episode','NOT_RECORDED'),
                         target_interval=config.get('tar_env_interact_interval','NOT_RECORDED'),
                         adaptive_strength=config.get('ea_adaptive_max_strength','NOT_RECORDED'),
                         adaptive_gate=config.get('ea_adaptive_gate','NOT_RECORDED'),
                         raw_return_available=bool(raw),tag=tag,
                         steps=';'.join(map(str,sorted(vals))),complete_5000=checkpoints,
                         event_files=len(events),event_hashes=';'.join(fingerprints),
                         config_signature=json.dumps(config_sig,sort_keys=True,default=str),
                         **{f'eval_{x}':vals.get(x,'') for x in STEPS}))
    except Exception as e:errors.append({'path':str(tb.relative_to(ROOT)),'reason':repr(e)})

runs.sort(key=lambda r:(r['method'],r['environment'],float(r['shift']),r['seed'],r['path']))
write_csv('all_candidates.csv',runs);write_csv('scan_errors.csv',errors)
bykey=defaultdict(list)
for r in runs:
    if r['method'] in METHODS and r['complete_5000'] and r['raw_return_available']:
        bykey[(r['method'],r['environment'],r['shift'],r['seed'])].append(r)
coverage=[];duplicates=[];missing=[]
for method in METHODS:
 for env in ENVS:
  for shift in SHIFTS:
   for seed in SEEDS:
    group=bykey[(method,env,shift,seed)]
    if not group:status='MISSING_OR_NOT_ELIGIBLE'
    elif len(group)==1:status='ONE_ELIGIBLE_CANDIDATE'
    else:status='DUPLICATE_REVIEW'
    row={'method':method,'environment':env,'shift':shift,'seed':seed,'status':status,'eligible_candidates':len(group),'paths':' | '.join(x['path'] for x in group)}
    coverage.append(row)
    if not group:missing.append(row)
    if len(group)>1:
        config_same=len({x['config_signature'] for x in group})==1
        # Equal evaluation values are NOT proof of same training run; just a diagnostic.
        eval_same=all(all(x[f'eval_{s}']==group[0][f'eval_{s}'] for s in STEPS) for x in group[1:])
        hashes_same=len({x['event_hashes'] for x in group})==1
        duplicates.append({**row,'recorded_config_identical':config_same,'eval_values_identical':eval_same,'event_hashes_identical':hashes_same,'decision':'MANUAL_REVIEW_REQUIRED'})
write_csv('coverage_576_slots.csv',coverage)
write_csv('missing_or_not_eligible.csv',missing)
write_csv('duplicate_review.csv',duplicates)
write_csv('unclassified_runs.csv',[r for r in runs if r['method'] not in METHODS])
write_csv('incomplete_or_no_raw.csv',[r for r in runs if not r['complete_5000'] or not r['raw_return_available']])
print('Candidate folders:',len(runs),' Scan errors:',len(errors))
print('Expected slots:',len(coverage),' One eligible:',sum(r['status']=='ONE_ELIGIBLE_CANDIDATE' for r in coverage),' Duplicates:',len(duplicates),' Missing/not eligible:',len(missing))
for method in METHODS:
    sub=[r for r in coverage if r['method']==method]
    print(f'{method:22s}: one={sum(r["status"]=="ONE_ELIGIBLE_CANDIDATE" for r in sub):2d} duplicate={sum(r["status"]=="DUPLICATE_REVIEW" for r in sub):2d} missing={sum(r["status"]=="MISSING_OR_NOT_ELIGIBLE" for r in sub):2d}')
print('Unclassified:',sum(r['method'] not in METHODS for r in runs))
print('Reports:',OUT)
print('READ-ONLY. No duplicates automatically discarded. Missing includes logs lacking raw-return or full checkpoints.')

