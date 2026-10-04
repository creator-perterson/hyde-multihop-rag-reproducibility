from pathlib import Path
import csv, hashlib, json, math, statistics, sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from evaluation.analyze_expert_systems_minimum import PRIMARY_CONTRASTS,paired_bootstrap_ci,exact_mcnemar,holm_adjust,write_csv
out=ROOT/'outputs';out.mkdir(exist_ok=True)
manifest=json.loads((ROOT/'manifest.json').read_text())
for entry in manifest['files']:
 p=ROOT/entry['path']
 if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('hash mismatch '+entry['path'])
contrasts=[]
for dataset in ('hotpotqa','2wiki'):
 rows=[json.loads(x) for x in (ROOT/f'metrics/{dataset}.jsonl').read_text().splitlines()]
 methods={}
 for r in rows:
  key=r['example_id'];m=r['method']
  if key in methods.setdefault(m,{}):raise ValueError('duplicate ID')
  methods[m][key]=r
 ids=[r['example_id'] for r in rows if r['method']=='dense_bge']
 if len(ids)!=500 or any(set(v)!=set(ids) for v in methods.values()):raise ValueError('incomplete pairing')
 for i,(target,reference) in enumerate(PRIMARY_CONTRASTS):
  before=[methods[reference][x]['all_support_hit'] for x in ids];after=[methods[target][x]['all_support_hit'] for x in ids]
  if any(x not in (0,1) for x in before+after):raise ValueError('nonbinary endpoint')
  delta=[a-b for a,b in zip(after,before)]
  lo,hi=paired_bootstrap_ci(delta,iterations=10000,seed=13+i);m=exact_mcnemar(before,after)
  contrasts.append(dict(dataset=dataset,target=target,reference=reference,endpoint='all-support hit@10',n=500,target_value=statistics.fmean(after),reference_value=statistics.fmean(before),paired_delta=statistics.fmean(delta),ci_low=lo,ci_high=hi,mcnemar_gains=m['gains'],mcnemar_losses=m['losses'],mcnemar_discordant=m['discordant_total'],mcnemar_p=m['p_value']))
for r,p in zip(contrasts,holm_adjust([r['mcnemar_p'] for r in contrasts])):r['holm_adjusted_p']=p
expected=list(csv.DictReader((ROOT/'artifacts/summaries/expert_systems_minimum/paired_contrasts.csv').open()))
for actual,old in zip(contrasts,expected):
 for k,v in actual.items():
  if isinstance(v,(int,float)):
   if not math.isclose(v,float(old[k]),rel_tol=1e-12,abs_tol=1e-15):raise ValueError(f'different statistic {k}: {v} vs {old[k]}')
  elif str(v)!=old[k]:raise ValueError('different key '+k)
if len(expected)!=len(contrasts):raise ValueError('family mismatch')
write_csv(out/'open_generator_paired_recomputed.csv',contrasts)
import subprocess
subprocess.run([sys.executable,'scripts/rebuild_main_tables_from_summaries.py','--root','.','--out','outputs/legacy_main_tables.tex'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'scripts/build_expert_systems_latex_tables.py','--out_main','outputs/table_open_generator_query2doc.tex','--out_leakage','outputs/table_open_generator_leakage.tex'],cwd=ROOT,check=True)
for name in ('table_open_generator_query2doc.tex','table_open_generator_leakage.tex'):
 if (out/name).read_bytes()!=(ROOT/'expected'/name).read_bytes():raise ValueError('table mismatch '+name)
(out/'verification.json').write_text(json.dumps(dict(complete=True,scope='summary table reconstruction and eight public metric paired retrieval contrasts only',python=sys.version,bootstrap_iterations=10000,seeds=[13,14,15,16],holm_family_size=8,n_per_dataset=500,statistics_match_frozen=True,open_tables_byte_identical=True,model_rerun=False,raw_prediction_rescoring=False),indent=2))
print('PASS: eight paired contrasts match; two open-generator tables byte-identical; legacy table rebuilt.')

