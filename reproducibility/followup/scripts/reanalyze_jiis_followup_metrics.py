"""Recompute F1 statistics from public numeric vectors, never from predictions.

Existing inference algorithms are imported unchanged. The input directory must
contain complete serializer/ and holdout/ public exports plus expected summaries.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_jiis_revision_pairs import infer, id_hash
from src.evaluation.paired_inference import holm_adjust
READERS = ('qwen', 'mistral')
METHODS = ('bm25', 'hyde', 'ce')
PAIRS = (('bm25','hyde'), ('bm25','ce'), ('hyde','ce'))
HOLDOUT_FAMILIES = {'primary_em':6,'primary_retrieval_all_support_titles':3,
                    'secondary_f1':6,'sensitivity_official_em':6,'sensitivity_official_f1':6}
NUMERIC = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$')
IDENTITY = {'id','id_sha256','effective_input_sha256','ordered_ids_sha256','protocol_sha256',
            'reader','method','endpoint','family','finish_reason','status','evidence','test'}


def verify_manifest(base):
    path=base/'manifest.json'
    if not path.exists(): return False
    manifest=json.loads(path.read_text(encoding='utf-8'))
    for entry in manifest['files']:
        target=(base/entry['path']).resolve()
        if not target.is_relative_to(base.resolve()): raise ValueError('manifest path escapes capsule')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('capsule manifest hash mismatch '+entry['path'])
    return True


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows: raise ValueError('empty public CSV: '+str(path))
    return rows


def number(value):
    if str(value).lower() in ('true','false'): return float(str(value).lower()=='true')
    result=float(value)
    if not math.isfinite(result): raise ValueError('nonfinite public metric')
    return result


def metric_rows(path):
    records=read_csv(path)
    forbidden={'question','answer','prediction','raw_prediction','prompt','text','paragraph_text','gold_answers'}
    for row in records:
        if forbidden.intersection(row): raise ValueError('nonpublic fields in numeric export')
        for key,value in row.items():
            if key not in IDENTITY: number(value)
    return records


def cells(records, keys, *, n, id_key):
    result={}
    for row in records:
        key=tuple(row[k] for k in keys)
        result.setdefault(key,[]).append(row)
    ordered=None
    for cell in result.values():
        ids=[r[id_key] for r in cell]
        if len(ids)!=n or len(set(ids))!=n: raise ValueError('incomplete or duplicate numeric-vector cell')
        if ordered is None: ordered=ids
        elif ids!=ordered: raise ValueError('paired ID ordering mismatch')
    return result,ordered


def compare(actual, expected, keys, *, ignored=()):
    actual_index={tuple(str(r[k]) for k in keys):r for r in actual}
    expected_index={tuple(str(r[k]) for k in keys):r for r in expected}
    if len(actual_index)!=len(actual) or len(expected_index)!=len(expected): raise ValueError('duplicate summary key')
    if list(actual_index)!=list(expected_index): raise ValueError('expected summary key/order mismatch')
    count=0
    for key,row in expected_index.items():
        computed=actual_index[key]
        for field,value in row.items():
            if field in ignored: continue
            if field not in computed: raise ValueError('unreconstructed summary field '+field)
            other=computed[field]
            if value!='' and NUMERIC.fullmatch(value) and field not in IDENTITY:
                if abs(number(value)-number(other))>1e-12: raise ValueError('frozen numeric mismatch '+field+' '+str(key))
                count+=1
            elif str(other)!=value: raise ValueError('frozen metadata mismatch '+field+' '+str(key))
    return count


def adjust(results, family_sizes, *, label):
    for family,size in family_sizes.items():
        subset=[r for r in results if r['family']==family]
        if len(subset)!=size: raise ValueError('incomplete declared family '+family)
        for row,p in zip(subset,holm_adjust([r['p_raw'] for r in subset])):
            row.update(family_size=size,p_holm=p)
            row[label]='positive' if p<.05 and row['delta']>0 else 'negative' if p<.05 and row['delta']<0 else 'unresolved'


def serializer(directory):
    answers=metric_rows(directory/'answer_metrics_per_example.csv')
    support=metric_rows(directory/'support_per_example_sanitized.csv')
    a,ids=cells(answers,('reader','chars'),n=100,id_key='id_sha256')
    s,sids=cells(support,('reader','chars'),n=100,id_key='id_sha256')
    expected_keys={(r,str(ch)) for r in READERS for ch in (450,900)}
    if set(a)!=expected_keys or set(s)!=expected_keys or ids!=sids: raise ValueError('serializer matrix/ID mismatch')
    results=[];summary=[]
    for reader in READERS:
        for chars in ('450','900'):
            cell=a[reader,chars];scell=s[reader,chars]
            if 'effective_input_sha256' in cell[0] and any(x['effective_input_sha256']!=y['effective_input_sha256'] for x,y in zip(cell,scell)): raise ValueError('answer/support input identity mismatch')
            row=dict(reader=reader,chars=int(chars),n=100)
            for metric in ('local_em','local_f1','official_em','official_f1'):
                row[metric]=sum(number(x[metric]) for x in cell)/100
            for metric in ('local_em','local_f1'):
                row['raw_'+metric]=sum(number(x['raw_'+metric]) for x in cell)/100
            for k in ('all_support_retrieved','all_support_chars','all_support_tokens'):
                row[k+'_count']=sum(number(x[k]) for x in scell)
            # Runtime summaries are checked by the private independent validator;
            # this capsule does not infer runtime from score-only exports.
            summary.append(row)
        for metric in ('local_em','local_f1'):
            results.append(dict(reader=reader,baseline=450,target=900,endpoint=metric,
                family='serializer_'+metric,**infer([number(x[metric]) for x in a[reader,'450']],
                [number(x[metric]) for x in a[reader,'900']],binary=metric=='local_em')))
    adjust(results,{'serializer_local_em':2,'serializer_local_f1':2},label='status')
    return results,summary,ids


def holdout(directory):
    vectors=metric_rows(directory/'per_example_metrics.csv')
    ret=metric_rows(directory/'retrieval_metrics_per_example.csv')
    a,ids=cells(vectors,('reader','method'),n=300,id_key='id')
    r,rids=cells(ret,('method',),n=300,id_key='id')
    if set(a)!={(q,m) for q in READERS for m in METHODS} or set(r)!={(m,) for m in METHODS} or ids!=rids: raise ValueError('holdout matrix/ID mismatch')
    anchor=id_hash(ids);results=[];summary=[]
    for method in METHODS:
        values=[number(x['all_support_titles']) for x in r[method,]]
        if any(x not in (0,1) for x in values): raise ValueError('nonbinary retrieval metric')
        summary.append(dict(reader='',method=method,endpoint='all_support_titles',n=300,mean=sum(values)/300,raw_mean='',ordered_ids_sha256=anchor))
    for reader in READERS:
        for method in METHODS:
            cell=a[reader,method]
            for metric in ('em','f1','official_em','official_f1'):
                summary.append(dict(reader=reader,method=method,endpoint=metric,n=300,
                    mean=sum(number(x[metric]) for x in cell)/300,
                    raw_mean=(sum(number(x['raw_'+metric]) for x in cell)/300 if 'raw_'+metric in cell[0] else ''),ordered_ids_sha256=anchor))
        for left,right in PAIRS:
            for metric,family in [('em','primary_em'),('f1','secondary_f1'),('official_em','sensitivity_official_em'),('official_f1','sensitivity_official_f1')]:
                results.append(dict(family=family,reader=reader,baseline=left,target=right,endpoint=metric,
                    ordered_ids_sha256=anchor,**infer([number(x[metric]) for x in a[reader,left]],
                        [number(x[metric]) for x in a[reader,right]],binary=metric.endswith('em'))))
    for left,right in PAIRS:
        results.append(dict(family='primary_retrieval_all_support_titles',reader='',baseline=left,target=right,
            endpoint='all_support_titles',ordered_ids_sha256=anchor,
            **infer([number(x['all_support_titles']) for x in r[left,]],
                    [number(x['all_support_titles']) for x in r[right,]],binary=True)))
    adjust(results,HOLDOUT_FAMILIES,label='evidence')
    return results,summary,ids


def write_csv(path, records):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for r in records for k in r))
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(records)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();checks={};manifest_checked=verify_manifest(args.input_root.parent)
    for name,fn in [('serializer',serializer),('holdout',holdout)]:
        d=args.input_root/name
        paired,summary,ids=fn(d)
        numeric=compare(paired,read_csv(d/'paired_comparisons.csv'),
            ('reader','baseline','target','endpoint','family'))
        expected_summary=read_csv(d/'descriptive_summary.csv')
        skipped=['eos_observed_count','output_cap_hit_count','mean_input_tokens','mean_output_tokens','reader_seconds'] if name=='serializer' else []
        if name=='holdout':
            for row in expected_summary:
                if row['endpoint'] in ('official_em','official_f1') and not next(x for x in summary if all(str(x[k])==row[k] for k in ('reader','method','endpoint')))['raw_mean']:
                    row['raw_mean']=''
                    skipped.append('raw_mean:'+row['reader']+':'+row['method']+':'+row['endpoint'])
        numeric+=compare(summary,expected_summary,('reader','method','endpoint') if name=='holdout' else ('reader','chars'),ignored=skipped)
        checks[name]=dict(complete=True,paired_contrasts=len(paired),n=len(ids),numeric_fields_verified=numeric,
            excluded_runtime_or_unavailable_fields=skipped,vector_order_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
        write_csv(args.out/(name+'_paired_recomputed.csv'),paired)
        write_csv(args.out/(name+'_descriptive_recomputed.csv'),summary)
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(complete=True,checks=checks,capsule_manifest_checked=manifest_checked,numeric_absolute_tolerance=1e-12,
        scope='public numeric vector inference only; no predictions, scorer, tokenizer, corpus or model execution',
        python=sys.version)
    (args.out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PASS: serializer4 and holdout27 paired contrasts and numeric summaries reproduced.')

if __name__=='__main__': main()
