"""Post-hoc P1-01, sealed records only; never modify historical families.

Run from any directory: python scripts/analyze_jiis_revision_pairs.py --freeze
then python scripts/analyze_jiis_revision_pairs.py --compute
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.paired_inference import holm_adjust
from src.ircot_fullwiki.analyze_fullwiki_retrieval import mcnemar_exact

OUT = ROOT / 'local-artifacts/jiis_priority_revision_20261003/paired'
PUBLIC = ROOT / 'artifacts/summaries/jiis_priority_revision_20261003/paired'
BASE = 'local-artifacts/jiis_acceptance_v3/multihop_rag/'
SUMMARY = 'artifacts/summaries/jiis_acceptance_v3/'
FIX = SUMMARY + 'reader_chatfix_v1/'
SEALS = {FIX+'answers/seal_manifest.json': '5c5e5a9e71491e5e261ae9a578c92b2e187881a9bfd5f819cb692f9be720dffc',
         FIX+'robustness/seal_manifest.json': '3f3a4a1b003ef8bfe1c0525eb440cbac886dc1f992808fe1dc29b0d15fc91c46'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def align(records, ids):
    actual = [r['id'] for r in records]
    if actual != ids or len(set(actual)) != len(actual):
        raise ValueError('IDs missing, duplicated, or out of frozen order')
    return records


def id_hash(ids):
    return hashlib.sha256('\n'.join(ids).encode()).hexdigest()


def infer(baseline, target, *, binary):
    b, t = np.asarray(baseline, dtype=float), np.asarray(target, dtype=float)
    if b.shape != t.shape or b.ndim != 1 or not b.size or not np.isfinite(b).all() or not np.isfinite(t).all():
        raise ValueError('finite equal paired vectors required')
    d = t-b
    rng = np.random.default_rng(13)
    samples = []
    for _ in range(100):
        samples.extend(d[rng.integers(0, len(d), size=(100, len(d)))].mean(axis=1))
    low, high = np.quantile(samples, [.025, .975])
    left, right = int(np.sum(d < 0)), int(np.sum(d > 0))
    if binary:
        if not np.isin(b, [0, 1]).all() or not np.isin(t, [0, 1]).all():
            raise ValueError('binary endpoint contains nonbinary values')
        p = mcnemar_exact(left, right)
        exact, test = p, 'two_sided_exact_mcnemar'
    else:
        rng = np.random.default_rng(13)
        extreme = 0
        observed = abs(d.sum())
        for _ in range(100):
            sums = (d * (rng.integers(0, 2, size=(100, len(d))) * 2-1)).sum(axis=1)
            extreme += int(np.sum(np.abs(sums) >= observed-1e-12))
        p = (extreme+1)/10001
        exact, test = '', 'paired_sign_flip_monte_carlo'
    return dict(n=len(d), baseline_mean=float(b.mean()), target_mean=float(t.mean()), delta=float(d.mean()),
                ci_low=float(low), ci_high=float(high), baseline_only=left if binary else '',
                target_only=right if binary else '', baseline_higher=left, target_higher=right,
                ties=int(np.sum(d == 0)), p_exact=exact, p_raw=p, test=test, seed=13, n_resamples=10000)


def frozen_sources():
    sources = {}
    for path, expected in SEALS.items():
        if sha(ROOT/path) != expected:
            raise ValueError('sealed manifest hash mismatch')
        sources[path] = expected
        seal = read(ROOT/path)
        if not seal['complete']:
            raise ValueError('incomplete seal')
        for pair in seal['copies']:
            for item in (pair['source'], pair['archive']):
                if sha(ROOT/item['path']) != item['sha256']:
                    raise ValueError('sealed source/archive mismatch')
            sources[pair['source']['path']] = pair['source']['sha256']
    for path in (BASE+'robustness/frozen_inputs/manifest.json', SUMMARY+'retrieval/analysis_manifest.json',
                 FIX+'answers/analysis_manifest.json'):
        sources[path] = sha(ROOT/path)
    return sources


def freeze():
    sources = frozen_sources()
    questions = rows(ROOT/(BASE+'questions.jsonl'))
    ids = [q['id'] for q in questions]
    nonnull = [q['id'] for q in questions if not q['is_null_query']]
    prompt = read(ROOT/(BASE+'robustness/frozen_inputs/manifest.json'))['prompt_ids']
    if len(ids) != 2556 or len(nonnull) != 2255 or len(prompt) != 500 or len(set(prompt)) != 500 or not set(prompt) <= set(nonnull):
        raise ValueError('wrong fixed denominators')
    contract = dict(status='post_hoc_analysis_frozen_before_this_computation', historical_families_unchanged=True,
        ids=ids, nonnull_ids=nonnull, prompt_ids=prompt, sources=sources,
        families={'retrieval_vs_dense': 6, 'answer_vs_dense': 16, 'prompt_vs_baselines': 36},
        metrics=['all_support', 'any_support', 'parent_recall'], answer_metrics=['em','f1'],
        targets=['qwen_hyde','mistral_hyde'], readers=['qwen','mistral'], chars=[450,900],
        prompt_modes=['standard','evidence_only','query2doc'], baselines=['dense','bm25'],
        inference='two-sided exact McNemar for binary; Monte Carlo paired sign-flip for continuous; Holm within each stated family; paired percentile bootstrap 95% CI; numpy RNG seed 13; 10000 resamples',
        continuous_exact_p_available=False, selection='original frozen prompt 500; no subset reselection')
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'analysis_contract.json'
    content = json.dumps(contract, indent=2, sort_keys=True)+'\n'
    if path.exists() and path.read_text(encoding='utf-8') != content:
        raise ValueError('refusing to alter frozen post-hoc contract')
    path.write_text(content, encoding='utf-8')


def compute():
    contract = read(OUT/'analysis_contract.json')
    for path, digest in contract['sources'].items():
        if sha(ROOT/path) != digest:
            raise ValueError('source changed since post-hoc freeze')
    allids, ids, promptids = contract['ids'], contract['nonnull_ids'], contract['prompt_ids']
    sourcepaths = contract['sources']
    ret = {}
    for name in ('dense','bm25','qwen_hyde','mistral_hyde'):
        suffix = {'dense':'dense_bge','bm25':'bm25','qwen_hyde':'qwen_hyde_bge','mistral_hyde':'mistral_hyde_bge'}[name]
        path = f'artifacts/per_example/jiis_acceptance_v3/mhrag_{suffix}_top10.jsonl'
        if path not in sourcepaths:
            raise ValueError('unsealed retrieval input')
        rr = align(rows(ROOT/path), allids)
        if [r['id'] for r in rr if not r['excluded_from_retrieval_denominator']] != ids:
            raise ValueError('retrieval non-null denominator mismatch')
        ret[name] = {r['id']:r for r in rr}
    results, descriptive = [], []

    def metric(row, name):
        return row['matched_evidence_count']/row['required_evidence_count'] if name == 'parent_recall' else float(row[name])

    def add(family, baseline, target, endpoint, selected, left, right, reader='', chars=''):
        results.append(dict(family=family, baseline=baseline, target=target, endpoint=endpoint,
            reader=reader, serializer_chars=chars, ordered_ids_sha256=id_hash(selected),
            **infer(left,right,binary=endpoint in ('all_support','any_support','em'))))

    for target in contract['targets']:
        for endpoint in contract['metrics']:
            add('retrieval_vs_dense','dense',target,endpoint,ids,
                [metric(ret['dense'][i],endpoint) for i in ids], [metric(ret[target][i],endpoint) for i in ids])
    for reader in contract['readers']:
        for chars in contract['chars']:
            ar = {}
            for name in ['dense',*contract['targets']]:
                paths = [p for p in sourcepaths if p.endswith(f'/readers/{reader}/{chars}/{name}.jsonl')]
                if len(paths) != 1:
                    raise ValueError('reader cell unavailable or ambiguous')
                rr = align(rows(ROOT/paths[0]),allids)
                if any(r['reader'] != reader or r['condition'] != name or r['serializer_chars'] != chars for r in rr):
                    raise ValueError('wrong reader cell metadata')
                if [r['id'] for r in rr if not r['is_null_query']] != ids:
                    raise ValueError('reader non-null denominator mismatch')
                ar[name] = {r['id']:r for r in rr}
            for target in contract['targets']:
                for endpoint in contract['answer_metrics']:
                    add('answer_vs_dense','dense',target,endpoint,ids,
                        [ar['dense'][i][endpoint] for i in ids],[ar[target][i][endpoint] for i in ids],reader,chars)
    for baseline in contract['baselines']:
        for endpoint in contract['metrics']:
            descriptive.append(dict(condition=baseline,endpoint=endpoint,n=500,ordered_ids_sha256=id_hash(promptids),
                mean=float(np.mean([metric(ret[baseline][i],endpoint) for i in promptids]))))
    for generator in contract['readers']:
        for mode in contract['prompt_modes']:
            path = BASE+f'reader_chatfix_v1/robustness/retrieval/{generator}_prompt_{mode}.jsonl'
            if path not in sourcepaths:
                raise ValueError('unsealed prompt input')
            pr = align(rows(ROOT/path),promptids)
            if any(r['excluded_from_retrieval_denominator'] for r in pr):
                raise ValueError('null question in prompt subset')
            for endpoint in contract['metrics']:
                values = [metric(r,endpoint) for r in pr]
                descriptive.append(dict(condition=f'{generator}_{mode}',endpoint=endpoint,n=500,
                    ordered_ids_sha256=id_hash(promptids),mean=float(np.mean(values))))
                for baseline in contract['baselines']:
                    add('prompt_vs_baselines',baseline,f'{generator}_{mode}',endpoint,promptids,
                        [metric(ret[baseline][i],endpoint) for i in promptids],values)
    for family, size in contract['families'].items():
        subset = [r for r in results if r['family'] == family]
        if len(subset) != size:
            raise ValueError('frozen family incomplete')
        for r,p in zip(subset,holm_adjust([r['p_raw'] for r in subset])):
            r.update(family_size=size,p_holm=p,analysis_status='post_hoc',comparison_evidence=
                'negative' if p < .05 and r['delta'] < 0 else 'positive' if p < .05 and r['delta'] > 0 else 'unresolved')
    PUBLIC.mkdir(parents=True,exist_ok=True)
    outputs = {}
    for filename,data in [('paired_comparisons.csv',results),('prompt_subset_descriptive.csv',descriptive)]:
        for directory in (OUT,PUBLIC):
            path=directory/filename
            with path.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
            outputs[path.relative_to(ROOT).as_posix()] = sha(path)
    summary = '# Post-hoc P1-01 paired diagnostics\n\n'
    summary += 'Historical families remain unchanged. Non-null n=2255; original fixed prompt subset n=500. Delta is target minus baseline. Holm families: retrieval 6; answers 16; prompt contrasts 36. CI: paired percentile bootstrap (10000 resamples, seed 13). Continuous p-values use Monte Carlo sign flips, not exact tests. Unresolved does not mean equivalent.\n\n'
    summary += '|Family|Reader/chars|Baseline|Target|Metric|n|Delta|95% CI|Discordance B/T|Raw p|Holm p|Evidence|\n|---|---|---|---|---|---:|---:|---|---|---:|---:|---|\n'
    for r in results:
        summary += f"|{r['family']}|{r['reader']}/{r['serializer_chars']}|{r['baseline']}|{r['target']}|{r['endpoint']}|{r['n']}|{r['delta']:.6f}|[{r['ci_low']:.6f}, {r['ci_high']:.6f}]|{r['baseline_only']}/{r['target_only']}|{r['p_raw']:.6g}|{r['p_holm']:.6g}|{r['comparison_evidence']}|\n"
    for directory in (OUT,PUBLIC):
        (directory/'paired_summary.md').write_text(summary,encoding='utf-8')
    manifest = dict(complete=True,status='post_hoc',contract_sha256=sha(OUT/'analysis_contract.json'),runner_sha256=sha(Path(__file__)),
        sources=sourcepaths,outputs=outputs,all_ids_sha256=id_hash(allids),nonnull_ids_sha256=id_hash(ids),
        prompt_ids_sha256=id_hash(promptids),historical_sources_verified_unchanged=True,private_text_exported=False)
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (PUBLIC/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    report = summary+'\n## Provenance and verification\n\nSources and all source/archive pairs verified against the two existing corrected seals before freeze. Reader Qwen/450 reuses its token-equivalent historical files as bound by the corrected seal; other reader cells use corrected records. IDs are checked in exact original order before any filtering; baselines are sliced by the original frozen prompt IDs. No generation, rescoring, historical-family edits, or manuscript edits occurred. Sources are rehashed against the post-hoc freeze.\n\n'
    report += f"Contract SHA-256: `{manifest['contract_sha256']}`. Full source hashes: `paired/analysis_manifest.json`.\n\nReproduce: `python scripts/analyze_jiis_revision_pairs.py --freeze`, then `python scripts/analyze_jiis_revision_pairs.py --compute`. Tests: `python -m pytest tests/test_jiis_revision_pairs.py -q`.\n"
    (OUT.parent/'paired_report.md').write_text(report,encoding='utf-8')
    print('Post-hoc paired diagnostics complete: 58 contrasts, no private text exported.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--compute',action='store_true')
    args=parser.parse_args()
    if args.freeze == args.compute:
        parser.error('choose exactly one of --freeze or --compute')
    freeze() if args.freeze else compute()
