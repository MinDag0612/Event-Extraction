import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from src.data.unified import read_rows, validate, normalize
from src.data.prepare_unified import verify_frozen_gold
from src.baselines.oneie.convert import flatten
from src.baselines.oneie.shared import DATASETS, to_native, dump_json, dump_rows

ROOT = Path(__file__).resolve().parents[3]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def make_vocabs(types, roles):
    def labels(values): return {'O':0, **{label:i for i,label in enumerate(sorted(values),1)}}
    def bio(values): return {'O':0, **{label:i for i,label in enumerate(
        [f'{prefix}-{value}' for value in sorted(values) for prefix in ('B','I')],1)}}
    if 'O' in types or 'O' in roles: raise ValueError('Gold uses reserved O label')
    return dict(entity_type=labels({'ARG'}),event_type=labels(types),role_type=labels(roles),
                relation_type={'O':0},mention_type={'NAM':0,'NOM':1,'PRO':2,'UNK':3},
                entity_label=bio({'ARG'}),trigger_label=bio(types))


def prepare(source, output, dataset, tokenizer, max_length=512, catalog=None):
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if 'files' in manifest:
        verify_frozen_gold(source, source/'manifest.json')
    gold={};seen_ids=set();seen_text={}
    for split in ('train','dev','test'):
        path=source/dataset/f'{split}.jsonl'
        expected = (manifest['files'][f'{dataset}/{split}.jsonl']['sha256'] if 'files' in manifest
                    else manifest['outputs'][f'{dataset}/{split}.jsonl'])
        if sha(path)!=expected: raise ValueError('Source checksum mismatch')
        rows=read_rows(path)
        if not rows: raise ValueError(f'Empty {split}')
        for row in rows:
            validate(row)
            if row['id'] in seen_ids: raise ValueError('Duplicate ID across splits')
            seen_ids.add(row['id']);text=normalize(row['text'])
            if text in seen_text: raise ValueError('Duplicate normalized text')
            seen_text[text]=split
        gold[split]=rows
    types={e['event_type'] for r in gold['train'] for e in r['events']}
    roles={a['role'] for r in gold['train'] for e in r['events'] for a in e['arguments']}
    if catalog:
        types.update(catalog['event_types']);roles.update(catalog['roles'])
    vocabs=make_vocabs(types,roles)
    if max(len(v) for v in vocabs.values())>=1000: raise ValueError('OneIE label radix exceeded')
    report=dict(dataset=dataset,manifest_sha256=sha(source/'manifest.json'),max_length=max_length,
                tokenizer_class=type(tokenizer).__name__,tokenizer_name=tokenizer.name_or_path,
                tokenizer_vocab_sha256=hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest(),
                vocabulary_policy='TRAIN labels + official GENEVA metadata when supplied; no held-out label harvesting',
                constraints='All vocabulary event-role combinations permitted; no hard ontology constraints',
                ontology_source=catalog.get('source_url') if catalog else None,splits={})
    for split,rows in gold.items():
        native=[];full=[];counts=Counter(samples=len(rows));long=[]
        for row in rows:
            graph=to_native(row)
            if any(e['event_type'] not in types or any(a['role'] not in roles for a in e['arguments']) for e in graph['event_mentions']):
                raise ValueError(f'{row["id"]}: unknown label; extend an explicit catalog, never build it from TEST')
            full.append(copy.deepcopy(graph))
            counts['events_before']+=len(graph['event_mentions'])
            counts['arguments_before']+=sum(len(e['arguments']) for e in graph['event_mentions'])
            counts.update(flatten(graph))
            pieces=[tokenizer.tokenize(t) or [tokenizer.unk_token] for t in row['tokens']]
            graph['token_lens']=[len(p) for p in pieces];graph['pieces']=[p for parts in pieces for p in parts]
            length=len(graph['pieces'])+tokenizer.num_special_tokens_to_add(pair=False)
            counts['max_encoder_length']=max(counts['max_encoder_length'],length)
            if length>max_length:long.append((row['id'],length))
            counts['events_after']+=len(graph['event_mentions'])
            counts['arguments_after']+=sum(len(e['arguments']) for e in graph['event_mentions'])
            native.append(graph)
        if long: raise ValueError(f'{split}: overlength records; no truncation allowed: {long[:20]}')
        dump_rows(output/f'{split}.jsonl',native)
        dump_rows(output/'gold'/f'{split}.jsonl',rows)
        dump_rows(output/'native_gold'/f'{split}.jsonl',full)
        report['splits'][split]=dict(counts)
    dump_json(output/'vocabs.json',vocabs)
    dump_json(output/'patterns/event_role.json',{t:sorted(roles) for t in sorted(types)})
    dump_json(output/'patterns/role_entity.json',{r:['ARG'] for r in sorted(roles)})
    dump_json(output/'patterns/relation_entity.json',{})
    if catalog:dump_json(output/'label_catalog.json',catalog)
    report['files']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    dump_json(output/'preparation_report.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/unified')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dataset',choices=DATASETS,required=True)
    parser.add_argument('--model',default='bert-base-multilingual-cased')
    parser.add_argument('--revision',default='3f076fdb1ab68d5b2880cb87a0886f315b8146f8')
    parser.add_argument('--local-files-only',action='store_true')
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--catalog',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Frozen prepared output exists; choose a new path')
    if not 2<args.max_length<=512:raise ValueError('mBERT supports at most 512 positions')
    from transformers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(args.model,revision=args.revision,do_lower_case=False,local_files_only=args.local_files_only)
    catalog=args.catalog or (ROOT/'configs/oneie/geneva_labels.json' if args.dataset=='GENEVA' else None)
    values=json.loads(catalog.read_text()) if catalog else None
    args.output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.prepare-',dir=args.output.parent))
    try:
        report=prepare(args.input,stage,args.dataset,tokenizer,args.max_length,values)
        report['tokenizer_revision']=args.revision
        dump_json(stage/'preparation_report.json',report)
        stage.rename(args.output)
    finally:
        if stage.exists():shutil.rmtree(stage)
    print(json.dumps(report['splits'],indent=2))

if __name__=='__main__':main()
