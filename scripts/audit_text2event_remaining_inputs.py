"""Audit BKEE, GENEVA and PHEE inputs against frozen adaptation and curated gold."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_sets(row):
    """Compare offsets using the same identifying fields as the project metrics."""
    result={key:set() for key in ['TI','TC','AI','AC']}
    for event in row['events']:
        kind=event['event_type']
        for trigger in event['trigger']:
            result['TI'].add(tuple(trigger['span']));result['TC'].add((kind,tuple(trigger['span'])))
        for arg in event['arguments']:
            for mention in arg['mentions']:
                span=tuple(mention['span']);result['AI'].add((kind,span));result['AC'].add((kind,arg['role'],span))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--curated',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    source=args.source.resolve();sys.path.insert(0,str(source))
    from transformers import AutoTokenizer
    from src.baselines.text2event.convert import convert,convert_substructure,parse,to_unified
    from src.data.unified import validate
    t=AutoTokenizer.from_pretrained(source/'bundle/tokenizer',local_files_only=True)
    normalize=t.backend_tokenizer.normalizer.normalize_str
    report={'scope':'BKEE GENEVA PHEE only; no model training or generation',
        'source_manifest_sha256':digest(source/'package-manifest.json'),'datasets':{}}
    for dataset in ['BKEE','GENEVA','PHEE']:
        folder=source/'bundle'/dataset;types,roles,mapping=[json.loads(l) for l in (folder/'event.schema').read_text().splitlines()]
        report['datasets'][dataset]={'schema_types':len(types),'schema_roles':len(roles),
            'same_name_type_role':sorted(set(types)&set(roles)),
            'all_roles_allowed_for_every_type':all(set(rs)==set(roles) for rs in mapping.values()),'splits':{}}
        for split in ['train','dev','test']:
            gold=read(folder/(split+'.gold.jsonl'));prepared=read(folder/(split+'.json'));curated=read(args.curated/dataset/(split+'.jsonl'))
            curated_by_id={r['id']:r for r in curated};c=Counter();issues=[];examples=[];seen=set()
            lengths=Counter();ambiguity_examples=[]
            if len(gold)!=len(prepared):issues.append({'error':'prepared_gold_count_mismatch'})
            if len(curated)!=len(gold):issues.append({'error':'curated_gold_count_mismatch'})
            for row,p in zip(gold,prepared):
                c['samples']+=1
                try:validate(row)
                except ValueError as error:issues.append({'id':row.get('id'),'error':'unified_validation','reason':str(error)});continue
                if row['id'] in seen:issues.append({'id':row['id'],'error':'duplicate_id'})
                seen.add(row['id'])
                if curated_by_id.get(row['id'])!=row:
                    c['curated_content_mismatch']+=1
                    if len(examples)<5:examples.append({'id':row['id'],'kind':'curated_gold_mismatch'})
                if convert(row)!=p:issues.append({'id':row['id'],'error':'prepared_target_mismatch'})
                trees=parse(p['event']);sub=convert_substructure(row);leaves=parse(sub['event'])
                triggers=sum(len(e['trigger']) for e in row['events'])
                arguments=sum(len(e['trigger'])*sum(len(a['mentions']) for a in e['arguments']) for e in row['events'])
                if len(trees)!=triggers or sum(len(e['roles']) for e in trees)!=arguments or len(leaves)!=triggers+arguments:
                    issues.append({'id':row['id'],'error':'annotation_count_loss'})
                c['triggers']+=triggers;c['argument_mentions_expanded']+=arguments;c['negative_samples']+=not row['events']
                for event in row['events']:
                    if event['event_type'] not in types:issues.append({'id':row['id'],'error':'unknown_type'})
                    c['multi_trigger_events']+=len(event['trigger'])>1
                    for arg in event['arguments']:
                        if arg['role'] not in roles:issues.append({'id':row['id'],'error':'unknown_role'})
                        for mention in arg['mentions']:
                            for field in ['char_span','mention_group','fragment_index']:c['mentions_with_'+field]+=field in mention
                for stage,target,prefix in [('substructure',sub['event'],'span: '),('full',p['event'],'event: ')]:
                    source_ids=t.encode(prefix+p['text']);target_ids=t.encode(target)
                    lengths[stage+'_max_source']=max(lengths[stage+'_max_source'],len(source_ids))
                    lengths[stage+'_max_target']=max(lengths[stage+'_max_target'],len(target_ids))
                    c[stage+'_source_overflow']+=len(source_ids)>832;c[stage+'_target_overflow']+=len(target_ids)>704
                    c[stage+'_target_unknown_tokens']+=target_ids.count(t.unk_token_id)
                    decoded=t.decode([i for i in target_ids if i!=t.eos_token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                    try:
                        decoded_records=parse(decoded)
                        pairs=lambda records:Counter((e['type'],normalize(e['trigger']),tuple((r,normalize(s)) for r,s in e['roles'])) for e in records)
                        if pairs(decoded_records)!=pairs(leaves if stage=='substructure' else trees):c[stage+'_token_roundtrip_changed']+=1
                    except ValueError:c[stage+'_token_roundtrip_parse_failure']+=1
                diagnostics=[]
                try:
                    reconstructed=to_unified(p['event'],row['id'],row['text'],row['tokens'],ambiguity_policy='nearest-trigger',
                        diagnostics=diagnostics,surface_normalizer=normalize)
                    before=metric_sets(row);after=metric_sets(reconstructed)
                    for key in before:
                        if before[key]!=after[key]:c['roundtrip_'+key+'_offset_changed_samples']+=1
                    c['ambiguous_surface_samples']+=bool(diagnostics)
                    if diagnostics and len(ambiguity_examples)<2:ambiguity_examples.append({'id':row['id'],'ambiguities':diagnostics[:2]})
                except (ValueError,KeyError) as error:
                    c['roundtrip_recovery_failure']+=1
                    if len(examples)<5:examples.append({'id':row['id'],'kind':'span_recovery_failure','reason':str(error)})
            report['datasets'][dataset]['splits'][split]={'counts':dict(c),'lengths':dict(lengths),'structural_issues':issues,
                'examples':examples,'ambiguity_examples':ambiguity_examples,
                'bundle_gold_sha256':digest(folder/(split+'.gold.jsonl')),'curated_sha256':digest(args.curated/dataset/(split+'.jsonl'))}
            print(dataset,split,json.dumps({'counts':dict(c),'lengths':dict(lengths),'structural_issues':len(issues)}),flush=True)
        row=next(r for r in read(folder/'train.gold.jsonl') if r['events'])
        report['datasets'][dataset]['example']={'id':row['id'],'gold':row,'encoder_substructure':'span: '+convert(row)['text'],
            'encoder_full':'event: '+convert(row)['text'],'substructure_target':convert_substructure(row)['event'],'full_target':convert(row)['event']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
