import argparse
import copy
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path):
    text = path.read_text(encoding='utf-8')
    try:
        rows = json.loads(text)
        return rows if isinstance(rows, list) else [rows]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def char_record(source, dataset):
    text = source['text'] if dataset == 'VHE' else source['context']
    offsets = [m.span() for m in re.finditer(r'\w+|[^\w\s]', text)]
    row = {'doc_id':str(source['id']), 'sent_id':str(source['id']), 'sentence':text,
           'tokens':[text[a:b] for a,b in offsets], 'entity_mentions':[],
           'event_mentions':[], 'relation_mentions':[]}
    counts = Counter()
    entities = {}
    def span(start, end, expected):
        if not 0 <= start < end <= len(text):
            raise ValueError(f'Invalid character span: {start}:{end}')
        if text[start:end].strip() != expected.strip():
            raise ValueError(f'Character text mismatch: {expected!r} != {text[start:end]!r}')
        indices=[i for i,(a,b) in enumerate(offsets) if a < end and b > start]
        if not indices: raise ValueError('Empty token span')
        a,b=indices[0],indices[-1]+1
        counts['expanded_char_spans'] += int(text[offsets[a][0]:offsets[b-1][1]].strip()!=expected.strip())
        return a,b
    def argument(start,end,mention,role):
        a,b=span(start,end,mention)
        if (a,b) not in entities:
            entity={'id':f"{row['sent_id']}-E{len(entities)}",'start':a,'end':b,
                    'text':' '.join(row['tokens'][a:b]),'entity_type':'ARG','mention_type':'UNK'}
            entities[a,b]=entity
        return {'entity_id':entities[a,b]['id'],'text':mention,'role':role}
    def fragments(node):
        if len(node['text']) != len(node['start']): raise ValueError('Mismatched fragment groups')
        for texts,starts in zip(node['text'],node['start']):
            if len(texts)!=len(starts):raise ValueError('Mismatched fragments')
            counts['discontinuous_groups']+=int(len(texts)>1)
            for value,start in zip(texts,starts):yield start,start+len(value),value
    def phee_args(node,prefix=''):
        for key,value in node.items():
            if key in {'text','start','entity_id','event_id','event_type','Trigger','value'}:continue
            role=f'{prefix}.{key}' if prefix else key
            if isinstance(value,dict):
                if 'text' in value:
                    for a,b,t in fragments(value):yield argument(a,b,t,role)
                yield from phee_args(value,role)
            elif isinstance(value,list):
                for child in value:
                    if isinstance(child,dict):yield from phee_args(child,role)
    events=source['events'] if dataset=='VHE' else [e for ann in source['annotations'] for e in ann['events']]
    for event in events:
        if dataset=='VHE':
            start,end=event['offset'];trigger=event['trigger_word'];etype=event['type']
            args=[argument(*arg['offset'],arg['mention'],arg['role']) for arg in event['arguments']]
        else:
            parts=list(fragments(event['Trigger']))
            if len(parts)!=1:
                counts['unsupported_trigger_events']+=1
                continue
            start,end,trigger=parts[0];etype=event['event_type'];args=list(phee_args(event))
        a,b=span(start,end,trigger)
        row['event_mentions'].append({'id':f"{row['sent_id']}-T{len(row['event_mentions'])}",
            'event_type':etype,'trigger':{'start':a,'end':b,'text':trigger},'arguments':args})
    row['entity_mentions']=list(entities.values())
    return row,counts


def flatten(row):
    """Prefer shortest spans, then leftmost, preserving stable ties; never relink gold."""
    stats=Counter()
    referenced = {a['entity_id'] for e in row['event_mentions'] for a in e['arguments']}
    stats['unused_entities_removed'] = sum(e['id'] not in referenced for e in row['entity_mentions'])
    row['entity_mentions'] = [e for e in row['entity_mentions'] if e['id'] in referenced]
    occupied=set(); kept=[]
    for entity in sorted(row['entity_mentions'],key=lambda e:(e['end']-e['start'],e['start'],e['id'])):
        positions=set(range(entity['start'],entity['end']))
        if occupied & positions:stats['removed_overlapping_entities']+=1;continue
        occupied.update(positions);kept.append(entity)
    ids={e['id'] for e in kept};row['entity_mentions']=sorted(kept,key=lambda e:e['start'])
    occupied=set();events=[]
    for event in sorted(row['event_mentions'],key=lambda e:(e['trigger']['end']-e['trigger']['start'],e['trigger']['start'],e['id'])):
        t=event['trigger'];positions=set(range(t['start'],t['end']))
        if occupied & positions:stats['removed_overlapping_events']+=1;continue
        occupied.update(positions); seen=set();args=[]
        for arg in sorted(event['arguments'], key=lambda a: -a['role'].count('.')):
            if arg['entity_id'] not in ids:stats['removed_arguments_overlap']+=1;continue
            if arg['entity_id'] in seen:stats['removed_multirole_or_duplicate_arguments']+=1;continue
            seen.add(arg['entity_id']);args.append(arg)
        event['arguments']=args;events.append(event)
    row['event_mentions']=sorted(events,key=lambda e:e['trigger']['start'])
    return stats


def load_splits(root,dataset):
    if dataset != 'VHE':
        return {s:read_rows(root/('val.json' if dataset=='GENEVA' and s=='dev' else s+'.json')) for s in ('train','dev','test')}
    groups=defaultdict(list)
    for row in read_rows(root/'event.json'):
        groups[' '.join(row['text'].split()).casefold()].append(row)
    keys=sorted(groups);random.Random(42).shuffle(keys)
    a=int(len(keys)*.8);b=int(len(keys)*.9)
    return {s:[r for key in part for r in groups[key]] for s,part in zip(('train','dev','test'),(keys[:a],keys[a:b],keys[b:]))}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--dataset',choices=['VHE','PHEE','GENEVA'],required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'dataset':args.dataset,'projection':'shortest non-overlapping spans; single role per event/entity; ARG entity type',
            'split_policy':'seed42 text-grouped 80/10/10' if args.dataset=='VHE' else 'original splits', 'splits':{}}
    manifest={}; all_ids=set()
    for split,rows in load_splits(args.input,args.dataset).items():
        stats=Counter(input_sentences=len(rows));rejected=[];manifest[split]=[]
        with (args.output/(split+'.json')).open('w') as out, (args.output/(split+'.full_gold.jsonl')).open('w') as gold:
            for source in rows:
                sid=str(source.get('wnd_id',source.get('id',source.get('doc_id'))))
                if sid in all_ids:raise ValueError(f'Duplicate ID {sid}')
                all_ids.add(sid);manifest[split].append(sid)
                try:
                    if args.dataset=='GENEVA':
                        row=copy.deepcopy(source);row['sent_id']=sid;row['relation_mentions']=[]
                        for e in row['entity_mentions']:e.update(entity_type='ARG',mention_type='UNK')
                        changes=Counter()
                    else:row,changes=char_record(source,args.dataset)
                    for m in row['entity_mentions']+[e['trigger'] for e in row['event_mentions']]:
                        if not 0<=m['start']<m['end']<=len(row['tokens']):raise ValueError('Invalid token offset')
                    gold.write(json.dumps(row,ensure_ascii=False)+'\n')
                    stats.update(changes);stats['original_events']+=len(row['event_mentions'])
                    stats['original_arguments']+=sum(len(e['arguments']) for e in row['event_mentions'])
                    stats.update(flatten(row));stats['kept_sentences']+=1
                    stats['projected_events']+=len(row['event_mentions'])
                    stats['projected_arguments']+=sum(len(e['arguments']) for e in row['event_mentions'])
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                except (ValueError,KeyError,IndexError) as exc:rejected.append({'id':sid,'reason':str(exc)})
        report['splits'][split]={'counts':dict(stats),'rejected':rejected}
    report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input.glob('*.json')}
    (args.output/'conversion_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (args.output/'split_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
