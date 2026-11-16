"""Convert unified GENEVA annotations through the author's Text2Tree formatter."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

PIN='dbe5a0ae933d639980c8e1f62c141511c12cefd6'


def label_maps(types,roles):
    """Keep label semantics while separating event/role names and terminating trie labels."""
    mapping={'event':{name:name+'__evt' for name in types},'role':{name:name+'__arg' for name in roles}}
    values=[v for group in mapping.values() for v in group.values()]
    if len(values)!=len(set(values)) or any(any(c.isspace() for c in v) for v in values):
        raise ValueError('Non-bijective label mapping')
    return mapping


def predicate_arguments(row,mapping):
    """Expand trigger and argument mentions into the author's indexed event records."""
    result=[];tokens=row['tokens']
    def indices(mention):
        a,b=mention['span']
        if not 0<=a<b<=len(tokens):raise ValueError('Invalid source span')
        return list(range(a,b))
    for event in row['events']:
        arguments=[(mapping['role'][argument['role']],indices(mention))
                   for argument in event['arguments'] for mention in argument['mentions']]
        for trigger in event['trigger']:
            result.append({'type':mapping['event'][event['event_type']],
                           'tokens':indices(trigger),'arguments':arguments})
    return result


def prepare(bundle,author,output):
    """Export full/span datasets, reversible sidecars and fixed smoke subsets."""
    output.mkdir(parents=True,exist_ok=False)
    lock=json.loads((Path('configs/text2event_author')/'source-lock.json').read_text())
    for name,digest in lock['files'].items():
        if hashlib.sha256((author/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Author source changed: '+name)
    sys.path.insert(0,str(author.resolve()))
    formatter=importlib.import_module('data_convert.format.text2tree').Text2Tree
    folder=bundle/'GENEVA';types,roles,type_roles=map(json.loads,(folder/'event.schema').read_text().splitlines())
    mapping=label_maps(types,roles)
    schema=[[mapping['event'][v] for v in types],[mapping['role'][v] for v in roles],
            {mapping['event'][k]:[mapping['role'][v] for v in vs] for k,vs in type_roles.items()}]
    report={'source_commit':PIN,'label_mapping':'semantic label plus kind-specific terminal suffix',
            'schema_source':'declared unified bundle schema','splits':{},'smoke_selection_seed':421}
    for split,outname in [('train','train'),('dev','val'),('test','test')]:
        gold=list(map(json.loads,(folder/(split+'.gold.jsonl')).read_text().splitlines()))
        if len({r['id'] for r in gold})!=len(gold):raise ValueError('Duplicate sample IDs')
        encoded={'full':[],'span':[]};collisions=0;arguments=0;triggers=0
        for row in gold:
            events=predicate_arguments(row,mapping)
            collisions+=sum(a['role']==e['event_type'] for e in row['events'] for a in e['arguments'])
            triggers+=len(events);arguments+=sum(len(e['arguments']) for e in events)
            for stage,method in [('full',formatter.annotate_predicate_arguments),('span',formatter.annotate_span)]:
                text,target=method(tokens=row['tokens'],predicate_arguments=events)
                if target.count('<extra_id_0>')!=1+len(events)+sum(len(e['arguments']) for e in events):
                    raise ValueError('Annotation omitted by author formatter: '+row['id'])
                encoded[stage].append({'text':text,'event':target})
        report['splits'][split]={'rows':len(gold),'serialized_triggers':triggers,
            'serialized_argument_mentions':arguments,'original_equal_type_role_argument_objects':collisions}
        subset=sorted(random.Random(421).sample(range(len(gold)),min(8,len(gold)))) if split!='test' else []
        if split!='test':report['splits'][split]['smoke_ids']=[gold[i]['id'] for i in subset]
        for stage,data in encoded.items():
            for smoke,items in [(False,data)]+([(True,[data[i] for i in subset])] if split!='test' else []):
                name=('GENEVA-smoke' if smoke else 'GENEVA')+('_span' if stage=='span' else '')
                destination=output/name;destination.mkdir(exist_ok=True)
                (destination/(outname+'.json')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in items))
                (destination/'event.schema').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in schema))
        (output/(split+'.original.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in gold))
    (output/'label-map.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
    (output/'adapter-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--author',type=Path,default=Path('third_party/Text2Event-author'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.bundle,a.author,a.output),ensure_ascii=False,indent=2))
