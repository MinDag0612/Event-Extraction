from collections import Counter
import json
from src.data.unified import validate

DATASETS = ('BKEE', 'GENEVA', 'PHEE', 'VHE')
PROTOCOL = 'event-span-micro-v1: AI=(event_type,arg_span), AC=AI+role; fragment-level; set matching'


def to_native(row):
    validate(row)
    entities = {}; events = []
    for event in row['events']:
        if len(event['trigger']) != 1:
            raise ValueError(f"{row['id']}: OneIE requires a single contiguous trigger")
        arguments = []
        for argument in event['arguments']:
            for mention in argument['mentions']:
                a, b = mention['span']; key = (a, b)
                if key not in entities:
                    entities[key] = dict(id=f"{row['id']}-E{len(entities)}", start=a, end=b,
                                         text=mention['text'], entity_type='ARG', mention_type='UNK')
                arguments.append(dict(entity_id=entities[key]['id'], role=argument['role'], text=mention['text']))
        t = event['trigger'][0]
        events.append(dict(id=f"{row['id']}-T{len(events)}", event_type=event['event_type'],
                           trigger=dict(text=t['text'], start=t['span'][0], end=t['span'][1]), arguments=arguments))
    return dict(doc_id=row['id'], sent_id=row['id'], sentence=row['text'], tokens=list(row['tokens']),
                entity_mentions=list(entities.values()), event_mentions=events, relation_mentions=[])


def graph_to_unified(graph, row, vocabs):
    event_labels = {v: k for k, v in vocabs['event_type'].items()}
    role_labels = {v: k for k, v in vocabs['role_type'].items()}
    events = []
    for a, b, label in graph.triggers:
        events.append(dict(event_type=event_labels[label],
                           trigger=[dict(text=' '.join(row['tokens'][a:b]), span=[a,b])], arguments=[]))
    for trigger_index, entity_index, role in graph.roles:
        a,b,_ = graph.entities[entity_index]
        events[trigger_index]['arguments'].append(dict(role=role_labels[role],
            mentions=[dict(text=' '.join(row['tokens'][a:b]), span=[a,b])]))
    prediction = dict(id=row['id'], text=row['text'], tokens=list(row['tokens']), events=events)
    validate(prediction)
    return prediction


def metric_sets(row):
    result = {key: set() for key in ('TI','TC','AI','AC')}
    for event in row['events']:
        kind = event['event_type']
        for trigger in event['trigger']:
            span = tuple(trigger['span'])
            result['TI'].add(span); result['TC'].add((kind, span))
        for argument in event['arguments']:
            for mention in argument['mentions']:
                span = tuple(mention['span'])
                result['AI'].add((kind,span))
                result['AC'].add((kind,span,argument['role']))
    return result


def score_samples(predictions, gold):
    def index(rows):
        result = {}
        for row in rows:
            validate(row)
            if row['id'] in result: raise ValueError('Duplicate evaluation ID')
            result[row['id']] = row
        return result
    pred = index(predictions); reference = index(gold)
    if pred.keys() != reference.keys(): raise ValueError('Prediction/gold ID coverage mismatch')
    counts = {key: Counter() for key in ('TI','TC','AI','AC')}
    for rid,g in reference.items():
        p = pred[rid]
        if p['tokens'] != g['tokens'] or p['text'] != g['text']: raise ValueError('Prediction input mismatch')
        ps, gs = metric_sets(p),metric_sets(g)
        for key,c in counts.items():
            c.update(tp=len(ps[key]&gs[key]), fp=len(ps[key]-gs[key]), fn=len(gs[key]-ps[key]))
    result = {}
    for key,c in counts.items():
        tp,fp,fn = c['tp'],c['fp'],c['fn']
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        result[key]=dict(c,precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return result


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def dump_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
