import hashlib
import json
from collections import defaultdict
from pathlib import Path

from transformers import BertTokenizer
from src.adapter.BKEE_adapter import BKEEAdapter
from src.adapter.GENEVA_adepter import GENEVAAdapter
from src.adapter.RAMS_adapter import RAMSAdapter
from src.adapter.VHE_adapter import VHEAdapter

ROOT = Path(__file__).resolve().parents[2]
BACKBONE = 'bert-base-multilingual-cased'


def export_record(sample, raw, tokenizer, uid):
    entities, by_span = [], {}
    notes = []
    def entity(span, label='ENTITY'):
        start, end = map(int, span)
        if not 0 <= start < end <= len(sample.tokens):
            raise ValueError(f'invalid token span {span}')
        key = start, end
        if key not in by_span:
            obj = dict(id=f'{uid}-E{len(entities)}', start=start, end=end,
                       text=' '.join(sample.tokens[start:end]),
                       entity_type=label, mention_type='UNK')
            by_span[key] = obj
            entities.append(obj)
        return by_span[key]
    # Preserve source entities, including entities without event arguments.
    for ent in raw.get('entity_mentions', []):
        entity((ent['start'], ent['end']), ent.get('entity_type', 'ENTITY'))
    for ent in raw.get('ent_spans', []):
        entity((ent[0], ent[1] + 1))  # RAMS labels are roles, not NER types.
    events = []
    for ev in sample.events:
        args = []
        roles = {}
        for arg in ev.arguments:
            for mention in arg.mentions:
                ent = entity(mention['span'])
                if ent['id'] in roles and roles[ent['id']] != arg.role:
                    notes.append(dict(kind='competing_role', annotation=dict(event_type=ev.event_type, entity_id=ent['id'], kept_role=roles[ent['id']], removed_role=arg.role)))
                    continue
                if ent['id'] not in roles:
                    args.append(dict(entity_id=ent['id'], text=ent['text'], role=arg.role))
                    roles[ent['id']] = arg.role
        for trigger in ev.trigger:
            start, end = trigger.span
            if not 0 <= start < end <= len(sample.tokens):
                raise ValueError('invalid trigger span')
            events.append(dict(id=f'{uid}-EV{len(events)}', event_type=ev.event_type,
                trigger=dict(start=start, end=end, text=' '.join(sample.tokens[start:end])),
                arguments=args))
    # Deterministic flat BIO projection: prioritize argument entities, then
    # longer spans; keep source event order for competing triggers.
    referenced = {a['entity_id'] for e in events for a in e['arguments']}
    occupied = set(); kept = []
    for ent in sorted(entities, key=lambda e: (e['id'] not in referenced, -(e['end']-e['start']), e['start'], e['id'])):
        indices = set(range(ent['start'], ent['end']))
        if occupied & indices:
            notes.append(dict(kind='overlap_entity', annotation=ent))
        else:
            occupied.update(indices); kept.append(ent)
    entities = sorted(kept, key=lambda e: e['start'])
    ids = {e['id'] for e in entities}
    occupied = set(); kept = []
    for ev in events:
        indices = set(range(ev['trigger']['start'], ev['trigger']['end']))
        if occupied & indices:
            notes.append(dict(kind='overlap_event', annotation=ev))
            continue
        occupied.update(indices)
        for arg in ev['arguments']:
            if arg['entity_id'] not in ids:
                notes.append(dict(kind='removed_argument_link', annotation=arg))
        ev['arguments'] = [a for a in ev['arguments'] if a['entity_id'] in ids]
        kept.append(ev)
    events = kept
    pieces = [tokenizer.tokenize(t) or [tokenizer.unk_token] for t in sample.tokens]
    flat = [p for group in pieces for p in group]
    if not flat or len(flat) > 510:
        raise ValueError(f'wordpiece length {len(flat)} outside 1..510')
    if tokenizer.encode(flat, add_special_tokens=True) != [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(flat) + [tokenizer.sep_token_id]:
        raise ValueError('tokenizer encode mapping mismatch')
    return dict(doc_id=str(raw.get('doc_id', raw.get('doc_key', sample.id))),
                sent_id=uid, sentence=sample.text, tokens=sample.tokens, pieces=flat,
                token_lens=[len(p) for p in pieces], entity_mentions=entities,
                event_mentions=events, relation_mentions=[], conversion_notes=notes)


def main():
    tokenizer = BertTokenizer.from_pretrained(BACKBONE, do_lower_case=False)
    out = ROOT / 'data/oneie'
    tokenizer.save_pretrained(out / 'tokenizer')
    adapters = dict(BKEE=BKEEAdapter(), GENEVA=GENEVAAdapter(), RAMS=RAMSAdapter(), VHE=VHEAdapter())
    for name, adapter in adapters.items():
        dest = out / name
        dest.mkdir(parents=True, exist_ok=True)
        rows = defaultdict(list)
        if name == 'VHE':
            # No split files are published at pinned upstream commit.
            # Hash text, not labels: identical texts cannot cross split boundaries.
            for i, raw in enumerate(json.loads((ROOT/'data/raw/VHE/event.json').read_text(encoding='utf-8'))):
                bucket = int(hashlib.sha256(('42:' + raw['text']).encode()).hexdigest(), 16) % 100
                split = 'train' if bucket < 80 else 'dev' if bucket < 90 else 'test'
                rows[split].append((i, raw))
        else:
            for split in ('train', 'dev', 'test'):
                filename = ('val' if name == 'GENEVA' and split == 'dev' else split) + ('.jsonlines' if name == 'RAMS' else '.json')
                with (ROOT/'data/raw'/name/filename).open(encoding='utf-8') as f:
                    rows[split] = list(enumerate(json.loads(line) for line in f if line.strip()))
        report, rejected, modified = {}, [], []
        event_role, role_entity = defaultdict(set), defaultdict(set)
        labels = {k: set() for k in ('event', 'role', 'entity')}
        for split, records in rows.items():
            count = 0
            with (dest/f'{split}.jsonl').open('w', encoding='utf-8') as f:
                for i, raw in records:
                    uid = f'{name}-{split}-{i}'
                    try:
                        result = export_record(adapter.adapt(raw), raw, tokenizer, uid)
                    except (ValueError, KeyError) as exc:
                        rejected.append(dict(id=uid, source_id=raw.get('id', raw.get('doc_id', raw.get('doc_key'))), reason=str(exc)))
                        continue
                    notes = result.pop('conversion_notes')
                    if notes: modified.append(dict(id=uid, notes=notes))
                    f.write(json.dumps(result, ensure_ascii=False)+'\n')
                    count += 1
                    if split == 'train':
                        entity_types = {e['id']: e['entity_type'] for e in result['entity_mentions']}
                        labels['entity'].update(entity_types.values())
                        for e in result['event_mentions']:
                            labels['event'].add(e['event_type'])
                            for a in e['arguments']:
                                labels['role'].add(a['role'])
                                event_role[e['event_type']].add(a['role'])
                                role_entity[a['role']].add(entity_types[a['entity_id']])
            report[split] = dict(source=len(records), exported=count, rejected=len(records)-count)
        patterns = dest/'patterns'; patterns.mkdir(exist_ok=True)
        for filename, value in [('event_role',event_role), ('role_entity',role_entity), ('relation_entity',{})]:
            (patterns/f'{filename}.json').write_text(json.dumps({k:sorted(v) for k,v in value.items()},indent=2),encoding='utf-8')
        (dest/'labels.json').write_text(json.dumps({k:sorted(v) for k,v in labels.items()},indent=2),encoding='utf-8')
        (dest/'rejected.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2),encoding='utf-8')
        (dest/'modified.json').write_text(json.dumps(modified,ensure_ascii=False,indent=2),encoding='utf-8')
        (dest/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        cfg=json.loads((ROOT/'external/OneIE/config/example.json').read_text())
        cfg.update(bert_model_name=BACKBONE, bert_cache_dir=str(ROOT/'data/oneie/model-cache'),
                   use_extra_bert=False, use_global_features=False, global_features=['role_entity'],
                   symmetric_relations=[], batch_size=4, accumulate_step=4, eval_batch_size=1,
                   max_epoch=10, warmup_epoch=1, beam_size=5, max_length=512,
                   log_path=str(ROOT/'runs/oneie'/name), valid_pattern_path=str(patterns))
        for split in ('train','dev','test'): cfg[f'{split}_file']=str(dest/f'{split}.jsonl')
        Path(cfg['log_path']).mkdir(parents=True,exist_ok=True)
        (ROOT/f'configs/oneie/{name}.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        print(name,report,flush=True)


if __name__ == '__main__':
    main()
