import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter.BKEE_adapter import BKEEAdapter
from src.adapter.GENEVA_adepter import GENEVAAdapter
from src.adapter.PHEE_adapter import PHEEAdapter
from src.adapter.VHE_adapter import VHEAdapter
from src.data.unified import from_dict, normalize, read_rows, validate

SOURCES = {
    'BKEE': (BKEEAdapter, {'train': 'train', 'dev': 'dev', 'test': 'test'}),
    'GENEVA': (GENEVAAdapter, {'train': 'train', 'dev': 'val', 'test': 'test'}),
    'PHEE': (PHEEAdapter, {'train': 'train', 'dev': 'dev', 'test': 'test'}),
    'VHE': (VHEAdapter, {'unsplit': 'event'}),
}
VERSION = 'curated-v2'
SEED = '42'


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def vhe_split(text):
    # Text groups stay together; independent of source order and gold annotations.
    bucket = int(digest(SEED + '\n' + normalize(text))[:16], 16) % 100
    return 'train' if bucket < 80 else 'dev' if bucket < 90 else 'test'


def event_signature(row):
    # Ignore event/argument ordering but retain spans, multiplicity and PHEE groups.
    events = []
    for event in row['events']:
        triggers = sorted(json.dumps(t, sort_keys=True, ensure_ascii=False) for t in event['trigger'])
        arguments = []
        for argument in event['arguments']:
            mentions = sorted(json.dumps(m, sort_keys=True, ensure_ascii=False) for m in argument['mentions'])
            arguments.append((argument['role'], mentions))
        events.append((event['event_type'], triggers, sorted(arguments)))
    return json.dumps([row['tokens'], sorted(events)], ensure_ascii=False)


def curate(records, dispositions):
    """Preserve higher-priority held-out inputs, then remove ambiguous duplicates."""
    result = {split: [] for split in ('train', 'dev', 'test')}
    seen = set()
    for split in ('test', 'dev', 'train'):
        groups = defaultdict(list)
        for row in records[split]:
            groups[normalize(row['text'])].append(row)
        for text, group in sorted(groups.items()):
            reason = None
            if text in seen:
                reason = 'text_overlap_with_higher_priority_split'
            elif len({event_signature(row) for row in group}) > 1:
                reason = 'conflicting_annotations_for_same_text'
            seen.add(text)  # Also reserve quarantined held-out texts.
            if reason:
                for row in group:
                    dispositions[row['id']].update(status='excluded', reason=reason)
                continue
            group.sort(key=lambda row: row['id'])
            result[split].append(group[0])
            dispositions[group[0]['id']].update(status='included', output_split=split)
            for row in group[1:]:
                dispositions[row['id']].update(status='excluded', reason='duplicate_within_split',
                                               retained_id=group[0]['id'])
        result[split].sort(key=lambda row: row['id'])
    return result


def build(raw_root, output):
    manifest = {'version': VERSION, 'span_unit': 'original token indices, half-open [start,end)',
                'policy': 'strict spans; whole-record quarantine; TEST > DEV > TRAIN text priority; conflicting same-text labels excluded',
                'vhe_split': {'algorithm': 'sha256(seed + newline + normalize) first 64 bits mod 100; 0:80 train, 80:90 dev, 90:100 test', 'seed': SEED},
                'sources': {}, 'code_sha256': {}, 'records': []}
    for path in sorted((ROOT / 'src/adapter').glob('*.py')) + [Path(__file__), ROOT / 'src/data/unified.py']:
        manifest['code_sha256'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {'version': VERSION, 'datasets': {}, 'ready_for_slm_data_preparation': True,
               'full_training_pipeline_ready': False}
    for name, (adapter_type, splits) in SOURCES.items():
        adapter = adapter_type()
        records = {split: [] for split in ('train', 'dev', 'test')}
        dispositions = {}
        quarantine = []
        raw_count = 0
        for split, source_split in splits.items():
            path = raw_root / name / f'{source_split}.json'
            manifest['sources'][f'{name}/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            for index, raw in enumerate(read_rows(path)):
                raw_count += 1
                original_id = raw.get('wnd_id') if name == 'GENEVA' else raw.get('id', raw.get('doc_id'))
                record_id = f'{name}:{source_split}:{index + 1:06d}'
                if record_id in dispositions:
                    raise ValueError(f'{name}: missing or duplicate source ID {original_id}')
                text = raw.get('context', raw.get('text', raw.get('sentence', '')))
                target_split = vhe_split(text) if split == 'unsplit' else split
                entry = {'id': record_id, 'source': f'{name}/{path.name}', 'source_index': index,
                         'source_id': str(original_id), 'assigned_split': target_split,
                         'text_sha256': digest(normalize(text))}
                dispositions[record_id] = entry
                try:
                    sample = adapter.adapt(raw)
                    row = json.loads(sample.to_json())
                    row['id'] = record_id
                    source_events = ([e for a in raw['annotations'] for e in a['events']] if name == 'PHEE'
                                     else raw.get('events', raw.get('event_mentions', [])))
                    if len(row['events']) != len(source_events):
                        raise ValueError('Adapter lost events')
                    validate(row)
                    if json.loads(from_dict(row).to_json()) != row:
                        raise ValueError('Unified round-trip changed the record')
                    if name in ('PHEE', 'VHE'):
                        blank = dict(raw, **({'annotations': []} if name == 'PHEE' else {'events': []}))
                        if adapter.adapt(blank).tokens != row['tokens']:
                            raise ValueError('Tokenization depends on gold annotations')
                    records[target_split].append(row)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    entry.update(status='quarantined', reason=str(exc))
                    quarantine.append(dict(entry, raw=raw))
        # Quarantined held-out inputs must not reappear in training.
        reserved = {split: set() for split in records}
        for entry in dispositions.values():
            reserved[entry['assigned_split']].add(entry['text_sha256'])
        for split, higher in [('train', ('dev', 'test')), ('dev', ('test',))]:
            hashes = set().union(*(reserved[key] for key in higher))
            retained = []
            for row in records[split]:
                if digest(normalize(row['text'])) in hashes:
                    dispositions[row['id']].update(status='excluded', reason='text_overlap_with_higher_priority_split')
                else:
                    retained.append(row)
            records[split] = retained
        curated = curate(records, dispositions)
        schema = defaultdict(set)
        for row in curated['train']:
            for event in row['events']:
                schema[event['event_type']].update(a['role'] for a in event['arguments'])
        stats = {}
        for split, rows in curated.items():
            if not rows:
                raise ValueError(f'{name}/{split}: empty curated split')
            write_jsonl(output / name / f'{split}.jsonl', rows)
            unseen_types, unseen_roles = set(), set()
            for row in rows:
                for event in row['events']:
                    kind = event['event_type']
                    if kind not in schema:
                        unseen_types.add(kind)
                    for argument in event['arguments']:
                        if argument['role'] not in schema.get(kind, set()):
                            unseen_roles.add((kind, argument['role']))
            stats[split] = {'samples': len(rows), 'events': sum(len(r['events']) for r in rows),
                            'negative_samples': sum(not r['events'] for r in rows),
                            'unseen_event_types_in_train': sorted(unseen_types),
                            'unseen_event_role_pairs_in_train': sorted(unseen_roles)}
        write_json(output / name / 'train_schema.json',
                   {'source': 'curated TRAIN only; not a complete official ontology',
                    'event_roles': {key: sorted(values) for key, values in sorted(schema.items())}})
        write_jsonl(output / name / 'quarantine.jsonl', quarantine)
        counts = Counter(entry['status'] for entry in dispositions.values())
        if sum(counts.values()) != raw_count:
            raise ValueError('Unaccounted source records')
        summary['datasets'][name] = {'raw_samples': raw_count, 'dispositions': dict(counts), 'splits': stats}
        manifest['records'].extend(dispositions.values())
    # Verify saved files across every dataset/split, including deserialization.
    text_splits = defaultdict(set)
    all_ids = set()
    for name in SOURCES:
        for split in ('train', 'dev', 'test'):
            for row in read_rows(output / name / f'{split}.jsonl'):
                from_dict(row)
                if row['id'] in all_ids:
                    raise ValueError(f'Duplicate exported ID: {row["id"]}')
                all_ids.add(row['id'])
                text_splits[normalize(row['text'])].add(split)
    if any(len(splits) > 1 for splits in text_splits.values()):
        raise ValueError('Cross-dataset/split input overlap in exported data')
    manifest['records'].sort(key=lambda entry: entry['id'])
    manifest['outputs'] = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted(output.rglob('*')) if path.is_file()}
    write_json(output / 'manifest.json', manifest)
    write_json(output / 'readiness.json', summary)
    return summary


def verify_frozen_gold(output, reference):
    expected = json.loads(reference.read_text(encoding='utf-8'))['files']
    required = {f'{name}/{split}.jsonl' for name in SOURCES for split in ('train', 'dev', 'test')}
    if set(expected) != required:
        raise ValueError('Reference manifest must list exactly the 12 frozen gold files')
    for name in sorted(required):
        payload = (output / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected[name]['sha256']:
            raise ValueError(f'Rebuilt gold differs from frozen reference: {name}')
        if len(read_rows(output / name)) != expected[name]['samples']:
            raise ValueError(f'Rebuilt sample count differs: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root', type=Path, default=ROOT / 'data/raw')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/processed/curated-v2-rebuilt')
    parser.add_argument('--reference-manifest', type=Path, default=ROOT / 'data/curated-v2/manifest.json',
                        help='Require byte-identical gold to this frozen manifest before publishing output')
    args = parser.parse_args()
    output = args.output.resolve()
    raw_root = args.raw_root.resolve()
    if output == raw_root or raw_root in output.parents or output in raw_root.parents:
        raise ValueError('Output must be separate from raw data')
    if output.exists():
        raise FileExistsError(f'{output} exists; use a new --output path to preserve the frozen manifest')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.unified-', dir=output.parent))
    try:
        summary = build(raw_root, staging)
        verify_frozen_gold(staging, args.reference_manifest)
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('PASS: all 12 gold files match the frozen reference SHA-256 and sample counts')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
