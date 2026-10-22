"""Export frozen curated gold to upstream Text2Event JSONL + three-line event.schema."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import shutil
from src.baselines.text2event.convert import convert, parse, records, to_unified, AmbiguousSpan, write_schema
from src.unified_format.schema import Schema
from src.unified_format.event_schema import EventSchema
from src.metrics.evaluator import Evaluator


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def build(root, output, ontology_dir=None, train_observed=False):
    manifest = json.loads((root / 'manifest.json').read_text())
    report = {'upstream_revision': 'dbe5a0ae933d639980c8e1f62c141511c12cefd6',
              'source_text': 'space-joined original tokens, as upstream',
              'schema_source': 'train-observed (incomplete)' if train_observed else 'explicit ontology files',
              'ready_for_training': False, 'datasets': {}}
    if ontology_dir is not None and (ontology_dir / 'provenance.json').exists():
        provenance = json.loads((ontology_dir / 'provenance.json').read_text())
        report['ontology_provenance'] = provenance
        dump(output / 'ontology-provenance.json', provenance)
    for name in ['BKEE', 'GENEVA', 'PHEE', 'VHE']:
        rows = {}
        for split in ['train', 'dev', 'test']:
            rel = f'{name}/{split}.jsonl'; blob = (root / rel).read_bytes()
            if hashlib.sha256(blob).hexdigest() != manifest['files'][rel]['sha256']:
                raise ValueError(f'Frozen data checksum mismatch: {rel}')
            rows[split] = [json.loads(line) for line in blob.splitlines()]
        if train_observed:
            schema = Schema([EventSchema(e['event_type'], [a['role'] for a in e['arguments']])
                             for row in rows['train'] for e in row['events']])
        else:
            schema = Schema.from_dict(json.loads((ontology_dir / f'{name}.json').read_text()))
        folder = output / name; folder.mkdir()
        write_schema(schema, folder / 'event.schema')
        mapping = {e.event_type: set(e.argument_roles) for e in schema.event_schemas}
        result = {'splits': {}, 'schema_sha256': hashlib.sha256((folder / 'event.schema').read_bytes()).hexdigest()}
        for split, samples in rows.items():
            ambiguous = []; unknown = set(); evaluated = 0
            with (folder / f'{split}.json').open('w') as target, (folder / f'{split}.gold.jsonl').open('w') as sidecar:
                for row in samples:
                    converted = convert(row)
                    if parse(converted['event']) != records(row):
                        raise ValueError(f'Tree roundtrip failed: {row["id"]}')
                    for e in row['events']:
                        if e['event_type'] not in mapping:
                            unknown.add((e['event_type'], None))
                        for a in e['arguments']:
                            if a['role'] not in mapping.get(e['event_type'], set()):
                                unknown.add((e['event_type'], a['role']))
                    try:
                        recovered = to_unified(converted['event'], row['id'], row['text'], row['tokens'])
                        evaluator = Evaluator(); evaluator.evaluate_on_all_sample([recovered], [row])
                        if any(m.fp or m.fn for m in [evaluator.ti, evaluator.tc, evaluator.ai, evaluator.ac]):
                            raise ValueError(f'Metric roundtrip failed: {row["id"]}')
                        evaluated += 1
                    except AmbiguousSpan as exc:
                        ambiguous.append({'id': row['id'], 'reason': str(exc)})
                    target.write(json.dumps(converted, ensure_ascii=False) + '\n')
                    sidecar.write(json.dumps(row, ensure_ascii=False) + '\n')
            dump(folder / f'{split}.ambiguities.json', ambiguous)
            result['splits'][split] = {'samples': len(samples), 'tree_roundtrip': len(samples),
                'exact_span_roundtrip': evaluated, 'ambiguous_samples': len(ambiguous),
                'uncovered_schema_pairs': sorted(unknown, key=str)}
        report['datasets'][name] = result
    dump(output / 'report.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('data/curated-v2'))
    p.add_argument('--output', type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--ontology-dir', type=Path)
    group.add_argument('--train-observed-schema', action='store_true', help='Audit only: TRAIN-derived schema may omit valid held-out pairs')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=args.output.parent))
    try:
        report = build(args.input, temp, args.ontology_dir, args.train_observed_schema)
        temp.rename(args.output)
    finally:
        if temp.exists(): shutil.rmtree(temp)
    for name, info in report['datasets'].items():
        print(name, {s: (v['samples'], v['ambiguous_samples']) for s,v in info['splits'].items()})
    print('Export complete; training readiness remains false pending span/schema/tokenizer checks.')

if __name__ == '__main__':
    main()
