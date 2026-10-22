import argparse
from collections import Counter
import copy
from pathlib import Path
import shutil
import tempfile

from src.data.prepare_unified import verify_frozen_gold
from src.data.unified import read_rows
from src.baselines.oneie.shared import DATASETS, to_native, dump_json, dump_rows


def flatten(row):
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



def convert(source, output, reference=None):
    reference = reference or source / 'manifest.json'
    verify_frozen_gold(source, reference)
    if output.exists():
        raise FileExistsError(f'{output} exists; choose a new output')
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.oneie-', dir=output.parent))
    report = {'source': str(source), 'reference': str(reference), 'splits': {}}
    try:
        for dataset in DATASETS:
            for split in ('train', 'dev', 'test'):
                rows = read_rows(source / dataset / f'{split}.jsonl')
                native = []; full = []; counts = Counter(samples=len(rows))
                for row in rows:
                    graph = to_native(row)
                    full.append(copy.deepcopy(graph))
                    counts['events_before'] += len(graph['event_mentions'])
                    counts['arguments_before'] += sum(len(e['arguments']) for e in graph['event_mentions'])
                    counts.update(flatten(graph))
                    counts['events_after'] += len(graph['event_mentions'])
                    counts['arguments_after'] += sum(len(e['arguments']) for e in graph['event_mentions'])
                    native.append(graph)
                dump_rows(stage / dataset / f'{split}.jsonl', native)
                # Preserve exact gold bytes, including discontinuous mention metadata.
                gold = stage / dataset / 'gold' / f'{split}.jsonl'
                gold.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / dataset / f'{split}.jsonl', gold)
                dump_rows(stage / dataset / 'native_gold' / f'{split}.jsonl', full)
                report['splits'][f'{dataset}/{split}'] = dict(counts)
        dump_json(stage / 'conversion_report.json', report)
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Verified curated Unified root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, help='Optional reference; defaults to the input manifest')
    args = parser.parse_args()
    report = convert(args.input, args.output, args.reference_manifest)
    print(f"Converted {len(report['splits'])} splits; full gold preserved")


if __name__ == '__main__':
    main()
