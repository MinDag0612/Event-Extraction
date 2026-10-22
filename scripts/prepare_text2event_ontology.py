"""Build reproducible decoder vocabularies without reading DEV/TEST annotations.

The default allows any known role for any known type. This is a deliberately
relaxed constraint, not a claim that all combinations are official ontology.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import tempfile
import shutil
from src.adapter.GENEVA_adepter import GENEVAAdapter
from src.adapter.VHE_adapter import VHEAdapter
from src.unified_format.schema import Schema
from src.unified_format.event_schema import EventSchema

SOURCES = {
 'GENEVA': ('PlusLabNLP/GENEVA', '559e21e98ed16b00422b6cba91750c33cfe4bbcc', 'meta_data/fn2geneva_mapping_annotations_with_time_place.tsv'),
 'VHE': ('hoangthuytruc/vhe-dataset', 'd5d7b03d016f1c25d33748424251cf346e5455bc', 'event-extraction/vhe_event_schema.json'),
}


def build(root, output):
    output.mkdir(parents=True, exist_ok=True)
    provenance = {'policy': 'known event types x known roles; no per-event role restriction',
                  'uses_held_out_annotations': False, 'datasets': {}}
    for name in ['BKEE','GENEVA','PHEE','VHE']:
        path = root / name / 'train.jsonl'; blob = path.read_bytes()
        events = [e for line in blob.splitlines() for e in json.loads(line)['events']]
        types = {e['event_type'] for e in events}
        roles = {a['role'] for e in events for a in e['arguments']}
        entry = {'train_sha256': hashlib.sha256(blob).hexdigest(), 'official_source': None,
                 'limitation': 'TRAIN vocabulary only; not verified as a complete official ontology'}
        if name in SOURCES:
            repo, rev, file = SOURCES[name]
            url = f'https://raw.githubusercontent.com/{repo}/{rev}/{file}'
            raw = urllib.request.urlopen(url, timeout=60).read()
            (output / (name + '.source')).write_bytes(raw)
            if name == 'GENEVA':
                table = list(csv.reader(io.StringIO(raw.decode('utf-8-sig')), delimiter='\t'))
                header = table[0]
                fixed = []
                for row in table[1:]:
                    # Pinned upstream TSV has an extra annotator field after definition.
                    if len(row) == len(header) + 1:
                        row = row[:5] + row[6:]
                    if len(row) != len(header):
                        raise ValueError('Unexpected GENEVA TSV columns')
                    fixed.append(dict(zip(header, row)))
                schema = GENEVAAdapter().get_schema(fixed)
            else:
                schema = VHEAdapter().get_schema([json.loads(line) for line in raw.splitlines() if line.strip()])
            (output / (name + '.official.json')).write_text(schema.to_json()+'\n')
            types.update(e.event_type for e in schema.event_schemas)
            roles.update(r for e in schema.event_schemas for r in e.argument_roles)
            entry.update(official_source=url, official_sha256=hashlib.sha256(raw).hexdigest(),
                         limitation='Official vocabulary plus TRAIN; role pairing intentionally relaxed')
        schema = Schema([EventSchema(t, sorted(roles)) for t in sorted(types)])
        (output / (name + '.json')).write_text(schema.to_json()+'\n')
        entry.update(event_types=len(types), roles=len(roles))
        provenance['datasets'][name] = entry
    (output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+'\n')

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('data/curated-v2'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(dir=a.output.parent))
    try:
        build(a.input,staging)
        staging.rename(a.output)
    finally:
        if staging.exists(): shutil.rmtree(staging)
