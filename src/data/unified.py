import copy
import json
import re
from pathlib import Path
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.trigger import Trigger
from src.unified_format.argument import Argument


def read_rows(path):
    text = Path(path).read_text(encoding='utf-8')
    if not text.strip():
        return []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(rows, dict):
        rows = rows.get('data', [rows])
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError(f'{path}: expected objects')
    return rows


def normalize(text):
    return ' '.join(text.split())


def validate(row):
    def fields(value, required, optional=()):
        if not isinstance(value, dict) or not set(required) <= value.keys() or set(value) - set(required) - set(optional):
            raise ValueError('Invalid Unified fields')

    def string(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Expected nonempty string')

    def array(value):
        if not isinstance(value, list):
            raise ValueError('Expected array')

    def span(value, limit):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('Expected two span boundaries')
        a, b = value
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= limit:
            raise ValueError('Invalid span boundaries')
        return a, b

    fields(row, ('id', 'text', 'tokens', 'events'))
    string(row['id']); string(row['text'])
    tokens = row['tokens']; array(tokens)
    if not tokens:
        raise ValueError('Missing tokens')
    for token in tokens:
        string(token)
    array(row['events'])
    for event in row['events']:
        fields(event, ('event_type', 'trigger', 'arguments'))
        string(event['event_type']); array(event['trigger']); array(event['arguments'])
        if not event['trigger']:
            raise ValueError('Missing trigger')
        mentions = []
        for trigger in event['trigger']:
            fields(trigger, ('text', 'span'))
            mentions.append(trigger)
        for arg in event['arguments']:
            fields(arg, ('role', 'mentions'))
            string(arg['role']); array(arg['mentions'])
            if not arg['mentions']:
                raise ValueError('Missing mentions')
            groups = {}
            for m in arg['mentions']:
                fields(m, ('text', 'span'), ('char_span', 'mention_group', 'fragment_index'))
                if ('mention_group' in m) != ('fragment_index' in m):
                    raise ValueError('Incomplete fragment metadata')
                if 'mention_group' in m:
                    group, index = m['mention_group'], m['fragment_index']
                    if any(type(v) is not int or v < 0 for v in (group, index)):
                        raise ValueError('Invalid fragment metadata')
                    indices = groups.setdefault(group, [])
                    indices.append(index)
                mentions.append(m)
            for indices in groups.values():
                if sorted(indices) != list(range(len(indices))):
                    raise ValueError('Duplicate or missing fragment index')
        for m in mentions:
            string(m['text'])
            a, b = span(m['span'], len(tokens))
            if re.sub(r'\s', '', m['text']) != re.sub(r'\s', '', ''.join(tokens[a:b])):
                raise ValueError(f'Mention/token mismatch: {m["text"]!r} at {a}:{b}')
            if 'char_span' in m:
                start, end = span(m['char_span'], len(row['text']))
                if row['text'][start:end] != m['text']:
                    raise ValueError('Invalid character mention')


def from_dict(row):
    validate(row)
    row = copy.deepcopy(row)
    return EventExtractionData(id=row['id'], text=row['text'], tokens=row['tokens'], events=[
        Event(event_type=e['event_type'], trigger=[Trigger(text=t['text'], span=tuple(t['span'])) for t in e['trigger']],
              arguments=[Argument(role=a['role'], mentions=a['mentions']) for a in e['arguments']])
        for e in row['events']])


def load_unified(path):
    records = [from_dict(row) for row in read_rows(path)]
    if len({r.id for r in records}) != len(records):
        raise ValueError('Duplicate ID')
    return records
