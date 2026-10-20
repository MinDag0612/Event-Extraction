"""Text2Event tree format, based on upstream revision dbe5a0ae933d639980c8e1f62c141511c12cefd6.

Span recovery is deliberately strict: repeated surface strings are ambiguous.
Gold sidecars are for auditing only, never for resolving model predictions.
"""
import re
import unicodedata
from src.data.unified import validate
from src.unified_format.schema import Schema

OPEN, CLOSE = '<extra_id_0>', '<extra_id_1>'
SPECIAL = re.compile(r'<extra_id_\d+>')


def label(value):
    if not value or any(c.isspace() for c in value) or SPECIAL.search(value):
        raise ValueError(f'Unsupported tree label: {value!r}')
    return value


def records(row):
    validate(row)
    for token in row['tokens']:
        if SPECIAL.search(token):
            raise ValueError('Token contains reserved tree marker')
    result = []
    def surface(m):
        a, b = m['span']
        return ' '.join(row['tokens'][a:b])
    for event in row['events']:
        args = [(label(a['role']), surface(m)) for a in event['arguments'] for m in a['mentions']]
        # One tree per trigger; preserves the current set-based TI/TC/AI/AC gold.
        for trigger in event['trigger']:
            result.append({'type': label(event['event_type']), 'trigger': surface(trigger), 'roles': args})
    return result


def convert(row):
    trees = []
    for event in records(row):
        args = ' '.join(f'{OPEN} {role} {text} {CLOSE}' for role, text in event['roles'])
        trees.append(f"{OPEN} {event['type']} {event['trigger']} {args} {CLOSE}")
    return {'text': ' '.join(row['tokens']), 'event': f"{OPEN} {' '.join(trees)} {CLOSE}"}


def parse(sequence):
    """Strict grammar parser returning surfaces, without consulting gold."""
    parts = re.split(r'(<extra_id_\d+>)', sequence)
    parts = [p.strip() for p in parts if p.strip()]
    pos = 0
    def take(expected=None):
        nonlocal pos
        if pos >= len(parts):
            raise ValueError('Truncated target')
        item = parts[pos]; pos += 1
        if expected is not None and item != expected:
            raise ValueError(f'Expected {expected}, got {item}')
        return item
    def leaf():
        head = take()
        if SPECIAL.search(head) or ' ' not in head:
            raise ValueError('Missing label or mention')
        name, text = head.split(' ', 1)
        return label(name), text
    take(OPEN); events = []
    while pos < len(parts) and parts[pos] == OPEN:
        take(OPEN); kind, trigger = leaf(); roles = []
        while pos < len(parts) and parts[pos] == OPEN:
            take(OPEN); roles.append(leaf()); take(CLOSE)
        take(CLOSE)
        events.append({'type': kind, 'trigger': trigger, 'roles': roles})
    take(CLOSE)
    if pos != len(parts):
        raise ValueError('Trailing output after tree')
    return events


class AmbiguousSpan(ValueError):
    pass


def to_unified(sequence, sample_id, text, tokens, ambiguity_policy="error", diagnostics=None, surface_normalizer=None):
    """No gold labels accepted. Explicit heuristic mode never guarantees exact recovery."""
    if ambiguity_policy not in {"error", "nearest-trigger"}:
        raise ValueError("Unknown ambiguity policy")
    def normalized(value):
        value = surface_normalizer(value) if surface_normalizer else unicodedata.normalize('NFKC', value)
        return ' '.join(value.split())
    used_triggers = {}
    def mention(surface, anchor=None, used=None):
        surface = normalized(surface)
        matches = []
        for i in range(len(tokens)):
            value = ''
            for j in range(i, len(tokens)):
                value += (' ' if j > i else '') + tokens[j]
                if normalized(value) == surface:
                    matches.append((i, j + 1))
                if len(normalized(value)) >= len(surface):
                    break
        if not matches:
            raise ValueError(f'Mention not in input: {surface!r}')
        if len(matches) != 1:
            if ambiguity_policy == "error":
                raise AmbiguousSpan(f'{surface!r}: {len(matches)} matching spans')
            # Trigger: leftmost. Argument: nearest trigger center, then leftmost.
            available = [span for span in matches if used is None or span not in used]
            if available:
                matches = available
            if anchor is not None:
                matches.sort(key=lambda span: (abs(sum(span) - sum(anchor)), span))
            if diagnostics is not None:
                diagnostics.append({"surface": surface, "candidates": matches, "selected": matches[0]})
        if used is not None:
            used.add(matches[0])
        a, b = matches[0]
        return {'text': ' '.join(tokens[a:b]), 'span': [a, b]}
    events = []
    for event in parse(sequence):
        used = used_triggers.setdefault((event['type'], normalized(event['trigger'])), set())
        trigger = mention(event['trigger'], used=used)
        events.append({'event_type': event['type'], 'trigger': [trigger],
                       'arguments': [{'role': role, 'mentions': [mention(value, trigger['span'])]}
                                     for role, value in event['roles']]})
    row = {'id': sample_id, 'text': text, 'tokens': tokens, 'events': events}
    validate(row)
    return row


def write_schema(schema: Schema, path):
    mapping = {label(e.event_type): [label(r) for r in e.argument_roles] for e in schema.event_schemas}
    import json
    path.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in
                             [list(mapping), sorted({r for rs in mapping.values() for r in rs}), mapping]) + '\n')
