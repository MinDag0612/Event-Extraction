"""Generation quality indicators, independent of gold labels."""
def generation_summary(predictions, diagnostics):
    count = len(predictions)
    empty = sum(not row['events'] for row in predictions)
    events = arguments = repeated_events = repeated_arguments = 0
    for row in predictions:
        seen_events = set()
        for event in row['events']:
            events += 1
            key = (event['event_type'], tuple(tuple(t['span']) for t in event['trigger']))
            repeated_events += key in seen_events
            seen_events.add(key)
            seen_args = set()
            for arg in event['arguments']:
                for mention in arg['mentions']:
                    arguments += 1
                    key = (arg['role'], tuple(mention['span']))
                    repeated_arguments += key in seen_args
                    seen_args.add(key)
    return {'samples': count, 'empty_predictions': empty,
            'empty_rate': empty / count if count else 0., 'events': events,
            'argument_mentions': arguments, 'repeated_events': repeated_events,
            'repeated_argument_mentions': repeated_arguments,
            'budget_constrained': sum(d.get('budget_constrained', False) for d in diagnostics),
            'invalid_predictions': sum(d['status'] != 'ok' for d in diagnostics)}
