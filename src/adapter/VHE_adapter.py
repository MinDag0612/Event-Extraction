from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger
from src.unified_format.argument import Argument
from src.unified_format.event import Event
from src.unified_format.event_schema import EventSchema

from typing import Any
from src.adapter.token_spans import tokenize_with_offsets, char_to_token_span

class VHEAdapter(AdapterInterface):
    def __init__(self):
        pass

    def adapt(self, data: Any) -> EventExtractionData:
        event_id = data.get("id", "")
        text = data.get("text", "")
        tokens, offsets = tokenize_with_offsets(text)
        events = self.get_events(data.get("events", []), text, offsets, event_id)

        return EventExtractionData(
            id=event_id,
            text=text,
            tokens=tokens,
            events=events
        )

    def get_schema(self, data: Any) -> EventSchema:
        type = data.get("event-type", "")
        arguments_roles = [
            role['role']
            for role in data.get("role-list", [])
        ]

        return EventSchema(
            event_type=type,
            argument_roles=arguments_roles
        )

    def get_events(self, events, text, offsets, event_id):
        def mention(value, span):
            if text[span[0]:span[1]] != value:
                raise ValueError(f"VHE {event_id}: mention/character mismatch")
            return {"text": value, "span": char_to_token_span(text, offsets, span)}
        return [Event(
            event_type=e["type"],
            trigger=[Trigger(**mention(e["trigger_word"], e["offset"]))],
            arguments=[Argument(role=a["role"], mentions=[mention(a["mention"], a["offset"])])
                       for a in e.get("arguments", [])]
        ) for e in events]
