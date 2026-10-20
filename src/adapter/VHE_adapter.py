from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger
from src.unified_format.argument import Argument
from src.unified_format.event import Event
from src.unified_format.event_schema import EventSchema

from typing import Any
from src.unified_format.schema import Schema
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

    def get_schema(self, data: Any) -> Schema:
        if isinstance(data, dict) and "event_schemas" in data:
            return super().get_schema(data)
        rows = [data] if isinstance(data, dict) else data
        events = []
        for row in rows:
            if not isinstance(row, dict) or not {"event-type", "role-list"} <= row.keys():
                raise ValueError("Expected VHE ontology entries")
            events.append(EventSchema(row["event-type"], [r["role"] for r in row["role-list"]]))
        return Schema(events)

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
