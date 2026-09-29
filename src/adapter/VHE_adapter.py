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

    # SUB-FUNCTIONS
    def get_events(self, events: list, text: str, offsets: list, event_id: str) -> list:
        event_list = []
        count_invalid_spans = 0
        invalid_spans = []

        for event in events:
            event_type = event.get("type", "")
            try: 
                trigger = [
                    Trigger(text=event["trigger_word"], span=char_to_token_span(text, offsets, event["offset"]))
                ]
                arguments = [
                    Argument(role=arg["role"], mentions=[
                        {"text": arg["mention"], "span": char_to_token_span(text, offsets, arg["offset"])}
                    ])
                    for arg in event.get("arguments", [])
                    ]

                event = Event(
                    event_type=event_type,
                    trigger=trigger,
                    arguments=arguments
                )
                event_list.append(event)
            
            except ValueError as e:
                count_invalid_spans += 1
                invalid_spans.append(str(e))
                continue  # Skip events with invalid spans
        
        if count_invalid_spans > 0:
            print(f"Event {event_id} has {count_invalid_spans} invalid spans. This event has been skipped.")
            print(f"Invalid spans for event {event_id}: {invalid_spans}")

        return event_list
