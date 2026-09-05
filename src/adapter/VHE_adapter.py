from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger
from src.unified_format.argument import Argument
from src.unified_format.event import Event

from typing import Any


class VHEAdapter(AdapterInterface):
    def __init__(self):
        pass
    
    def adapt(self, data: Any) -> EventExtractionData:
        event_id = data.get("id", "")
        text = data.get("text", "")
        events = self.get_events(data.get("events", []))
        
        return EventExtractionData(
            id=event_id,
            text=text,
            events=events
        )
        
    # SUB-FUNCTIONS
    def get_events(self, events: list) -> list:
        event_list = []
        
        for event in events:
            event_type = event.get("type", "")
            trigger = [
                Trigger(text=event["trigger_word"], span=tuple(event["offset"]))
            ]
            arguments = [
                Argument(role=arg["role"], mentions=[
                    {"text": arg["mention"], "span": tuple(arg["offset"])}
                ])
                for arg in event.get("arguments", [])
                ]
            
            event = Event(
                event_type=event_type,
                trigger=trigger,
                arguments=arguments
            )
            event_list.append(event)
        
        return event_list
    
    def get_schema(self, data: Any) -> dict:
        pass