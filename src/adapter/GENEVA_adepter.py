from typing import Any, Dict
from pprint import pprint

from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.argument import Argument
from src.unified_format.trigger import Trigger
from src.unified_format.event_schema import EventSchema
from src.unified_format.schema import Schema

class GENEVAAdapter(AdapterInterface):
    def __init__(self):
        self.events_schema = []
    
    def adapt(self, data: Any) -> EventExtractionData:
        data_id = str(data["_unified_id"] if "_unified_id" in data else data["wnd_id"])
        data_text = data["sentence"]
        
        entities_mapping = {
            entity["id"]: {"text": entity["text"],
                           "span": [entity["start"], entity["end"]]}
            for entity in data["entity_mentions"]
        }
        data_events = self.get_events(data["event_mentions"], entities_mapping)
        
        event = EventExtractionData(
            id=data_id,
            text=data_text,
            tokens=list(data["tokens"]),
            events=data_events
        )
        
        
        return event
        
    # SUB-FUNCTIONS
    def get_events(self, events: list, entities_map: list):
        event_list = []
        
        for event in events:
            event_type = event["event_type"]
            
            triggers = [
                Trigger(text=event["trigger"]["text"],
                        span=(event["trigger"]["start"], event["trigger"]["end"]))]
            
            arguments = [
                Argument(role=argu["role"], mentions=[entities_map[argu["entity_id"]]])
                for argu in event["arguments"]]
            
            event_list.append(Event(
                event_type=event_type,
                trigger=triggers,
                arguments=arguments
            ))
            
        return event_list
    
    
    def get_schema(self, data: Any) -> Schema:
        if isinstance(data, dict):
            return super().get_schema(data)
        mapping = {}
        current_event = None
        for row in data:
            event_name = row.get("# Event Name", "").strip()
            if event_name:
                current_event = event_name
                mapping.setdefault(current_event, [])
            flag = row.get("Is Argument Role?", "").strip()
            if flag in {"1", "3"}:
                if current_event is None:
                    raise ValueError("GENEVA role occurs before event header")
                key = "Merged Argument Role (if previous column is 3)" if flag == "3" else "Frame Element Name"
                role = row.get(key, "").strip()
                if not role:
                    raise ValueError("GENEVA role is empty")
                mapping[current_event].append(role)
        return Schema([EventSchema(name, roles) for name, roles in mapping.items()])
