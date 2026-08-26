from typing import Any, Dict
from pprint import pprint

from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.argument import Argument
from src.unified_format.trigger import Trigger
from src.unified_format.event_schema import EventSchema
from collections import defaultdict

class GENEVAAdapter(AdapterInterface):
    def __init__(self):
        self.events_schema = []
    
    def adapt(self, data: Any) -> EventExtractionData:
        data_id = data["doc_id"]
        data_text = data["sentence"]
        
        entities_mapping = {
            entity["id"]: {"text": entity["text"],
                           "span": [entity["start"], entity["end"]]}
            for entity in data["entity_mentions"]
        }
        data_events = self.get_events(data["event_mentions"], entities_mapping)
        
        return EventExtractionData(
            id=data_id,
            text=data_text,
            events=data_events
        )
        
    # SUB-FUNCTIONS
    def get_events(self, events: list, entities_map: list):
        event_list = []
        
        for event in events:
            event_type = event["event_type"]
            
            triggers = [
                Trigger(text=event["trigger"]["text"],
                        span=[event["trigger"]["start"], event["trigger"]["end"]])]
            
            arguments = [
                Argument(role=argu["role"], mentions=entities_map[argu["entity_id"]])
                for argu in event["arguments"]]
            
            event_list.append(Event(
                event_type=event_type,
                trigger=triggers,
                arguments=arguments
            ))
            
        return event_list
    
    
    def get_schema(self, data: Any) -> EventSchema:
        mapping_events = defaultdict(list)
        event_schema_list = []
        
        for row in data:
            event_name = row.get("# Event Name", "").strip()
            if event_name:
                role_list = []
                current_event = event_name
            else:
                is_argument_role = row.get("Is Argument Role?", "").strip()
                if is_argument_role == "1":
                    role = row.get("Frame Element Name", "").strip()

                    if role:
                        mapping_events[current_event].append(role)
                        
        for event_type, role in mapping_events.items():
            event_schema_list.append(EventSchema(
                event_type=event_type,
                argument_roles=role
            ))
        return event_schema_list