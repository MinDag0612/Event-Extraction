from typing import Any, Dict
from pprint import pprint


from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.argument import Argument
from src.unified_format.trigger import Trigger

class GENEVAAdapter(AdapterInterface):
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
    
    
    def get_schema(self):
        pass
            
            