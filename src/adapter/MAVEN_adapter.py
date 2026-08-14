from typing import Any, Dict
from pprint import pprint


from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event import Event
from src.unified_format.argument import Argument
from src.unified_format.trigger import Trigger


class MAVENAdapter(AdapterInterface):

    def adapt(self, data: Any) -> EventExtractionData:
        entity_list = data.get("entities", [])
        self.entity_mentions_by_id = {
                    entity.get("id"): entity.get("mention", [])
                    for entity in entity_list
                }
        eventData = EventExtractionData(
            id=data.get("id", None),
            text=data.get("document", None),
            events=self.get_events(data)
        )
        return eventData

    # SUB-FUNCTIONS
    def get_events(self, data: Any) -> list[Event]:
        list_events = []
        for event in data.get("events", []):
            event_type = event.get("type", None)
            # [Trigger(mention.get("trigger_word", None), mention.get("offset", None)) for mention in event.get("mention", [])]
            trigger = [Trigger(mention.get("trigger_word", None), mention.get("offset", None)) for mention in event.get("mention", [])]
            arguments = self.get_argument(event["argument"])

            event_object = Event(event_type=event_type, trigger=trigger, arguments=arguments)
            list_events.append(event_object)
        return list_events
    # SUB-FUNCTIONS (END) -----------------------------------------
    
    # SUPPORT FUNCTIONS -----------------------------------------
    # Function to convert the MAVEN-Arg data into a list of Event objects
    def get_argument(self, argument_dict: dict) -> list[Argument]:
        
        arguments = []
        for role, attributes in argument_dict.items():
            mentions = []
            for attribute in attributes:
                # MAVEN-Arg stores non-entity arguments directly as text spans.
                if "content" in attribute and "offset" in attribute:
                    mentions.append({
                        "text": attribute["content"],
                        "span": attribute["offset"],
                    })
                # Entity arguments are resolved elsewhere, so preserve the id.
                elif "entity_id" in attribute:
                    entity_mentions = self.entity_mentions_by_id.get(
                        attribute["entity_id"], []
                    )
                    mentions.extend(
                        {
                            "text": mention.get("mention"),
                            "span": mention.get("offset"),
                        }
                        for mention in entity_mentions
                    )
                    
            arguments.append(Argument(role=role, mentions=mentions))
        return arguments
    
    # SUPPORT FUNCTIONS (END) -----------------------------------




    def get_schema(self) -> Dict[str, Any]: # Mock data. Implementation soon
        return {
            "type": "object",
            "properties": {
                "source": {"type": "object"},
                "adapted": {"type": "boolean"},
                "mock": {"type": "boolean"},
            },
            "required": ["source", "adapted", "mock"],
        }

