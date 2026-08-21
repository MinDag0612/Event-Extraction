from dataclasses import dataclass, field

from src.unified_format.event_schema import EventSchema

@dataclass
class Schema:
    event_schemas: list[EventSchema]
    
    def add_schema(self, new_schema: EventSchema):
        pass
    # Implement logic add schema here