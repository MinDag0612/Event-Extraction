from dataclasses import dataclass, field

@dataclass
class EventSchema:
    event_type: str
    argument_roles: list[str]
    