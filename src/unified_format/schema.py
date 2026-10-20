"""Event/role ontology; separate from the JSON format of a labeled sample."""
from dataclasses import asdict, dataclass
import json
from src.unified_format.event_schema import EventSchema


@dataclass
class Schema:
    event_schemas: list[EventSchema]

    def __post_init__(self):
        mapping = {}
        for event in self.event_schemas:
            if not isinstance(event, EventSchema):
                raise ValueError("Expected EventSchema entries")
            if not isinstance(event.event_type, str) or not event.event_type.strip():
                raise ValueError("Expected nonempty event type")
            if not isinstance(event.argument_roles, list):
                raise ValueError("Expected a list of roles")
            roles = mapping.setdefault(event.event_type, set())
            for role in event.argument_roles:
                if not isinstance(role, str) or not role.strip():
                    raise ValueError("Expected nonempty role")
                roles.add(role)
        self.event_schemas = [EventSchema(name, sorted(roles))
                              for name, roles in sorted(mapping.items())]

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != {"event_schemas"}:
            raise ValueError("Expected ontology with event_schemas; not labeled samples")
        if not isinstance(data["event_schemas"], list):
            raise ValueError("event_schemas must be a list")
        events = []
        for item in data["event_schemas"]:
            if not isinstance(item, dict) or set(item) != {"event_type", "argument_roles"}:
                raise ValueError("Expected event_type and argument_roles")
            events.append(EventSchema(**item))
        return cls(events)

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)
