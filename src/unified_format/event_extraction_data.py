from dataclasses import asdict, dataclass, field
import json
from src.unified_format.event import Event

@dataclass
class EventExtractionData:

    id: str
    text: str
    events: list[Event] = field(default_factory=list)
    
    tokens: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            asdict(self),
            # indent=4,
            ensure_ascii=False
        )
