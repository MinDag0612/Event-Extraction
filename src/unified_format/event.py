from dataclasses import dataclass, field

from src.unified_format.argument import Argument
from src.unified_format.trigger import Trigger

@dataclass
class Event:
    event_type: str
    trigger: Trigger
    arguments: list[Argument] = field(default_factory=list)
