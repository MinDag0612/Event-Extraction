from dataclasses import dataclass

@dataclass
class Argument:
    role: str
    mentions: list[dict]