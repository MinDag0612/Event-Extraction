from dataclasses import dataclass


@dataclass
class Trigger:
    text: str
    span: tuple[int, int]
