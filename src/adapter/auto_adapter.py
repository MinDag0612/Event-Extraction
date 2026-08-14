from __future__ import annotations

from typing import Any

from src.adapter.BKEE_adapter import BKEEAdapter
from src.adapter.MAVEN_adapter import MAVENAdapter
from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData


class AutoAdapter(AdapterInterface):

    def __init__(self) -> None:
        self.maven_adapter = MAVENAdapter()
        self.bkee_adapter = BKEEAdapter()

    def adapt(self, data: Any) -> EventExtractionData:
        if isinstance(data, EventExtractionData):
            return data
        if not isinstance(data, dict):
            raise TypeError("AutoAdapter expects dict or EventExtractionData")

        if self._looks_maven(data):
            return self.maven_adapter.adapt(data)
        return self.bkee_adapter.adapt(data)

    def get_schema(self):
        return {
            "oneOf": [
                {"title": "maven_arg"},
                {"title": "bkee_like"},
            ]
        }

    def _looks_maven(self, data: dict[str, Any]) -> bool:
        return "document" in data and "entities" in data and "events" in data
