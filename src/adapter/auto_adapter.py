from __future__ import annotations

from typing import Any

from src.adapter.BKEE_adapter import BKEEAdapter
from src.adapter.MAVEN_adapter import MAVENAdapter
from src.adapter.RAMS_adapter import RAMSAdapter
from src.adapter.PHEE_adapter import PHEEAdapter
from src.adapter.VHE_adapter import VHEAdapter
from src.adapter.GENEVA_adepter import GENEVAAdapter
from src.adapter.base_adapter import AdapterInterface
from src.unified_format.event_extraction_data import EventExtractionData


class AutoAdapter(AdapterInterface):

    def __init__(self) -> None:
        self.phee_adapter = PHEEAdapter()
        self.vhe_adapter = VHEAdapter()
        self.geneva_adapter = GENEVAAdapter()
        self.maven_adapter = MAVENAdapter()
        self.bkee_adapter = BKEEAdapter()
        self.rams_adapter = RAMSAdapter()

    def adapt(self, data: Any) -> EventExtractionData:
        if isinstance(data, EventExtractionData):
            return data
        if not isinstance(data, dict):
            raise TypeError("AutoAdapter expects dict or EventExtractionData")

        if "context" in data and "annotations" in data:
            return self.phee_adapter.adapt(data)
        if all(k in data for k in ("wnd_id", "sentence", "entity_mentions", "event_mentions")):
            return self.geneva_adapter.adapt(data)
        if "text" in data and "events" in data and "tokens" not in data:
            return self.vhe_adapter.adapt(data)
        if self._looks_rams(data):
            return self.rams_adapter.adapt(data)
        if self._looks_maven(data):
            return self.maven_adapter.adapt(data)
        return self.bkee_adapter.adapt(data)

    def get_schema(self):
        return {
            "oneOf": [
                {"title": "maven_arg"},
                {"title": "bkee_like"},
                {"title": "rams"},
                {"title": "phee"}, {"title": "vhe"}, {"title": "geneva"},
            ]
        }

    def _looks_maven(self, data: dict[str, Any]) -> bool:
        return "document" in data and "entities" in data and "events" in data

    def _looks_rams(self, data: dict[str, Any]) -> bool:
        return "sentences" in data and "evt_triggers" in data and "doc_key" in data
