from abc import ABC, abstractmethod
from typing import Any, Dict

from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.schema import Schema


class AdapterInterface(ABC):
    @abstractmethod
    def adapt(self, data: Any) -> EventExtractionData:
        """Chuyển đổi dữ liệu đầu vào theo định dạng cần thiết."""
        pass

    def get_schema(self, data: Any) -> "Schema":
        """Load an explicit ontology, never infer it from held-out labels.

        All adapters accept {"event_schemas": [...]} and return Schema.
        Dataset adapters may additionally accept their native ontology format.
        """
        return Schema.from_dict(data)
