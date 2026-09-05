from abc import ABC, abstractmethod
from typing import Any, Dict

from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.event_schema import EventSchema


class AdapterInterface(ABC):
    @abstractmethod
    def adapt(self, data: Any) -> EventExtractionData:
        """Chuyển đổi dữ liệu đầu vào theo định dạng cần thiết."""
        pass

    @abstractmethod
    def get_schema(self) -> EventSchema:
        """Trả về schema tùy thuộc vào cách tổ chức dataset. Có thể phụ thuộc adapter nếu dataset khong có schema cố định."""
        pass



