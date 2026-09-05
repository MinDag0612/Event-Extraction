from __future__ import annotations

from typing import Any, Dict

from src.adapter.base_adapter import AdapterInterface
from src.unified_format.argument import Argument
from src.unified_format.event import Event
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger


class RAMSAdapter(AdapterInterface):
   

    def adapt(self, data: Any) -> EventExtractionData:
        if not isinstance(data, dict):
            raise TypeError("RAMSAdapter expects a dictionary sample")

        tokens = self._flatten_sentences(data.get("sentences", []))
        events = self._get_events(data, tokens)
        return EventExtractionData(
            id=str(data.get("doc_key", "")),
            text=" ".join(tokens),
            tokens=tokens,
            events=events,
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "doc_key": {"type": "string"},
                "sentences": {"type": "array"},
                "evt_triggers": {"type": "array"},
                "gold_evt_links": {"type": "array"},
            },
            "required": ["doc_key", "sentences", "evt_triggers"],
        }

    def _get_events(self, data: dict[str, Any], tokens: list[str]) -> list[Event]:
        arguments_by_trigger: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = {}
        for link in data.get("gold_evt_links", []):
            if not isinstance(link, list) or len(link) < 3:
                continue
            trigger_span = self._span(link[0])
            argument_span = self._span(link[1])
            role = str(link[2])
            mention = {
                "text": self._span_text(tokens, argument_span),
                "span": argument_span,
            }
            arguments_by_trigger.setdefault(trigger_span, {}).setdefault(role, []).append(mention)

        events: list[Event] = []
        for raw_trigger in data.get("evt_triggers", []):
            if not isinstance(raw_trigger, list) or len(raw_trigger) < 3:
                continue
            trigger_span = self._span(raw_trigger[:2])
            labels = raw_trigger[2]
            event_type = ""
            if isinstance(labels, list) and labels:
                first_label = labels[0]
                if isinstance(first_label, list) and first_label:
                    event_type = str(first_label[0])
                elif isinstance(first_label, str):
                    event_type = first_label

            grouped = arguments_by_trigger.get(trigger_span, {})
            events.append(
                Event(
                    event_type=event_type,
                    trigger=[
                        Trigger(
                            text=self._span_text(tokens, trigger_span),
                            span=trigger_span,
                        )
                    ],
                    arguments=[
                        Argument(role=role, mentions=mentions)
                        for role, mentions in grouped.items()
                    ],
                )
            )
        return events

    @staticmethod
    def _flatten_sentences(sentences: Any) -> list[str]:
        if not isinstance(sentences, list):
            return []
        return [str(token) for sentence in sentences if isinstance(sentence, list) for token in sentence]

    @staticmethod
    def _span(value: Any) -> tuple[int, int]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return (int(value[0]), int(value[1]) + 1)
        return (0, 0)

    @staticmethod
    def _span_text(tokens: list[str], span: tuple[int, int]) -> str:
        start, end = span
        if start < 0 or end < start or start >= len(tokens):
            return ""
        return " ".join(tokens[start:end])
