from __future__ import annotations

from typing import Any, Dict, Iterable

from src.adapter.base_adapter import AdapterInterface
from src.unified_format.argument import Argument
from src.unified_format.event import Event
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger


class BKEEAdapter(AdapterInterface):

    def adapt(self, data: Any) -> EventExtractionData:
        if not isinstance(data, dict):
            raise TypeError("BKEEAdapter expects a dictionary sample")

        self._entity_mentions_by_id = {
            entity.get("id"): entity
            for entity in data.get("entity_mentions", [])
            if isinstance(entity, dict) and entity.get("id") is not None
        }

        return EventExtractionData(
            id=self._as_str(
                self._pick(
                    data,
                    [
                        "id",
                        "doc_id",
                        "document_id",
                        "sentence_id",
                        "uid",
                    ],
                )
            ),
            text=self._as_str(self._pick(data, ["text", "sentence", "content", "document"])),
            events=self._get_events(data),
        )

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": ["string", "number"]},
                "text": {"type": "string"},
                "events": {"type": "array"},
            },
            "required": ["text"],
        }

    # Internal mapping 
    def _get_events(self, data: dict[str, Any]) -> list[Event]:
        raw_events = self._pick(
            data,
            [
                "events",
                "event_mentions",
                "golden-event-mentions",
                "golden_event_mentions",
                "labels",
            ],
            default=[],
        )

        if not isinstance(raw_events, list):
            return []

        events: list[Event] = []
        for raw_event in raw_events:
            if not isinstance(raw_event, dict):
                continue

            event_type = self._as_str(
                self._pick(raw_event, ["event_type", "type", "subtype", "label"])
            )
            trigger = self._parse_trigger(raw_event)
            arguments = self._parse_arguments(raw_event)

            events.append(
                Event(
                    event_type=event_type,
                    trigger=trigger,
                    arguments=arguments,
                )
            )

        return events

    def _parse_trigger(self, raw_event: dict[str, Any]) -> Trigger:
        raw_trigger = self._pick(raw_event, ["trigger", "event_trigger"]) 

        if isinstance(raw_trigger, dict):
            trigger_text = self._as_str(
                self._pick(raw_trigger, ["text", "trigger_word", "word", "mention", "content"])
            )
            trigger_span = self._parse_span(raw_trigger)
        else:
            trigger_text = self._as_str(
                self._pick(raw_event, ["trigger_word", "trigger_text", "trigger"])
            )
            trigger_span = self._parse_span(raw_event)

        return Trigger(text=trigger_text, span=trigger_span)

    def _parse_arguments(self, raw_event: dict[str, Any]) -> list[Argument]:
        raw_arguments = self._pick(raw_event, ["arguments", "argument", "args"], default=[])

        # Case 1: {"role": [mentions...]}
        if isinstance(raw_arguments, dict):
            output: list[Argument] = []
            for role, mentions in raw_arguments.items():
                output.append(
                    Argument(
                        role=self._as_str(role),
                        mentions=self._normalize_mentions(mentions),
                    )
                )
            return output

        # Case 2: [{"role": "X", ...}, ...]
        if isinstance(raw_arguments, list):
            grouped: dict[str, list[dict[str, Any]]] = {}
            for item in raw_arguments:
                if not isinstance(item, dict):
                    continue

                role = self._as_str(
                    self._pick(item, ["role", "argument_role", "label", "type"])
                )
                grouped.setdefault(role, []).extend(self._normalize_mentions(item))

            return [Argument(role=role, mentions=mentions) for role, mentions in grouped.items()]

        return []

    def _normalize_mentions(self, raw: Any) -> list[dict[str, Any]]:
        items: Iterable[Any]
        if isinstance(raw, list):
            items = raw
        else:
            items = [raw]

        mentions: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                # Some formats encode an argument object that also carries role.
                entity = self._entity_mentions_by_id.get(item.get("entity_id"))
                target = (
                    entity
                    if isinstance(entity, dict)
                    else item.get("mention")
                    if isinstance(item.get("mention"), dict)
                    else item
                )
                text = self._as_str(
                    self._pick(target, ["text", "content", "mention", "value", "argument"])
                )
                span = self._parse_span(target)
            else:
                text = self._as_str(item)
                span = (0, 0)

            mentions.append({"text": text, "span": span})

        return mentions

    # ---------- Small helpers ----------
    def _pick(self, source: dict[str, Any], keys: list[str], default: Any = None) -> Any:
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
        return default

    def _parse_span(self, source: dict[str, Any]) -> tuple[int, int]:
        # BKEE exposes token offsets (`start`/`end`) and character offsets
        # (`start_char`/`end_char`). Unified spans refer to the full text, so
        # character offsets are the unambiguous representation here.
        start_char = source.get("start_char")
        end_char = source.get("end_char")
        if start_char is not None and end_char is not None:
            return (int(start_char), int(end_char))

        span = self._pick(source, ["span", "offset", "position", "trigger_span"]) 
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            return (int(span[0]), int(span[1]))

        start = self._pick(source, ["start", "start_offset", "begin"])
        end = self._pick(source, ["end", "end_offset", "stop"])
        if start is not None and end is not None:
            return (int(start), int(end))

        return (0, 0)

    def _as_str(self, value: Any) -> str:
        return "" if value is None else str(value)
