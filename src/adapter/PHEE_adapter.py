import re
from dataclasses import dataclass
from typing import Any, Iterator

from src.adapter.base_adapter import AdapterInterface
from src.unified_format.argument import Argument
from src.unified_format.event import Event
from src.unified_format.event_extraction_data import EventExtractionData
from src.unified_format.trigger import Trigger

Mention = dict[str, Any]
TokenOffsets = list[tuple[int, int]]

ANNOTATION_FIELDS = {
    "Trigger", "event_id", "event_type", "text", "start", "entity_id", "value",
}


@dataclass
class ParsedEvent:
    event_type: str
    triggers: list[Mention]
    arguments: list[tuple[str, list[Mention]]]

    def boundaries(self) -> set[int]:
        mentions = self.triggers + [
            mention for _, values in self.arguments for mention in values
        ]
        return {edge for mention in mentions for edge in mention["char_span"]}


class PHEEAdapter(AdapterInterface):
    def get_schema(self) -> dict[str, Any]:
        return {"title": "phee", "required": ["id", "context", "annotations"]}

    def adapt(self, data: dict[str, Any]) -> EventExtractionData:
        text = data["context"]
        record_id = str(data["id"])
        parsed = [
            self._parse_event(event, text, record_id)
            for annotation in data["annotations"]
            for event in annotation["events"]
        ]
        boundaries = {edge for event in parsed for edge in event.boundaries()}
        tokens, offsets = self._tokenize_at_boundaries(text, boundaries)
        return EventExtractionData(
            id=record_id,
            text=text,
            tokens=tokens,
            events=[self._convert_event(event, offsets) for event in parsed],
        )

    def _parse_event(self, event: dict, text: str, record_id: str) -> ParsedEvent:
        return ParsedEvent(
            event_type=event["event_type"],
            triggers=self._parse_mentions(event["Trigger"], text, record_id),
            arguments=list(self._parse_arguments(event, text, record_id)),
        )

    def _parse_arguments(
        self, node: dict, text: str, record_id: str, prefix: str = ""
    ) -> Iterator[tuple[str, list[Mention]]]:
        for key, value in node.items():
            if key in ANNOTATION_FIELDS:
                continue
            role = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if "text" in value and "start" in value:
                    yield role, self._parse_mentions(value, text, record_id)
                yield from self._parse_arguments(value, text, record_id, role)
            elif isinstance(value, list):
                for nested in value:
                    if isinstance(nested, dict):
                        yield from self._parse_arguments(nested, text, record_id, role)

    def _convert_event(self, event: ParsedEvent, offsets: TokenOffsets) -> Event:
        triggers = [
            Trigger(text=mention["text"], span=self._to_token_mention(mention, offsets)["span"])
            for mention in event.triggers
        ]
        grouped: dict[str, list[Mention]] = {}
        for role, mentions in event.arguments:
            target = grouped.setdefault(role, [])
            # Each nested occurrence gets distinct discontinuous-mention groups.
            group_offset = max((m["mention_group"] for m in target), default=-1) + 1
            for mention in mentions:
                converted = self._to_token_mention(mention, offsets)
                converted["mention_group"] += group_offset
                target.append(converted)
        return Event(
            event_type=event.event_type,
            trigger=triggers,
            arguments=[Argument(role=role, mentions=values) for role, values in grouped.items()],
        )

    def _parse_mentions(self, node: dict, text: str, record_id: str) -> list[Mention]:
        if len(node["text"]) != len(node["start"]):
            raise ValueError("PHEE text/start groups differ")

        mentions = []
        for group, (parts, starts) in enumerate(zip(node["text"], node["start"])):
            if len(parts) != len(starts):
                raise ValueError("PHEE text/start fragments differ")
            for fragment, (part, start) in enumerate(zip(parts, starts)):
                end = start + len(part)
                if not part or not 0 <= start < end <= len(text) or text[start:end] != part:
                    raise ValueError(
                        f"PHEE {record_id}: invalid annotated text at {start}: {part!r}"
                    )
                mentions.append({
                    "text": part,
                    "char_span": (start, end),
                    "mention_group": group,
                    "fragment_index": fragment,
                })
        return mentions


    def _tokenize_at_boundaries(self, text: str, boundaries: set[int]) -> tuple[list[str], TokenOffsets]:
        """Split words at annotation boundaries, including boundaries inside words."""
        edges = sorted({0, len(text), *boundaries})
        offsets = []
        for start, end in zip(edges, edges[1:]):
            for match in re.finditer(r"\w+|[^\w\s]", text[start:end]):
                offsets.append((start + match.start(), start + match.end()))
        return [text[start:end] for start, end in offsets], offsets


    def _to_token_mention(self, mention: Mention, offsets: TokenOffsets) -> Mention:
        start, end = mention["char_span"]
        indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if not indices:
            raise ValueError("PHEE whitespace-only mention")
        return dict(mention, span=(indices[0], indices[-1] + 1))
