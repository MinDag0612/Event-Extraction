import json

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from constrain_decode.json_prefix_parser import (
    EVENT_EXTRACTION_DATA,
    IncompleteJsonPrefix,
    InvalidJsonPrefix,
    JsonPrefixParser,
)
from unified_format.argument import Argument
from unified_format.event import Event
from unified_format.event_extraction_data import EventExtractionData
from unified_format.trigger import Trigger


class EventConstraintProcessor(LogitsProcessor):

    def __init__(
        self,
        tokenizer,
        schema: dict,
        source_data: EventExtractionData,
        prompt_length: int,
    ):
        self.tokenizer = tokenizer
        self.schema = schema
        self.source_data = source_data
        self.prompt_length = prompt_length

        self.special_token_ids = set(tokenizer.all_special_ids)
        self.eos_token_id = tokenizer.eos_token_id

    # ======================================================
    # STRUCTURE VALIDATION
    # ======================================================

    def _validate_structure(self, data):
        """
        Validate a complete JSON object against
        EventExtractionData structure.
        """

        if not isinstance(data, dict):
            return False

        if set(data) != {"id", "text", "events", "tokens"}:
            return False

        if not isinstance(data["id"], str):
            return False

        if not isinstance(data["text"], str):
            return False

        if not isinstance(data["events"], list):
            return False

        if not isinstance(data["tokens"], list):
            return False

        if not all(isinstance(token, str) for token in data["tokens"]):
            return False

        for event in data["events"]:

            if not isinstance(event, dict):
                return False

            if set(event) != {"event_type", "trigger", "arguments"}:
                return False

            if not isinstance(event["event_type"], str):
                return False

            if not isinstance(event["trigger"], list):
                return False

            if not isinstance(event["arguments"], list):
                return False

            for trigger in event["trigger"]:

                if not isinstance(trigger, dict):
                    return False

                if set(trigger) != {"text", "span"}:
                    return False

                if not isinstance(trigger["text"], str):
                    return False

                if not self._validate_span(trigger["span"]):
                    return False

            for argument in event["arguments"]:

                if not isinstance(argument, dict):
                    return False

                if set(argument) != {"role", "mentions"}:
                    return False

                if not isinstance(argument["role"], str):
                    return False

                if not isinstance(argument["mentions"], list):
                    return False

                for mention in argument["mentions"]:

                    if not isinstance(mention, dict):
                        return False

                    if set(mention) != {"text", "span"}:
                        return False

                    if not isinstance(mention["text"], str):
                        return False

                    if not self._validate_span(mention["span"]):
                        return False

        return True

    def _validate_span(self, span):
        """
        Validate span structure.

        Expected:
            [start, end]

        Both values must be integers.
        """

        if not isinstance(span, list):
            return False

        if len(span) != 2:
            return False

        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in span
        ):
            return False

        return True

    # ======================================================
    # PREFIX VALIDATION
    # ======================================================

    def _is_valid_prefix(self, text: str) -> tuple[bool, bool]:
        """
        Check whether generated text is a valid prefix
        of the expected EventExtractionData JSON.

        Returns:
            is_valid: Whether the text can still become
                      a valid JSON structure.
            is_complete: Whether a complete valid JSON
                         object has been generated.
        """

        try:
            parser = JsonPrefixParser(text)
            parser.parse(EVENT_EXTRACTION_DATA)
            parser.skip_whitespace()
            if parser.position != len(text):
                return False, False
        except IncompleteJsonPrefix:
            return True, False
        except InvalidJsonPrefix:
            return False, False

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return False, False
        is_valid = self._validate_structure(data)
        return is_valid, is_valid

    def parse_output(self, text: str) -> EventExtractionData:
        """Parse and validate a completed generation as the unified dataclass."""
        is_valid, is_complete = self._is_valid_prefix(text)
        if not is_valid or not is_complete:
            raise ValueError("Generated output is not a complete EventExtractionData JSON object.")

        data = json.loads(text)
        return EventExtractionData(
            id=data["id"],
            text=data["text"],
            events=[
                Event(
                    event_type=event["event_type"],
                    trigger=[
                        Trigger(text=trigger["text"], span=tuple(trigger["span"]))
                        for trigger in event["trigger"]
                    ],
                    arguments=[
                        Argument(role=argument["role"], mentions=argument["mentions"])
                        for argument in event["arguments"]
                    ],
                )
                for event in data["events"]
            ],
            tokens=data["tokens"],
        )

    # ======================================================
    # GENERATION STATE
    # ======================================================

    def _get_generated_tokens(
        self,
        input_ids: torch.LongTensor,
    ):
        """
        Extract generated tokens, excluding prompt tokens.
        """

        return input_ids[self.prompt_length:]

    # ======================================================
    # TOKEN CONSTRAINT
    # ======================================================

    def _get_allowed_tokens(self, generated_ids):

        generated_ids = generated_ids.tolist()

        current_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        is_valid, is_complete = self._is_valid_prefix(
            current_text
        )

        if not is_valid:
            raise ValueError(
                f"Invalid generated prefix: {current_text}"
            )

        # JSON is complete: only EOS is allowed.
        if is_complete:
            if self.eos_token_id is not None:
                return [self.eos_token_id]

            return []

        allowed_tokens = []

        vocab_size = len(self.tokenizer)

        for token_id in range(vocab_size):

            # Skip special tokens.
            if token_id in self.special_token_ids:
                continue

            candidate_ids = generated_ids + [token_id]

            candidate_text = self.tokenizer.decode(
                candidate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            valid, _ = self._is_valid_prefix(
                candidate_text
            )

            if valid:
                allowed_tokens.append(token_id)
        
        if not allowed_tokens or len(allowed_tokens) == 0:
            print(token_id)

        return allowed_tokens

    # ======================================================
    # LOGITS MASKING
    # ======================================================

    def _apply_mask(
        self,
        scores: torch.FloatTensor,
        allowed_tokens: list[int],
    ):

        if not allowed_tokens:
            raise RuntimeError(
                "No valid tokens available."
            )

        masked_scores = torch.full_like(
            scores,
            float("-inf"),
        )

        masked_scores[allowed_tokens] = scores[
            allowed_tokens
        ]

        return masked_scores

    # ======================================================
    # HUGGING FACE INTERFACE
    # ======================================================

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:

        batch_scores = []

        for batch_idx in range(input_ids.shape[0]):

            # 1. Extract generated tokens
            generated_ids = self._get_generated_tokens(
                input_ids[batch_idx]
            )

            # 2. Determine allowed tokens
            allowed_tokens = self._get_allowed_tokens(
                generated_ids
            )

            # 3. Apply logits masking
            masked_scores = self._apply_mask(
                scores[batch_idx],
                allowed_tokens,
            )

            batch_scores.append(masked_scores)

        return torch.stack(batch_scores)


def make_event_constraint_logits_processor(
    tokenizer,
    schema: dict,
    source_data: EventExtractionData,
    prompt_length: int,
) -> LogitsProcessorList:
    return LogitsProcessorList(
        [
            EventConstraintProcessor(
                tokenizer=tokenizer,
                schema=schema,
                source_data=source_data,
                prompt_length=prompt_length,
            )
        ]
    )