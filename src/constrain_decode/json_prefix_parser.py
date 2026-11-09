import json


class IncompleteJsonPrefix(Exception):
    pass


class InvalidJsonPrefix(Exception):
    pass


class JsonPrefixParser:
    def __init__(self, text: str):
        self.text = text
        self.position = 0

    def parse(self, expected):
        kind = expected[0]

        if kind == "string":
            self._parse_string()
        elif kind == "integer":
            self._parse_integer()
        elif kind == "array":
            self._parse_array(expected[1], expected[2], expected[3])
        elif kind == "object":
            self._parse_object(expected[1])
        else:
            raise ValueError(f"Unknown JSON structure type: {kind}")

    def skip_whitespace(self):
        while (
            self.position < len(self.text)
            and self.text[self.position] in " \t\r\n"
        ):
            self.position += 1

    def expect(self, char: str):
        self.skip_whitespace()
        if self.position == len(self.text):
            raise IncompleteJsonPrefix
        if self.text[self.position] != char:
            raise InvalidJsonPrefix
        self.position += 1

    def _parse_string(self, expected: str | None = None):
        self.expect('"')
        start = self.position - 1
        decoded_characters = []

        while self.position < len(self.text):
            char = self.text[self.position]

            if char == '"':
                self.position += 1
                try:
                    value = json.loads(self.text[start:self.position])
                except json.JSONDecodeError as error:
                    raise InvalidJsonPrefix from error
                if expected is not None and value != expected:
                    raise InvalidJsonPrefix
                return value

            if ord(char) < 0x20:
                raise InvalidJsonPrefix

            if char == "\\":
                self.position += 1
                if self.position == len(self.text):
                    raise IncompleteJsonPrefix

                escape = self.text[self.position]
                if escape == "u":
                    hex_start = self.position + 1
                    hex_end = hex_start + 4
                    if hex_end > len(self.text):
                        if all(
                            digit in "0123456789abcdefABCDEF"
                            for digit in self.text[hex_start:]
                        ):
                            raise IncompleteJsonPrefix
                        raise InvalidJsonPrefix
                    if not all(
                        digit in "0123456789abcdefABCDEF"
                        for digit in self.text[hex_start:hex_end]
                    ):
                        raise InvalidJsonPrefix
                    decoded_characters.append(
                        chr(int(self.text[hex_start:hex_end], 16))
                    )
                    self.position = hex_end
                elif escape in '"\\/bfnrt':
                    decoded_characters.append(
                        {
                            '"': '"',
                            "\\": "\\",
                            "/": "/",
                            "b": "\b",
                            "f": "\f",
                            "n": "\n",
                            "r": "\r",
                            "t": "\t",
                        }[escape]
                    )
                    self.position += 1
                else:
                    raise InvalidJsonPrefix
            else:
                decoded_characters.append(char)
                self.position += 1

            if (
                expected is not None
                and not expected.startswith("".join(decoded_characters))
            ):
                raise InvalidJsonPrefix

        raise IncompleteJsonPrefix

    def _parse_integer(self):
        self.skip_whitespace()
        if self.position == len(self.text):
            raise IncompleteJsonPrefix

        if self.text[self.position] == "-":
            self.position += 1
            if self.position == len(self.text):
                raise IncompleteJsonPrefix

        if self.text[self.position] == "0":
            self.position += 1
            if (
                self.position < len(self.text)
                and self.text[self.position] in "0123456789"
            ):
                raise InvalidJsonPrefix
            return

        if self.text[self.position] not in "123456789":
            raise InvalidJsonPrefix

        while (
            self.position < len(self.text)
            and self.text[self.position] in "0123456789"
        ):
            self.position += 1

    def _parse_array(self, item, minimum: int, maximum: int | None):
        self.expect("[")
        self.skip_whitespace()

        if self.position == len(self.text):
            raise IncompleteJsonPrefix
        if self.text[self.position] == "]":
            if minimum:
                raise InvalidJsonPrefix
            self.position += 1
            return

        count = 0
        while True:
            if maximum is not None and count >= maximum:
                raise InvalidJsonPrefix
            self.parse(item)
            count += 1
            self.skip_whitespace()

            if self.position == len(self.text):
                raise IncompleteJsonPrefix
            if self.text[self.position] == "]":
                if count < minimum:
                    raise InvalidJsonPrefix
                self.position += 1
                return
            if self.text[self.position] != ",":
                raise InvalidJsonPrefix
            if maximum is not None and count >= maximum:
                raise InvalidJsonPrefix
            self.position += 1

    def _parse_object(self, fields):
        self.expect("{")

        for index, (key, value) in enumerate(fields):
            actual_key = self._parse_string(expected=key)
            if actual_key != key:
                raise InvalidJsonPrefix
            self.expect(":")
            self.parse(value)
            self.expect("," if index < len(fields) - 1 else "}")


STRING = ("string",)
INTEGER = ("integer",)
TRIGGER = (
    "object",
    (
        ("text", STRING),
        ("span", ("array", INTEGER, 2, 2)),
    ),
)
ARGUMENT = (
    "object",
    (
        ("role", STRING),
        ("mentions", ("array", TRIGGER, 0, None)),
    ),
)
EVENT = (
    "object",
    (
        ("event_type", STRING),
        ("trigger", ("array", TRIGGER, 0, None)),
        ("arguments", ("array", ARGUMENT, 0, None)),
    ),
)
EVENT_EXTRACTION_DATA = (
    "object",
    (
        ("id", STRING),
        ("text", STRING),
        ("events", ("array", EVENT, 0, None)),
        ("tokens", ("array", STRING, 0, None)),
    ),
)
