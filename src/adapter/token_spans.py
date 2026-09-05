import re


def tokenize_with_offsets(text: str):
    matches = list(re.finditer(r"[^\W\d_]+|\d+|[^\w\s]|_", text))
    return [m.group() for m in matches], [m.span() for m in matches]


def char_to_token_span(text, offsets, span):
    start, end = span
    if not 0 <= start < end <= len(text):
        raise ValueError(f"Invalid character span {span}")
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    starts = {s: i for i, (s, e) in enumerate(offsets)}
    ends = {e: i + 1 for i, (s, e) in enumerate(offsets)}
    if start not in starts or end not in ends:
        raise ValueError(f"Character span {span} cuts a token: {text[start:end]!r}")
    return starts[start], ends[end]
