"""Strict JSON storage metadata edits without reserializing scientific values."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


MAX_JSON_DEPTH = 64
MAX_JSON_TOKENS = 2_000_000


def _shape(text: str) -> None:
    depth = tokens = 0
    quoted = escaped = False
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
            tokens += 1
        elif char in "[{":
            depth += 1
            tokens += 1
        elif char in "]}":
            depth -= 1
        elif char in ",:":
            tokens += 1
        if depth > MAX_JSON_DEPTH or tokens > MAX_JSON_TOKENS:
            raise ValueError("publication JSON exceeds nesting/token limits")


def decode_storage_json(data: bytes) -> Any:
    """Reject ambiguous/non-JSON metadata and preserve arbitrary decimal values."""
    text = data.decode("utf-8")
    _shape(text)
    return json.loads(
        text, object_pairs_hook=_object, parse_constant=_constant, parse_float=Decimal
    )


def replace_metadata(data: bytes, updates: dict[tuple, str | int]) -> bytes:
    """Replace selected scalar leaves, preserving every other original byte.

    Validation is separate from locating token spans: the standard decoder
    rejects malformed JSON, duplicate keys and nonfinite constants first.
    """
    decode_storage_json(data)
    text = data.decode("utf-8")
    decoder = json.JSONDecoder(parse_float=Decimal, parse_constant=_constant)
    edits = []

    def whitespace(offset: int) -> int:
        while offset < len(text) and text[offset] in " \t\r\n":
            offset += 1
        return offset

    def visit(offset: int, path: tuple) -> int:
        offset = whitespace(offset)
        start = offset
        if text[offset] == "{":
            offset = whitespace(offset + 1)
            if text[offset] == "}":
                return offset + 1
            while True:
                key, offset = decoder.raw_decode(text, offset)
                offset = visit(whitespace(offset) + 1, (*path, key))
                offset = whitespace(offset)
                if text[offset] == "}":
                    return offset + 1
                offset = whitespace(offset + 1)
        if text[offset] == "[":
            offset = whitespace(offset + 1)
            if text[offset] == "]":
                return offset + 1
            index = 0
            while True:
                offset = whitespace(visit(offset, (*path, index)))
                if text[offset] == "]":
                    return offset + 1
                offset += 1
                index += 1
        _value, end = decoder.raw_decode(text, offset)
        if path in updates:
            edits.append((start, end, json.dumps(updates[path], ensure_ascii=False)))
        return end

    visit(0, ())
    if len(edits) != len(updates):
        raise ValueError("storage metadata replacement does not name a scalar leaf")
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8")
