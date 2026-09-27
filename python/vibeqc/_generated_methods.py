"""Runtime public-method metadata derived from the audited manifest.

This module intentionally keeps the historical import path used by the Python
API, but it is not a generated source file. Source checkouts read the canonical
repository manifest directly; installed wheels read the same manifest bundled
as package data.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _manifest_text() -> str:
    bundled = resources.files("vibeqc").joinpath("data/public_methods.json")
    if bundled.is_file():
        return bundled.read_text(encoding="utf-8")

    source = Path(__file__).resolve().parents[2] / "manifests/public_methods.json"
    if source.is_file():
        return source.read_text(encoding="utf-8")

    raise RuntimeError("public method manifest is unavailable")


def _load_manifest() -> dict[str, Any]:
    payload = json.loads(_manifest_text())
    if payload.get("schema_version") != 2:
        raise RuntimeError("unsupported public method manifest schema")
    if not isinstance(payload.get("methods"), list):
        raise RuntimeError("public method manifest requires methods")
    if not isinstance(payload.get("composite_methods", []), list):
        raise RuntimeError("public method manifest composite_methods must be a list")
    return payload


_PAYLOAD = _load_manifest()
_METHODS = _PAYLOAD["methods"]
_COMPOSITES = _PAYLOAD.get("composite_methods", [])

METHOD_CONSTANTS = MappingProxyType(
    {f"METHOD_{method['symbol']}": int(method["abi_id"]) for method in _METHODS}
)
globals().update(METHOD_CONSTANTS)

METHOD_METADATA = MappingProxyType(
    {
        str(method["name"]): MappingProxyType(
            {
                "abi_id": int(method["abi_id"]),
                "family": str(method["family"]),
                "provider": str(method["provider"]),
                "properties": tuple(method["properties"]),
                "supports_batch": bool(method["supports_batch"]),
                "aliases": tuple(method.get("aliases", [])),
            }
        )
        for method in _METHODS
    }
)

_method_name_to_id: dict[str, int] = {}
for _method in _METHODS:
    _method_id = int(_method["abi_id"])
    _method_name_to_id[str(_method["name"])] = _method_id
    for _alias in _method.get("aliases", []):
        _method_name_to_id[str(_alias)] = _method_id
METHOD_NAME_TO_ID = MappingProxyType(_method_name_to_id)

METHOD_ID_TO_NAME = MappingProxyType(
    {int(method["abi_id"]): str(method["name"]) for method in _METHODS}
)

HF_METHOD_IDS = frozenset(
    int(method["abi_id"]) for method in _METHODS if method["provider"] == "hf"
)
NATIVE_DFT_METHOD_IDS = frozenset(
    int(method["abi_id"]) for method in _METHODS if method["provider"] == "dft"
)

_composite_aliases: dict[str, tuple[str, str]] = {}
for _method in _COMPOSITES:
    _target = (str(_method["identifier"]), str(_method["spin"]))
    _composite_aliases[str(_method["name"])] = _target
    for _alias in _method.get("aliases", []):
        _composite_aliases[str(_alias)] = _target
COMPOSITE_METHOD_ALIASES = MappingProxyType(_composite_aliases)
