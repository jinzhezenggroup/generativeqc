"""Lossless logical JSON parts for the cache-state raw receipts.

Parts group source files by subsystem and environment files by Python package;
they are not arbitrary byte fragments. Every recovered original file is checked
against its original byte length and SHA-256 before an output can be published.
"""

from __future__ import annotations

import base64
import hashlib
import json
import lzma
import os
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

LIMIT = 128 * 1024
SCHEMA = "generativeqc.runtime-cache-raw.v1"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pretty(value: Any) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def json_encoding(value: Any, raw: bytes) -> str | None:
    encoded = pretty(value)
    if encoded == raw:
        return "json-indent2-lf"
    if encoded.replace(b"\n", b"\r\n") == raw:
        return "json-indent2-crlf"
    return None


def relative_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or "\\" in value
        or ":" in value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
    ):
        raise ValueError("unsafe capsule member path")
    return path


def store(root: Path, name: str, value: Any) -> dict:
    data = lzma.compress(pretty(value), preset=6)
    if len(data) >= LIMIT:
        raise ValueError(
            f"logical part needs a finer scientific grouping: {name} ({len(data)} bytes)"
        )
    path = root / relative_name(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_bytes(data)
    return {"path": name, "bytes": len(data), "sha256": sha(data)}


def load(root: Path, entry: dict) -> Any:
    relative = relative_name(entry["path"])
    raw = (root / relative).read_bytes()
    if len(raw) != entry["bytes"] or sha(raw) != entry["sha256"]:
        raise ValueError("capsule part byte identity differs")
    return json.loads(lzma.decompress(raw))


def initial_group(path: str) -> str:
    parts = PurePosixPath(path).parts
    if "site-packages" in parts:
        index = parts.index("site-packages")
        return "package/" + parts[index + 1]
    if parts[:2] == ("benchmarks", "results") and len(parts) > 2:
        name = parts[2].lower()
        families = {
            "readme-and-endpoints": ("readme", "endpoint", "default", "production"),
            "dft-and-xc": (
                "pbe",
                "wb97",
                "wb97m",
                "r2scan",
                "xc-",
                "dft",
                "grid",
                "lda",
                "b3lyp",
                "m06",
                "hybrid",
                "nonlocal",
                "laplacian",
                "semilocal",
                "vv10",
                "mgga",
            ),
            "correlation": ("mp2", "ccsdt", "ccsd", "cbs", "correlation", "dfcc"),
            "integrals-and-scf": (
                "df-",
                "direct",
                "hf-",
                "integral",
                "rys",
                "shell",
                "fock",
                "metric",
                "gram",
                "coulomb",
                "exchange",
                "schwarz",
            ),
            "compiler-and-runtime": (
                "cache",
                "native",
                "compiler",
                "tensor",
                "closure",
                "array-api",
                "compiled",
                "codegen",
                "library",
            ),
        }
        return "benchmark-inputs/" + next(
            (
                family
                for family, tokens in families.items()
                if any(token in name for token in tokens)
            ),
            "historical-contracts",
        )
    if parts[:2] == ("tests", "python") and len(parts) > 2:
        name = PurePosixPath(parts[2]).stem.removeprefix("test_")
        domains = {
            "compiler-and-arrays": (
                "tensor",
                "array",
                "graph",
                "compiler",
                "codegen",
                "scalar",
                "expression",
                "ir_",
                "cpp",
                "cpu_",
                "native_runtime",
            ),
            "dft-and-xc": (
                "dft",
                "ks_",
                "xc_",
                "wb97",
                "pbe",
                "libxc",
                "grid",
                "point_",
                "r2scan",
                "becke",
            ),
            "methods-and-integrals": (
                "cc",
                "mp2",
                "df_",
                "scf",
                "hf_",
                "integral",
                "shell",
                "force",
                "gradient",
                "hessian",
                "gfn",
                "stationary",
                "direct_",
            ),
        }
        return "test-inputs/" + next(
            (
                domain
                for domain, prefixes in domains.items()
                if name.startswith(prefixes)
            ),
            "contracts-and-tooling",
        )
    if len(parts) == 1:
        return "repository-root-files"
    if len(parts) == 2:
        return parts[0] + "/entrypoints-and-root-files"
    return "/".join(parts[:2])


def list_parts(root: Path, label: str, rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[initial_group(row["path"])].append([index, row])
    output = []

    def write_group(key: str, values: list, depth: int = 1) -> None:
        components = PurePosixPath(key).parts
        if "site-packages" in components:
            key_label = "package/" + "/".join(
                components[components.index("site-packages") + 1 :]
            )
        else:
            key_label = key
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", key_label).strip("-")[:72]
        name = f"parts/{label}-{slug}-{sha(key.encode())[:8]}.json.xz"
        if len(lzma.compress(pretty(values), preset=6)) < LIMIT:
            output.append(store(root, name, values))
            return
        # Refine by actual directory components, preserving original row indices.
        nested = defaultdict(list)
        for value in values:
            path = PurePosixPath(value[1]["path"]).parts
            nested["/".join(path[:depth])].append(value)
        if len(nested) == 1:
            if depth >= max(len(PurePosixPath(v[1]["path"]).parts) for v in values):
                raise ValueError("one manifest entry exceeds the logical part cap")
            write_group(key, values, depth + 1)
            return
        for child, items in sorted(nested.items()):
            write_group(child, items, depth + 1)

    for key, values in sorted(groups.items()):
        write_group(key, values)
    return output


def encode_member(path: Path, relative: str, config: dict) -> dict:
    raw = path.read_bytes()
    result = {
        "path": relative,
        "bytes": len(raw),
        "sha256": sha(raw),
        "mode": f"{path.stat().st_mode & 0o777:04o}",
    }
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        value = None
    encoding = json_encoding(value, raw) if value is not None else None
    if encoding:
        result.update(encoding=encoding, value=value)
        if isinstance(value, dict) and "config" in value:
            original = value["config"]
            if not isinstance(original, dict) or any(
                key not in config or pretty(item) != pretty(config[key])
                for key, item in original.items()
            ):
                raise ValueError("raw process contains a different configuration")
            result["config_reference"] = list(original)
            value["config"] = None
    else:
        try:
            result.update(encoding="utf8", value=raw.decode("utf-8"))
        except UnicodeDecodeError:
            result.update(
                encoding="base64", value=base64.b64encode(raw).decode("ascii")
            )
    return result


def decode_member(member: dict, config: dict) -> bytes:
    if member["encoding"] == "configuration-json":
        raw = pretty(config)
    elif member["encoding"] in ("json-indent2-lf", "json-indent2-crlf"):
        value = member["value"]
        if "config_reference" in member:
            value = dict(value)
            value["config"] = {key: config[key] for key in member["config_reference"]}
        raw = pretty(value)
        if member["encoding"] == "json-indent2-crlf":
            raw = raw.replace(b"\n", b"\r\n")
    elif member["encoding"] == "utf8":
        raw = member["value"].encode("utf-8")
    elif member["encoding"] == "base64":
        raw = base64.b64decode(member["value"], validate=True)
    else:
        raise ValueError("unknown raw-member encoding")
    if len(raw) != member["bytes"] or sha(raw) != member["sha256"]:
        raise ValueError(f"lossless raw recovery failed: {member['path']}")
    return raw


def restore(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    compressed_manifest = root / "raw-manifest.json.xz"
    manifest = json.loads(
        lzma.decompress(compressed_manifest.read_bytes())
        if compressed_manifest.exists()
        else (root / "raw-manifest.json").read_bytes()
    )
    if manifest["schema"] != SCHEMA:
        raise ValueError("unknown capsule schema")
    members = []
    source_rows = None
    for descriptor in manifest["file_manifests"]:
        value = descriptor["properties"]
        rows = [row for part in descriptor["parts"] for row in load(root, part)]
        if sorted(row[0] for row in rows) != list(range(descriptor["rows"])):
            raise ValueError("missing/duplicate manifest row")
        value["files"] = [row[1] for row in sorted(rows, key=lambda row: row[0])]
        member = {
            **descriptor["original"],
            "encoding": descriptor["encoding"],
            "value": value,
        }
        raw = decode_member(member, {})
        members.append((member, raw))
        if member["path"] == "source-ececb48-manifest.json":
            source_rows = value["files"]
    stored_config = load(root, manifest["configuration"])
    if stored_config.get("codec") == "source-sha256-references.v1":
        if source_rows is None:
            raise ValueError("configuration references a missing source manifest")
        config = stored_config["configuration"]
        for item in config["sdk_files"]:
            reference = item["sha256"]
            if isinstance(reference, dict):
                if set(reference) != {"source_sha256_row"}:
                    raise ValueError("unknown configuration hash reference")
                index = reference["source_sha256_row"]
                if type(index) is not int or not 0 <= index < len(source_rows):
                    raise ValueError("invalid configuration source reference")
                item["sha256"] = source_rows[index]["sha256"]
    else:
        config = stored_config
    for entry in manifest["raw_parts"]:
        members.extend(
            (member, decode_member(member, config)) for member in load(root, entry)
        )
    by_name = {member["path"]: raw for member, raw in members}
    for member in manifest.get("aliased_files", []):
        raw = by_name[member["alias_of"]]
        if len(raw) != member["bytes"] or sha(raw) != member["sha256"]:
            raise ValueError("aliased raw byte identity differs")
        members.append((member, raw))
        by_name[member["path"]] = raw
    names = [member["path"] for member, raw in members]
    if len(names) != len(set(names)):
        raise ValueError("duplicate restored raw file")
    for member, raw in members:
        path = relative_name(member["path"])
        if ".env" in path.parts or path.name.startswith(".env."):
            raise ValueError("unsafe recovered member path")
    output.mkdir(parents=True, exist_ok=False)
    for member, raw in members:
        path = output / member["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        if os.name == "posix":
            path.chmod(int(member["mode"], 8))
    return {
        "files": len(members),
        "raw_bytes": sum(len(raw) for member, raw in members),
        "raw_sha256_and_lengths": "PASS",
        "physical_modes": "restored"
        if os.name == "posix"
        else "UNAVAILABLE on this host",
    }
