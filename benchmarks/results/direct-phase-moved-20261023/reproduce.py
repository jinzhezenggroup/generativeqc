"""Restore retained bytes and recompute the bounded PBE0 cohort without a GPU."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import lzma
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

# The immutable audit script uses assertions for its acceptance gates. Refuse
# optimized execution at import as well as at the CLI, before restoring bytes.
if sys.flags.optimize:
    raise RuntimeError("replay requires Python assertions; disable optimization")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def safe_path(name: str) -> Path:
    """Apply the shared reader's canonical relative-path constraints."""
    if (
        not isinstance(name, str)
        or not name
        or PurePosixPath(name).is_absolute()
        or any(part in ("", ".", "..") for part in name.split("/"))
        or "\\" in name
        or ":" in name
    ):
        raise ValueError("unsafe retained path")
    return Path(*PurePosixPath(name).parts)


def decode_xz(raw: bytes) -> bytes:
    """Bound memory and expanded bytes; reject truncation or extra streams."""
    decoder = lzma.LZMADecompressor(memlimit=96 << 20)
    decoded = decoder.decompress(raw, max_length=(16 << 20) + 1)
    if len(decoded) > 16 << 20 or not decoder.eof or decoder.unused_data:
        raise ValueError("invalid or oversized retained XZ stream")
    return decoded


def restore(bundle: Path, output: Path) -> None:
    """Check all publication/member hashes before creating any restored file."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    manifest = json.loads((bundle / "publication.json").read_text())
    entries = {}
    for entry in manifest["files"]:
        name = entry["path"]
        path = bundle / safe_path(name)
        if (
            name in entries
            or not path.resolve().is_relative_to(bundle.resolve())
            or type(entry["bytes"]) is not int
            or not 0 <= entry["bytes"] <= 1 << 20
        ):
            raise ValueError("duplicate, escaping or oversized publication entry")
        entries[name] = (entry, path)
    if "scientific-environment.json.xz" not in entries:
        raise ValueError("missing scientific environment entry")
    payloads = {}
    for entry, path in entries.values():
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("publication size mismatch")
        raw = path.read_bytes()
        if len(raw) != entry["bytes"] or digest(raw) != entry["sha256"]:
            raise ValueError(f"publication identity mismatch: {entry['path']}")
        if path.name not in {
            "selected.json.xz",
            "diagnostics.json.xz",
            "cpu.json.xz",
            "sdk.json.xz",
            "execution.json.xz",
            "protocols.json.xz",
        }:
            continue
        record = json.loads(decode_xz(raw))
        if record.get("schema") != "generativeqc.direct-phase-moved.records.v1":
            raise ValueError("unexpected retained record collection")
        for member in record["files"]:
            name = member["path"]
            safe_path(name)
            if ("utf8" in member) == ("base64" in member):
                raise ValueError("ambiguous member encoding")
            data = (
                member["utf8"].encode("utf-8")
                if "utf8" in member
                else base64.b64decode(member["base64"], validate=True)
            )
            if (
                len(data) != member["bytes"]
                or digest(data) != member["sha256"]
                or name in payloads
            ):
                raise ValueError(f"retained member identity mismatch: {name}")
            payloads[name] = data
    environment = decode_xz((bundle / "scientific-environment.json.xz").read_bytes())
    if (
        digest(environment)
        != "6da3a11b6ae0ace154745fd546895d13a8e7768f8a9f1193628d1d4d537680cd"
    ):
        raise ValueError("environment manifest identity mismatch")
    environment_name = "retrieved-final/scientific-environment.json"
    if environment_name in payloads:
        raise ValueError("duplicate scientific environment member")
    payloads[environment_name] = environment
    output.mkdir(parents=True, exist_ok=False)
    for name, data in payloads.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if path.read_bytes() != data:
            raise ValueError("restored bytes differ")
    print(f"Restored and byte-verified {len(payloads)} retained files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[3]
    )
    args = parser.parse_args()
    restore(Path(__file__).resolve().parent, args.output)
    # The immutable audit eagerly evaluates a parents[3] CLI default even when
    # --repo is supplied. Execute identical bytes at a sufficiently deep temporary
    # path so any accepted output depth works; restored evidence stays untouched.
    with tempfile.TemporaryDirectory(prefix="direct-phase-moved-") as directory:
        audit = Path(directory) / "retained" / "audit" / "audit-selected.py"
        audit.parent.mkdir(parents=True)
        raw = (args.output / "audit-selected.py").read_bytes()
        audit.write_bytes(raw)
        if audit.read_bytes() != raw:
            raise ValueError("temporary audit bytes differ")
        subprocess.run(
            [
                sys.executable,
                "-E",  # Keep assertions enabled if the parent ignored PYTHONOPTIMIZE.
                str(audit),
                "--records",
                str(args.output.resolve()),
                "--repo",
                str(args.repo.resolve()),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
