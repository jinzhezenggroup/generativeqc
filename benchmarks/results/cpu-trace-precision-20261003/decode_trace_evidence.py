"""Verify or expand scientific JSON and exact harnesses; never execute them."""

import argparse
import gzip
import hashlib
import json
import typing
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def load(root: Path) -> tuple[dict[str, typing.Any], dict[str, bytes]]:
    manifest = json.loads((root / "capsule.json").read_bytes())
    for name, row in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("unsafe capsule path")
        data = (root / name).read_bytes()
        if len(data) != row["bytes"] or sha(data) != row["sha256"]:
            raise ValueError("capsule member changed: " + name)
    payload = json.loads(
        gzip.decompress((root / "scientific-records.json.gz").read_bytes())
    )
    if manifest.get("codec") == "shared-json-v1":
        values = []

        def expand(value: typing.Any) -> typing.Any:
            if isinstance(value, dict):
                if set(value) == {"$trace_ref"}:
                    index = value["$trace_ref"]
                    if type(index) is not int or not 0 <= index < len(values):
                        raise ValueError("invalid or cyclic reference")
                    return values[index]
                return {k: expand(v) for k, v in value.items()}
            return [expand(v) for v in value] if isinstance(value, list) else value

        for value in payload["table"]:
            values.append(expand(value))
        payload = expand(payload["root"])
    if sha(canonical(payload)) != manifest["expanded_payload_sha256"]:
        raise ValueError("expanded payload changed")
    prepared = {}
    for name, value in payload["records"].items():
        prepared[name] = (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    sources = payload["sources"]
    for name, row in sources.items():
        if "text" in row:
            data = row["text"]
        else:
            base = sources[row["base"]]["text"]
            data = "".join(
                base[p[0] : p[1]] if isinstance(p, list) else p for p in row["pieces"]
            )
        raw = data.encode()
        if sha(raw) != row["sha256"]:
            raise ValueError("reconstructed harness changed: " + name)
        prepared[name] = raw
    prepared["identities.json"] = (
        json.dumps(payload["identities"], indent=2, sort_keys=True) + "\n"
    ).encode()
    for name in prepared:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError("unsafe expanded path")
    return manifest, prepared


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("verify", "expand"))
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    manifest, prepared = load(Path(__file__).resolve().parent)
    if args.action == "expand":
        if args.output is None or args.output.exists():
            raise ValueError("expansion requires a new --output directory")
        args.output.mkdir(parents=True)
        for name, data in prepared.items():
            p = args.output / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    print(
        json.dumps(
            {
                "verified": True,
                "contract": manifest["contract"],
                "expanded_files": len(prepared),
                "expanded_payload_sha256": manifest["expanded_payload_sha256"],
            }
        )
    )


if __name__ == "__main__":
    main()
