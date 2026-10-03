"""Verify/expand hash-bound scientific JSON. Never execute recovered sources."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def load(root: Path) -> tuple[dict, dict[str, str], dict]:
    manifest = json.loads((root / "publication.json").read_bytes())
    for row in manifest["files"]:
        name = row["path"]
        if Path(name).name != name:
            raise ValueError("unsafe publication path")
        raw = (root / name).read_bytes()
        if len(raw) != row["bytes"] or digest(raw) != row["sha256"]:
            raise ValueError("publication member changed: " + name)
    raw = gzip.decompress((root / "scientific-records.json.gz").read_bytes())
    graph = json.loads(raw)
    if graph["schema"] != "generativeqc.lossless-json-dag.v1":
        raise ValueError("unsupported storage graph")
    values = []
    for node in graph["nodes"]:
        kind, value = node

        def child(index: int) -> object:
            if type(index) is not int or not 0 <= index < len(values):
                raise ValueError("invalid/cyclic graph reference")
            return values[index]

        if kind == "v":
            if isinstance(value, (list, dict)):
                raise ValueError("container in scalar node")
        elif kind == "l":
            value = [child(index) for index in value]
        elif kind == "d":
            pairs = [(child(key), child(item)) for key, item in value]
            if any(not isinstance(key, str) for key, _ in pairs) or len(
                {key for key, _ in pairs}
            ) != len(pairs):
                raise ValueError("invalid dictionary keys")
            value = dict(pairs)
        else:
            raise ValueError("invalid node kind")
        values.append(value)
    root_index = graph["root"]
    if type(root_index) is not int or not 0 <= root_index < len(values):
        raise ValueError("invalid graph root index")
    payload = values[root_index]
    if digest(canonical(payload)) != graph["expanded_sha256"]:
        raise ValueError("expanded scientific JSON changed")
    sources = payload["reproduction_sources"]
    expanded = {}
    for path, row in sources.items():
        if "text" in row:
            text = row["text"]
        else:
            base = sources[row["base"]]["text"]
            text = "".join(
                base[p["copy"][0] : p["copy"][1]] if "copy" in p else p["text"]
                for p in row["pieces"]
            )
        if digest(text.encode()) != row["sha256"]:
            raise ValueError("recovered source differs from original receipt: " + path)
        expanded[path] = text
    return payload, expanded, graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "expand"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload, sources, graph = load(Path(__file__).resolve().parent)
    if args.action == "expand":
        if args.output is None or args.output.exists():
            raise ValueError("expansion needs a new --output directory")
        prepared = {}
        for name, value in payload["scientific_records"].items():
            prepared[name] = (
                json.dumps(
                    value.get("record", value),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            ).encode()
        prepared.update({name: text.encode() for name, text in sources.items()})
        prepared["expanded-identities.json"] = (
            json.dumps(
                {
                    "expanded_payload_sha256": graph["expanded_sha256"],
                    "source_originals": {
                        name: {
                            key: row[key]
                            for key in ("original_sha256", "original_bytes")
                            if key in row
                        }
                        for name, row in payload["scientific_records"].items()
                        if "original_sha256" in row
                    },
                },
                indent=2,
            )
            + "\n"
        ).encode()
        for name in prepared:
            path = Path(name)
            if path.is_absolute() or any(p in {".", ".."} for p in path.parts):
                raise ValueError("unsafe expanded path")
        args.output.mkdir(parents=True)
        for name, raw in prepared.items():
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print(
        json.dumps(
            {
                "scientific_records": len(payload["scientific_records"]),
                "source_snapshots": len(sources),
                "expanded_payload_sha256": graph["expanded_sha256"],
                "verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
