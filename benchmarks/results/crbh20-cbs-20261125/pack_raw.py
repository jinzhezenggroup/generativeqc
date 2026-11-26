"""Preserve raw journals and process output in a compact allocation bundle."""

import argparse
import hashlib
import json
import lzma
from pathlib import Path

from protocol import load


def redact(text: str, assets_path: str, python_root: str | None) -> str:
    if python_root:
        text = text.replace(python_root, "$GQC_PYTHON")
    return text.replace(assets_path, "$GQC_ASSETS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("gpu", "cpu"))
    parser.add_argument("allocation", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets-path", required=True)
    parser.add_argument(
        "--python-root", help="Interpreter installation prefix to redact"
    )
    args = parser.parse_args()
    allocation = args.allocation
    sources = [
        allocation / "provenance.json",
        *sorted(allocation.glob("*.jsonl")),
        *sorted(allocation.glob("*.txt")),
    ]
    files = {path.name: path.read_bytes() for path in sources}
    sanitized = {
        name: redact(
            contents.decode(errors="replace"), args.assets_path, args.python_root
        )
        for name, contents in files.items()
    }
    journals, torn = {}, {}
    for name, text in sanitized.items():
        if not name.endswith(".jsonl"):
            continue
        lines = text.splitlines()
        rows = []
        for number, line in enumerate(lines, 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if number != len(lines) or text.endswith("\n"):
                    raise
                torn[name] = line
        journals[name] = rows
    bundle = {
        "kind": args.kind,
        "provenance": json.loads(sanitized["provenance.json"]),
        "journals": journals,
        "torn_final_lines": torn,
        "process_output": {
            name: text for name, text in sanitized.items() if name.endswith(".txt")
        },
        "acquired_sha256": {
            name: hashlib.sha256(contents).hexdigest()
            for name, contents in files.items()
        },
    }
    if not bundle["journals"] or not load(allocation / "provenance.json")["systems"]:
        raise ValueError("allocation has no endpoint journal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with lzma.open(args.output, "xt", preset=6) as stream:
        json.dump(bundle, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
