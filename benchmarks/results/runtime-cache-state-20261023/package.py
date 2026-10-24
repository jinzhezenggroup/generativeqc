"""Prepare a byte-recoverable logical capsule from a verified raw-file index."""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from capsule import (
    SCHEMA,
    encode_member,
    json_encoding,
    list_parts,
    pretty,
    restore,
    sha,
    store,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.raw / "raw-index.json").read_bytes())
    records = index["files"]
    if len({row["path"] for row in records}) != len(records):
        raise ValueError("duplicate raw input")
    config = json.loads((args.raw / "frozen-inputs/config.json").read_bytes())
    args.output.mkdir(parents=True, exist_ok=False)
    stored_config = copy.deepcopy(config)
    source_rows = json.loads((args.raw / "source-ececb48-manifest.json").read_bytes())[
        "files"
    ]
    source_sha = {row["sha256"]: index for index, row in enumerate(source_rows)}
    for item in stored_config["sdk_files"]:
        if item["sha256"] in source_sha:
            item["sha256"] = {"source_sha256_row": source_sha[item["sha256"]]}
    configuration = {
        "codec": "source-sha256-references.v1",
        "configuration": stored_config,
    }
    manifest = {
        "schema": SCHEMA,
        "configuration": store(args.output, "configuration.json.xz", configuration),
        "file_manifests": [],
        "raw_parts": [],
        "aliased_files": [],
        "raw_index_sha256": sha((args.raw / "raw-index.json").read_bytes()),
    }
    groups = defaultdict(list)
    seen = {}
    priority = {
        "source-ececb48-manifest.json": 0,
        "receipts/scientific-environment.json": 1,
    }
    for row in sorted(
        records, key=lambda item: (priority.get(item["path"], 2), item["path"])
    ):
        relative = row["path"]
        path = args.raw / relative
        raw = path.read_bytes()
        if sha(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError(f"raw input changed after remote index: {relative}")
        if row["sha256"] in seen:
            manifest["aliased_files"].append({**row, "alias_of": seen[row["sha256"]]})
            continue
        seen[row["sha256"]] = relative
        if relative in (
            "source-ececb48-manifest.json",
            "receipts/scientific-environment.json",
        ):
            value = json.loads(raw)
            encoding = json_encoding(value, raw)
            if encoding is None:
                raise ValueError(
                    "large manifest is not the original declared serialization"
                )
            rows = value["files"]
            value["files"] = None  # Preserve the original dictionary position.
            label = "source" if relative.startswith("source-") else "environment"
            manifest["file_manifests"].append(
                {
                    "original": row,
                    "properties": value,
                    "rows": len(rows),
                    "encoding": encoding,
                    "parts": list_parts(args.output, label, rows),
                }
            )
            continue
        member = encode_member(path, relative, config)
        member["mode"] = row["mode"]
        if relative == "frozen-inputs/config.json":
            member = {**row, "encoding": "configuration-json"}
        if relative.startswith(("pilot/", "formal/")) and "/process-" in relative:
            cohort = relative.split("/", 1)[0]
            request = json.loads((path.parent / "request.json").read_bytes())
            group = cohort + "-" + request["case"]
        else:
            stem = relative.replace("/", "-").replace(".", "-")
            group = (
                "preparation-" + stem
                if not relative.startswith(("pilot/", "formal/"))
                else "campaigns"
            )
        groups[group].append(member)
    for group, members in sorted(groups.items()):
        manifest["raw_parts"].append(store(args.output, group + ".json.xz", members))
    input_index = encode_member(args.raw / "raw-index.json", "raw-index.json", config)
    input_index["mode"] = index.get("control_file_mode", input_index["mode"])
    manifest["raw_parts"].append(
        store(args.output, "preparation-input-index.json.xz", [input_index])
    )
    store(args.output, "raw-manifest.json.xz", manifest)
    recovered = args.output.parent / (args.output.name + "-recovered")
    result = restore(args.output, recovered)
    for row in records:
        data = (recovered / row["path"]).read_bytes()
        if sha(data) != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError("complete raw-index recovery failed")
    result.update(
        stored_bytes=sum(
            path.stat().st_size for path in args.output.rglob("*") if path.is_file()
        ),
        source_files=index.get("source_files"),
        status="raw recovery PASS; scientific gates separate",
    )
    (args.output / "recovery.json").write_bytes(pretty(result))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
