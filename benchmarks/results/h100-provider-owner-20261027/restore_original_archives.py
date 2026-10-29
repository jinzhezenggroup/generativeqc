"""Recover the three original published V4/V6 compressed archives; never run native code."""

import argparse
import hashlib
import io
import json
import lzma
import tarfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "restore requires a fresh directory"
    root = Path(__file__).resolve().parent
    summary = json.loads((root / "summary.json").read_bytes())
    delta_bytes = (root / "source-closure.json.xz").read_bytes()
    assert (
        hashlib.sha256(delta_bytes).hexdigest()
        == summary["archives"]["source-closure.json.xz"]["sha256"]
    )
    delta = json.loads(lzma.decompress(delta_bytes))
    anchor = delta["base_archive"]
    assert anchor["path"] == "v6/source-closure.json.xz"
    base = (root / anchor["path"]).read_bytes()
    assert (
        len(base) == anchor["bytes"]
        and hashlib.sha256(base).hexdigest() == anchor["sha256"]
    )
    sources = json.loads(lzma.decompress(base))["sources"]
    source = {
        "manifest": delta["manifest"],
        "sources": {
            name: delta["changed_sources"][name]
            if name in delta["changed_sources"]
            else sources[name]
            for name in delta["manifest"]["files"]
        },
    }
    raw = (json.dumps(source, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    assert hashlib.sha256(raw).hexdigest() == delta["original_uncompressed_sha256"]
    restored = {"v4/source-closure.json.xz": lzma.compress(raw, preset=9)}
    pool_meta = summary["receipt_storage"]
    pool = (root / pool_meta["path"]).read_bytes()
    assert (
        len(pool) == pool_meta["bytes"]
        and hashlib.sha256(pool).hexdigest() == pool_meta["sha256"]
    )
    with tarfile.open(fileobj=io.BytesIO(pool), mode="r:xz") as archive:
        assert set(archive.getnames()) == {"v4/receipt.json", "v6/receipt.json"}
        for entry in archive:
            assert entry.isfile()
            restored[entry.name + ".xz"] = lzma.compress(
                archive.extractfile(entry).read(), preset=9
            )
    expected = {
        "v4/source-closure.json.xz": summary["original_archives"][
            "source-closure.json.xz"
        ],
        "v4/receipt.json.xz": summary["original_archives"]["receipt.json.xz"],
        "v6/receipt.json.xz": json.loads((root / "v6/summary.json").read_bytes())[
            "original_archives"
        ]["receipt.json.xz"],
    }
    for name, data in restored.items():
        assert (
            len(data) == expected[name]["bytes"]
            and hashlib.sha256(data).hexdigest() == expected[name]["sha256"]
        )
    args.output.mkdir()
    for name, data in restored.items():
        path = args.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    print(
        json.dumps(
            {
                "status": "PASS_EXACT_ORIGINAL_COMPRESSED_ARCHIVE_RECOVERY",
                "files": expected,
                "science": "NOT_RUN",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
