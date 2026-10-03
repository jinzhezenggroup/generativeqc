"""Verify/expand retained evidence; never execute bundled reproduction code."""

import argparse
import gzip
import hashlib
import json
import lzma
from pathlib import Path


def safe_path(name: str) -> Path:
    """Reject empty, noncanonical, absolute, and traversing archive paths."""
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("unsafe member path: " + name)
    return path


def verify_bytes(data: bytes, row: dict, prefix: str = "") -> None:
    if (
        len(data) != row[prefix + "bytes"]
        or hashlib.sha256(data).hexdigest() != row[prefix + "sha256"]
    ):
        raise ValueError("member size/checksum mismatch")


def load(root: Path) -> tuple[dict[str, bytes], dict]:
    """Validate the complete publication before creating any expanded output."""
    publication = json.loads((root / "publication.json").read_bytes())
    members = {}
    for row in publication["files"]:
        name = row["path"]
        path = safe_path(name)
        if name in members or name == "publication.json":
            raise ValueError("duplicate or self-referential publication member")
        target = root / path
        if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
            raise ValueError("publication member escapes package")
        data = target.read_bytes()
        verify_bytes(data, row)
        members[name] = data
    expected = set(members) - {"SHA256SUMS"}
    checksums = {}
    for line in members["SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        safe_path(name)
        if name in checksums:
            raise ValueError("duplicate checksum member")
        checksums[name] = digest
    if set(checksums) != expected or any(
        hashlib.sha256(members[name]).hexdigest() != digest
        for name, digest in checksums.items()
    ):
        raise ValueError("checksum inventory mismatch")
    prepared = {}
    documents = {}
    manifest = json.loads(members["manifest.json"])
    for row in manifest["files"]:
        name = row["path"]
        safe_path(name)
        data = members[name]
        verify_bytes(data, row)
        raw = lzma.decompress(data) if name.endswith(".xz") else data
        verify_bytes(raw, row, "decoded_")
        output = name.removesuffix(".xz")
        if output in prepared:
            raise ValueError("duplicate decoded member")
        prepared[output] = raw
        if output.endswith(".json"):
            documents[output] = json.loads(raw)
    evidence = documents["evidence.json"]
    primary, historical = evidence["primary_current_main"], evidence["historical"]
    counts = {
        "primary_processes": len(primary["processes"]),
        "primary_endpoints": sum(len(row["samples"]) for row in primary["processes"]),
        "historical_balanced_endpoints": sum(
            len(process[2])
            for cohort in historical["balanced_cohorts"]
            for process in cohort["processes"]
        ),
        "intrusive_endpoints": sum(
            len(row["samples"])
            for row in historical["intrusive_startup_diagnostic"]["raw_processes"]
        ),
        "historical_failed_dft_attempts": len(
            historical["scientific_summary"]["direct_spherical_dft"]["failed_attempts"]
        ),
    }
    if list(counts.values()) != [54, 216, 432, 108, 13]:
        raise ValueError("scientific count mismatch")
    for name, row in documents["reproducer.json"]["files"].items():
        relative = safe_path(name)
        data = row["text"].encode()
        verify_bytes(data, row)
        prepared[str(Path("reproducer") / relative)] = data
    prepared["validation.json"] = gzip.decompress(members["validation.json.gz"])
    json.loads(prepared["validation.json"])
    prepared["source-aliases.json"] = members["source-aliases.json"]
    prepared["README.md"] = members["README.md"]
    return prepared, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepared, counts = load(Path(__file__).resolve().parent)
    if args.output:
        # lexists-equivalent check also refuses dangling output symlinks.
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("refusing existing output directory")
        args.output.mkdir(parents=True, exist_ok=False)
        for name, raw in prepared.items():
            target = args.output / safe_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print(
        json.dumps(
            dict(
                counts,
                verified=True,
                raw_files="local only; not recovered",
                supplied_code_executed=False,
            )
        )
    )


if __name__ == "__main__":
    main()
