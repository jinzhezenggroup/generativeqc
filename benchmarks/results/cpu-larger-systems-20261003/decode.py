"""Verify and expand local evidence/protocol text; never execute recovered code."""

import argparse
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or "\x00" in name
    ):
        raise ValueError("unsafe member path")
    path = Path(name)
    if path.is_absolute() or any(x in {"", ".", ".."} for x in name.split("/")):
        raise ValueError("unsafe member path")
    return path


def verify(raw, row):
    if len(raw) != row["bytes"] or sha(raw) != row["sha256"]:
        raise ValueError("byte identity mismatch")


def unzip(raw, xz=False):
    if xz:
        decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=128 * 1024**2)
        out = decoder.decompress(raw, max_length=8 * 1024**2 + 1)
    else:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:
            out = f.read(8 * 1024**2 + 1)
    if len(out) > 8 * 1024**2:
        raise ValueError("decoded payload exceeds bounded text limit")
    if xz and (not decoder.eof or decoder.unused_data):
        raise ValueError("incomplete or trailing XZ payload")
    return out


def load(root):
    publication = json.loads((root / "publication.json").read_bytes())
    if publication["schema"] != "generativeqc.benchmark-publication.v1":
        raise ValueError("wrong publication schema")
    files = {}
    for row in publication["files"]:
        name = row["path"]
        relative = safe(name)
        if name in files or name == "publication.json" or len(relative.parts) != 1:
            raise ValueError("duplicate/nonlocal publication member")
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("publication member escapes root")
        raw = path.read_bytes()
        verify(raw, row)
        files[name] = raw
    if set(files) != {
        "README.md",
        "decode.py",
        "scientific-records.json.xz",
        "validation.json.gz",
        "SHA256SUMS",
    }:
        raise ValueError("unexpected publication files")
    checks = {}
    for line in files["SHA256SUMS"].decode().splitlines():
        digest, name = line.split("  ", 1)
        safe(name)
        if name in checks:
            raise ValueError("duplicate checksum entry")
        checks[name] = digest
    if set(checks) != set(files) - {"SHA256SUMS"} or any(
        sha(files[n]) != h for n, h in checks.items()
    ):
        raise ValueError("checksum inventory mismatch")
    payload_raw = unzip(files["scientific-records.json.xz"], xz=True)
    verify(payload_raw, publication["bundle"])
    payload = json.loads(payload_raw)
    if payload["schema"] != "generativeqc.cpu-larger-text-bundle.v3":
        raise ValueError("wrong scientific bundle schema")
    members = {}
    for row in payload["members"]:
        name = row["path"]
        path = safe(name)
        if path.parts[0] == "validation.json":
            raise ValueError("reserved generated member path")
        if name in members:
            raise ValueError("duplicate decoded member")
        raw = row["text"].encode("utf-8")
        verify(raw, row)
        members[name] = raw
    for name in members:
        if any(str(parent) in members for parent in safe(name).parents):
            raise ValueError("file/directory member collision")
    if (
        len(members) != publication["bundle"]["expanded_member_count"]
        or sum(map(len, members.values()))
        != publication["bundle"]["expanded_member_bytes"]
    ):
        raise ValueError("expanded member count/bytes mismatch")
    scalars = json.loads(members["evidence/scalars.json"])
    inventory = json.loads(members["evidence/inventory.json"])
    originals = {
        str(Path(inventory["directories"][d]) / inventory["names"][n]): [size, h]
        for d, n, size, h in inventory["files"]
    }
    for row in payload["members"]:
        if "original_path" in row and originals.get(row["original_path"]) != [
            row["bytes"],
            row["sha256"],
        ]:
            raise ValueError("copied original differs from retained byte inventory")
    counts = {
        "processes": len(scalars["processes"]),
        "endpoints": len(scalars["endpoints"]),
        "oracle_states": len(scalars["oracle_states"]),
        "original_files": len(originals),
    }
    if list(counts.values()) != [11, 44, 8, 372]:
        raise ValueError("scientific count mismatch")
    for cohort, expected in [
        ("pentane-pilot", 8),
        ("pentane-final", 32),
        ("hexane-candidate", 4),
    ]:
        if (
            sum(scalars["processes"][e[0]][0] == cohort for e in scalars["endpoints"])
            != expected
        ):
            raise ValueError("cohort endpoint count mismatch")
    required = {
        "frozen/runner/" + x + ".py"
        for x in ("common", "driver", "endpoint", "test_runner")
    }
    required |= {
        "frozen/qualification-harness/" + x + ".py"
        for x in (
            "freeze_inputs",
            "run_oracle",
            "run_oracle_full_density",
            "finalize_oracles",
        )
    }
    required |= {
        "frozen/qualification-harness/" + x + ".sh"
        for x in ("run_all_oracles", "run_all_oracles_full_density")
    }
    required |= {
        "frozen/inputs/" + x + ".json"
        for x in ("manifest", "n-pentane", "n-hexane", "n-heptane")
    }
    if not required <= members.keys():
        raise ValueError("essential protocol/input member missing")
    validation_raw = unzip(files["validation.json.gz"])
    validation = json.loads(validation_raw)
    if (
        validation["schema"] != "generativeqc.validation"
        or validation["revision"] != publication["source"]["revision"]
    ):
        raise ValueError("validation/source mismatch")
    for attachment in validation["attachments"]:
        if (
            attachment["path"] not in files
            or sha(files[attachment["path"]]) != attachment["sha256"]
        ):
            raise ValueError("validation attachment mismatch")
    if (
        publication["decision"]["status"] != "inconclusive"
        or validation["performance"]["status"] != "not-run"
    ):
        raise ValueError("unsupported performance promotion")
    members["validation.json"] = validation_raw
    counts.update(
        expanded_files=len(members), expanded_bytes=sum(map(len, members.values()))
    )
    return members, counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    members, counts = load(Path(__file__).resolve().parent)
    if args.output:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("refusing existing output")
        args.output.mkdir(parents=True, exist_ok=False)
        for name, raw in members.items():
            path = args.output / safe(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    print(
        json.dumps(
            dict(
                verified=True,
                supplied_scientific_code_executed=False,
                omitted_raw_arrays_recovered=False,
                **counts,
            )
        )
    )


if __name__ == "__main__":
    main()
