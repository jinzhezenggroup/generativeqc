"""Keep the larger CPU scalar/protocol capsule explicit and hash-bound."""

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType

import pytest

from tools.generativeqc_validation.publication import validate_publication

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/cpu-larger-systems-20261003"


def decoder() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "larger_cpu_decoder", CAPSULE / "decode.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_larger_cpu_evidence_scope_and_exact_protocol() -> None:
    publication = json.loads((CAPSULE / "publication.json").read_text())
    files = {
        r["path"]: (CAPSULE / r["path"]).read_bytes() for r in publication["files"]
    }
    validate_publication(publication, files)
    assert publication["decision"]["status"] == "inconclusive"
    assert hashlib.sha256(files["scientific-records.json.xz"]).hexdigest() == (
        "9b08aa1b39926c68a308a571fab9164df9cbce5decdc6b90c6c610930d89ebbe"
    )
    members, counts = decoder().load(CAPSULE)
    assert (
        counts["processes"],
        counts["endpoints"],
        counts["oracle_states"],
        counts["original_files"],
    ) == (11, 44, 8, 372)
    assert "frozen/runner/driver.py" in members
    assert "frozen/qualification-harness/run_oracle_full_density.py" in members
    assert "frozen/inputs/n-heptane.json" in members
    assert hashlib.sha256(members["review/REPORT.md"]).hexdigest() == (
        "bb5096c66847f7f2456af63a81a3fbc02bf5779f4b4ac225f2ba6710f99138f3"
    )


def test_larger_cpu_decoder_rejects_changed_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    (target / "README.md").write_text("changed")
    with pytest.raises(ValueError, match="byte identity mismatch"):
        decoder().load(target)


def test_larger_cpu_decoder_rejects_duplicate_member(tmp_path: Path) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    manifest = json.loads((target / "publication.json").read_text())
    manifest["files"].append(manifest["files"][0])
    (target / "publication.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate/nonlocal"):
        decoder().load(target)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "a/../b", "a//b", "a\\b", "x:y"]
)
def test_larger_cpu_decoder_rejects_unsafe_paths(name: str) -> None:
    with pytest.raises(ValueError, match="unsafe member path"):
        decoder().safe(name)


@pytest.mark.parametrize("field", ["expanded_member_count", "expanded_member_bytes"])
def test_larger_cpu_decoder_checks_declared_expansion(
    tmp_path: Path, field: str
) -> None:
    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    path = target / "publication.json"
    publication = json.loads(path.read_text())
    publication["bundle"][field] += 1
    path.write_text(json.dumps(publication))
    with pytest.raises(ValueError, match="expanded member count/bytes"):
        decoder().load(target)


@pytest.mark.parametrize(
    "name", ["validation.json", "validation.json/child", "bad\x00name"]
)
def test_larger_cpu_decoder_reserves_generated_output(
    tmp_path: Path, name: str
) -> None:
    import lzma

    target = tmp_path / "capsule"
    shutil.copytree(CAPSULE, target)
    publication = json.loads((target / "publication.json").read_text())
    payload = json.loads(
        lzma.decompress((target / "scientific-records.json.xz").read_bytes())
    )
    payload["members"].append(
        {
            "path": name,
            "text": "x",
            "bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    )
    raw = json.dumps(payload).encode()
    publication["bundle"]["bytes"] = len(raw)
    publication["bundle"]["sha256"] = hashlib.sha256(raw).hexdigest()
    (target / "scientific-records.json.xz").write_bytes(lzma.compress(raw))
    checks = [r["path"] for r in publication["files"] if r["path"] != "SHA256SUMS"]
    (target / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256((target / n).read_bytes()).hexdigest()}  {n}\n"
            for n in checks
        )
    )
    for row in publication["files"]:
        data = (target / row["path"]).read_bytes()
        row.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (target / "publication.json").write_text(json.dumps(publication))
    with pytest.raises(ValueError, match="reserved generated member|unsafe member"):
        decoder().load(target)


def test_larger_cpu_decoder_bounds_xz_dictionary_memory() -> None:
    import lzma
    import zlib

    # Alter only a tiny valid XZ block header to request a 4-GiB LZMA2
    # dictionary, then fix its CRC. No large compressor allocation is needed.
    raw = bytearray(lzma.compress(b"bounded decoder"))
    start = 12
    length = (raw[start] + 1) * 4
    assert raw[start + 2 : start + 4] == bytes([0x21, 1])
    raw[start + 4] = 40
    crc = zlib.crc32(raw[start : start + length - 4]).to_bytes(4, "little")
    raw[start + length - 4 : start + length] = crc
    with pytest.raises(lzma.LZMAError, match="Memory usage limit"):
        decoder().unzip(bytes(raw), xz=True)


def test_larger_cpu_decoder_rejects_trailing_xz_bytes() -> None:
    import lzma

    with pytest.raises(ValueError, match="trailing XZ"):
        decoder().unzip(lzma.compress(b"text") + b"trailing", xz=True)
