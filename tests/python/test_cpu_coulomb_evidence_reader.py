"""The standalone Coulomb evidence reader preserves both compressed formats."""

import gzip
import hashlib
import importlib.util
import json
import lzma
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

BUNDLE = (
    Path(__file__).resolve().parents[2] / "benchmarks/results/cpu-eri-coulomb-20261001"
)


def reader() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "cpu_coulomb_evidence_reader", BUNDLE / "verify.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capsule(path: str = "campaign/result.json") -> tuple[bytes, bytes]:
    content = {"failed": False, "samples": [1.125, -0.0, 2.5e-14]}
    raw = (json.dumps(content, indent=2, allow_nan=False) + "\n").encode()
    encoded = json.dumps(
        {
            "schema": "generativeqc.cpu-eri-evidence-capsule.v1",
            "files": [
                {
                    "path": path,
                    "encoding": "json-indent2-newline",
                    "content": content,
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            ],
        }
    ).encode()
    return encoded, raw


@pytest.mark.parametrize(
    "suffix,compress", [("gz", gzip.compress), ("xz", lzma.compress)]
)
def test_reader_recovers_exact_json_and_text_bytes(
    tmp_path: Path, suffix: str, compress: Callable[[bytes], bytes]
) -> None:
    encoded, expected = capsule()
    payload = json.loads(encoded)
    text = "Failed pilot\nUnicode: α\n".encode()
    payload["files"].append(
        {
            "path": "campaign/failed.txt",
            "encoding": "utf8",
            "content": text.decode(),
            "bytes": len(text),
            "sha256": hashlib.sha256(text).hexdigest(),
        }
    )
    (tmp_path / f"records.json.{suffix}").write_bytes(
        compress(json.dumps(payload).encode())
    )
    assert dict(reader().members(tmp_path)) == {
        "campaign/result.json": expected,
        "campaign/failed.txt": text,
    }


def test_reader_combines_gzip_and_xz_capsules(tmp_path: Path) -> None:
    expected = {}
    for suffix, compress in [("gz", gzip.compress), ("xz", lzma.compress)]:
        name = f"campaign/{suffix}.json"
        encoded, raw = capsule(name)
        expected[name] = raw
        (tmp_path / f"records.json.{suffix}").write_bytes(compress(encoded))
    assert dict(reader().members(tmp_path)) == expected


@pytest.mark.parametrize(
    "suffix,compress", [("gz", gzip.compress), ("xz", lzma.compress)]
)
@pytest.mark.parametrize("failure", ["record", "path", "schema"])
def test_reader_rejects_corrupt_or_unsafe_records(
    tmp_path: Path, suffix: str, compress: Callable[[bytes], bytes], failure: str
) -> None:
    encoded, _ = capsule()
    payload = json.loads(encoded)
    if failure == "record":
        payload["files"][0]["content"]["failed"] = True
        message = "Changed original record"
    elif failure == "path":
        payload["files"][0]["path"] = "../escape.json"
        message = "Unsafe retained path"
    else:
        payload["schema"] = "unknown"
        message = "Unknown capsule schema"
    (tmp_path / f"records.json.{suffix}").write_bytes(
        compress(json.dumps(payload).encode())
    )
    with pytest.raises(ValueError, match=message):
        dict(reader().members(tmp_path))


@pytest.mark.parametrize(
    "suffix,compress", [("gz", gzip.compress), ("xz", lzma.compress)]
)
@pytest.mark.parametrize("changed_identity", ["decoded_bytes", "decoded_sha256"])
def test_verifier_rejects_changed_decoded_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    compress: Callable[[bytes], bytes],
    changed_identity: str,
) -> None:
    module = reader()
    encoded, _ = capsule()
    stored = compress(encoded)
    name = f"records.json.{suffix}"
    (tmp_path / name).write_bytes(stored)
    identity = {
        "path": name,
        "bytes": len(stored),
        "sha256": hashlib.sha256(stored).hexdigest(),
        "decoded_bytes": len(encoded),
        "decoded_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    identity[changed_identity] = 0 if changed_identity == "decoded_bytes" else "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({"files": [identity]}))
    monkeypatch.setattr(module, "HERE", tmp_path)
    monkeypatch.setattr("sys.argv", ["verify.py"])
    with pytest.raises(ValueError, match="Changed decoded member"):
        module.main()


def test_lossless_gzip_to_xz_preserves_capsule_and_failed_records(
    tmp_path: Path,
) -> None:
    encoded, raw = capsule()
    payload = json.loads(encoded)
    failure = "Initial oracle pilot failed\nUnicode: α\n".encode()
    payload["files"].append(
        {
            "path": "campaign/failed.txt",
            "encoding": "utf8",
            "content": failure.decode(),
            "bytes": len(failure),
            "sha256": hashlib.sha256(failure).hexdigest(),
        }
    )
    original = (json.dumps(payload, indent=2) + "\n").encode()
    gz = tmp_path / "records.json.gz"
    gz.write_bytes(gzip.compress(original, mtime=0))
    before = list(reader().members(tmp_path))
    decoded = gzip.decompress(gz.read_bytes())
    stored = lzma.compress(decoded, preset=9 | lzma.PRESET_EXTREME)
    assert stored == lzma.compress(decoded, preset=9 | lzma.PRESET_EXTREME)
    assert lzma.decompress(stored) == original
    gz.unlink()
    (tmp_path / "records.json.xz").write_bytes(stored)
    assert list(reader().members(tmp_path)) == before
    assert dict(before) == {"campaign/result.json": raw, "campaign/failed.txt": failure}
