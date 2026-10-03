"""Stdlib-only integrity and scope checks for immutable scalar HF evidence.

All corruption probes use disposable copies. Decoding never runs the retained
scientific replay; no historical hash is compared to forever-current source.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import shutil
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[2]
CAPSULE = ROOT / "benchmarks/results/cpu-bounded-scalar-hf-20261003"
REPLAY_SHA = "7053d9284a92d16d51871180552683d5531731a76b8d3545213f11988979334d"
CAPSULE_SHA = "e81f30ac3379e832bf9f280e0c3245e8012327f99f7566d04d4609232c11fc0d"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_member(
    root: Path, manifest: dict, name: str, stored: bytes, decoded: bytes | None = None
) -> None:
    """Rebind a synthetic mutation so the decoder reaches the intended guard."""
    (root / name).write_bytes(stored)
    row = next(row for row in manifest["files"] if row["path"] == name)
    row.update(bytes=len(stored), sha256=sha(stored))
    if decoded is not None:
        row.update(decoded_bytes=len(decoded), decoded_sha256=sha(decoded))


class TestScalarHFPublication(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="scalar-hf-publication-")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.counter = 0

    def decode(
        self,
        root: Path = CAPSULE,
        output: Path | None = None,
        *,
        optimized: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        args = [sys.executable, "-I", "-S", "-B"]
        if optimized:
            args.append("-O")
        args.append(str(root / "decode.py"))
        if output is not None:
            args.extend(("--output-dir", str(output)))
        return subprocess.run(
            args, capture_output=True, text=True, timeout=30, check=False
        )

    def reject(
        self,
        mutation: Callable[[Path, dict], Any],
        message: str,
        *,
        optimized: bool = False,
    ) -> None:
        self.counter += 1
        target = self.root / f"case-{self.counter}"
        shutil.copytree(CAPSULE, target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        mutation(target, manifest)
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        output = self.root / f"expanded-{self.counter}"
        result = self.decode(target, output, optimized=optimized)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(
            output.exists(), "invalid input must fail before output creation"
        )

    def test_exact_expansion_roundtrip_and_verify_only(self) -> None:
        verified = self.decode()
        self.assertEqual(verified.returncode, 0, verified.stdout + verified.stderr)
        self.assertEqual(list(self.root.iterdir()), [])
        output = self.root / "expanded"
        result = self.decode(output=output)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            {p.name for p in output.iterdir()}, {"capsule.json", "fresh_endpoint.py"}
        )
        manifest = json.loads((CAPSULE / "manifest.json").read_bytes())
        for row in manifest["files"]:
            stored = (CAPSULE / row["path"]).read_bytes()
            self.assertEqual((len(stored), sha(stored)), (row["bytes"], row["sha256"]))
            if row["path"].endswith(".xz"):
                decoded = (output / row["path"][:-3]).read_bytes()
                self.assertEqual(decoded, lzma.decompress(stored))
                self.assertEqual(
                    (len(decoded), sha(decoded)),
                    (row["decoded_bytes"], row["decoded_sha256"]),
                )
        self.assertEqual(sha((output / "fresh_endpoint.py").read_bytes()), REPLAY_SHA)
        data = (output / "capsule.json").read_bytes()
        self.assertEqual(sha(data), CAPSULE_SHA)
        self.assertEqual(
            json.loads(data)["schema"], "bounded-scalar-selective-evidence-v2"
        )

    def test_standard_publication_and_validation_envelope(self) -> None:
        # Repository schema validators are stdlib-only. Isolate them from modules
        # that an enclosing test session may already have imported.
        script = """
import json, sys
from pathlib import Path
root, capsule = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root / 'python')]
from tools.generativeqc_validation.publication import validate_publication
manifest = json.loads((capsule / 'publication.json').read_bytes())
files = {row['path']: (capsule / row['path']).read_bytes() for row in manifest['files']}
validate_publication(manifest, files)
assert not {'numpy', 'scipy', 'pyscf', 'generativeqc'} & sys.modules.keys()
for name in files:
    corrupt = dict(files)
    corrupt[name] += b'changed'
    try:
        validate_publication(manifest, corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted corrupt publication member: ' + name)
for scope in ('numerical', 'performance'):
    promoted = dict(manifest, decision={'status': 'accepted', 'scope': scope, 'reason': 'injected'})
    try:
        validate_publication(promoted, files)
    except ValueError:
        pass
    else:
        raise AssertionError('accepted unsupported promotion: ' + scope)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", script, str(ROOT), str(CAPSULE)],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        publication = json.loads((CAPSULE / "publication.json").read_bytes())
        envelope = json.loads(
            gzip.decompress((CAPSULE / "validation.json.gz").read_bytes())
        )
        manifest = json.loads((CAPSULE / "manifest.json").read_bytes())
        self.assertEqual(publication["decision"]["status"], "inconclusive")
        self.assertEqual(publication["decision"]["scope"], "performance")
        self.assertEqual(publication["archives"], [])
        self.assertEqual(envelope["revision"], publication["source"]["revision"])
        self.assertEqual(envelope["stages"]["numerical"]["status"], "fail")
        self.assertEqual(envelope["performance"]["status"], "not-run")
        for key in ("performance_accepted", "default_acceptance_established"):
            self.assertIs(manifest[key], False)
        failed = {
            key: row
            for key, row in envelope["block_errors"].items()
            if not row["passed"]
        }
        self.assertEqual(
            set(failed),
            {
                "retained_OH_raw_density_failures",
                "retained_standard_UHF24_torque_failures",
            },
        )
        self.assertEqual(sum(row["shape"][0] for row in failed.values()), 6)
        self.assertIsNone(envelope["memory"]["allocated_bytes"])
        self.assertIsNone(envelope["memory"]["peak_bytes"])

    def test_decoder_never_executes_recovered_code(self) -> None:
        target = self.root / "synthetic"
        shutil.copytree(CAPSULE, target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        marker = self.root / "executed"
        raw = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('executed')\n".encode()
        replace_member(
            target, manifest, "fresh_endpoint.py.xz", lzma.compress(raw), raw
        )
        (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        result = self.decode(target, self.root / "expanded")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(marker.exists())
        self.assertEqual((self.root / "expanded/fresh_endpoint.py").read_bytes(), raw)

    def test_rejects_existing_and_dangling_output(self) -> None:
        existing = self.root / "existing"
        existing.mkdir()
        marker = existing / "keep"
        marker.write_bytes(b"unchanged")
        dangling = self.root / "dangling"
        dangling.symlink_to(self.root / "missing", target_is_directory=True)
        for output in (existing, dangling):
            with self.subTest(output=output.name):
                result = self.decode(output=output)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("FileExistsError", result.stderr)
        self.assertEqual(marker.read_bytes(), b"unchanged")
        self.assertFalse((self.root / "missing").exists())

    def test_rejects_unsafe_duplicate_and_missing_members(self) -> None:
        for name in (
            "../escape",
            "/absolute",
            "a/../b",
            "a//b",
            "a\\b",
            "x:y",
            "unknown",
        ):
            with self.subTest(name=name):
                self.reject(
                    lambda p, m, name=name: m["files"][0].update(path=name),
                    "unsafe/duplicate path",
                )
        self.reject(
            lambda p, m: m["files"].append(m["files"][0]), "unsafe/duplicate path"
        )
        self.reject(lambda p, m: m["files"].pop(), "incomplete inventory")

    def test_rejects_stored_and_decoded_identity_changes(self) -> None:
        for key, value, message in (
            ("bytes", 0, "stored bytes/hash"),
            ("sha256", "0" * 64, "stored bytes/hash"),
            ("decoded_bytes", 0, "decoded bytes/hash"),
            ("decoded_sha256", "0" * 64, "decoded bytes/hash"),
        ):
            with self.subTest(field=key):
                self.reject(
                    lambda p, m, key=key, value=value: next(
                        r for r in m["files"] if r["path"] == "capsule.json.xz"
                    ).update({key: value}),
                    message,
                )
        self.reject(
            lambda p, m: (p / "README.md").write_bytes(b"corrupt"), "stored bytes/hash"
        )

    def test_rejects_truncated_trailing_and_multistream_xz(self) -> None:
        original = (CAPSULE / "capsule.json.xz").read_bytes()
        for label, stored in (
            ("truncated", original[:-1]),
            ("trailing", original + b"trailing"),
            ("multistream", original + lzma.compress(b"{}")),
        ):
            with self.subTest(label=label):
                self.reject(
                    lambda p, m, stored=stored: replace_member(
                        p, m, "capsule.json.xz", stored
                    ),
                    "XZ output/EOF/trailing or multistream",
                )

    def test_rejects_both_decoded_output_caps(self) -> None:
        for name, cap in (
            ("capsule.json.xz", 2 << 20),
            ("fresh_endpoint.py.xz", 64 << 10),
        ):
            with self.subTest(name=name):
                raw = b" " * (cap + 1)
                stored = lzma.compress(raw, preset=0)
                self.reject(
                    lambda p, m, name=name, stored=stored, raw=raw: replace_member(
                        p, m, name, stored, raw
                    ),
                    "XZ output/EOF/trailing or multistream",
                )

    def test_rejects_stored_and_manifest_size_caps(self) -> None:
        self.reject(
            lambda p, m: replace_member(
                p, m, "capsule.json.xz", b"x" * ((1 << 20) + 1)
            ),
            "stored bytes/hash",
        )
        self.reject(lambda p, m: m.update(padding="x" * 16385), "manifest too large")
        self.reject(
            lambda p, m: replace_member(p, m, "README.md", b"x" * ((32 << 10) + 1)),
            "text too large",
        )

    def test_bounds_xz_dictionary_memory_without_allocating_it(self) -> None:
        stored = bytearray(lzma.compress(b"{}"))
        start = 12
        size = (stored[start] + 1) * 4
        self.assertEqual(stored[start + 2 : start + 4], bytes([0x21, 1]))
        stored[start + 4] = 40  # Advertise 4-GiB dictionary, never allocate it here.
        stored[start + size - 4 : start + size] = zlib.crc32(
            stored[start : start + size - 4]
        ).to_bytes(4, "little")
        self.reject(
            lambda p, m: replace_member(p, m, "capsule.json.xz", bytes(stored)),
            "Memory usage limit",
        )

    def test_rejects_invalid_decoded_json(self) -> None:
        raw = b"not json"
        self.reject(
            lambda p, m: replace_member(
                p, m, "capsule.json.xz", lzma.compress(raw), raw
            ),
            "JSONDecodeError",
        )

    def test_guards_survive_optimized_python(self) -> None:
        result = self.decode(optimized=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.reject(
            lambda p, m: m["files"][0].update(path="../escape"),
            "unsafe/duplicate path",
            optimized=True,
        )


if __name__ == "__main__":
    unittest.main()
