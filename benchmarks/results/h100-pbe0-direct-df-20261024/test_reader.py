"""Small storage/negative fixtures; no historical Git fetch or scientific calls."""

import copy
import json
import lzma
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import reader as r


class ReaderTests(unittest.TestCase):
    def local_pinned_repo(self, directory: Path) -> tuple[Path, dict, bytes]:
        repo = directory / "pinned"
        repo.mkdir()

        def git(*args: str, data: bytes | None = None) -> bytes:
            return subprocess.run(
                ["git", *args],
                cwd=repo,
                input=data,
                capture_output=True,
                timeout=10,
                check=True,
            ).stdout

        git("init", "--quiet")
        raw = b"exact existing offline blob\n"
        blob = git("hash-object", "-w", "--stdin", data=raw).decode().strip()
        tree = (
            git("mktree", data=f"100644 blob {blob}\tfile.txt\n".encode())
            .decode()
            .strip()
        )
        commit = (
            f"tree {tree}\nauthor OfflineFixture <fixture@example.invalid> 1 +0000\ncommitter OfflineFixture <fixture@example.invalid> 1 +0000\n\nsynthetic offline fixture\n"
        ).encode()
        revision = (
            git("hash-object", "-t", "commit", "-w", "--stdin", data=commit)
            .decode()
            .strip()
        )
        return (
            repo,
            {
                "revision": revision,
                "path": "file.txt",
                "mode": "100644",
                "blob": blob,
                "bytes": len(raw),
                "sha256": r.sha(raw),
            },
            raw,
        )

    def fixture(self, directory: Path) -> tuple[dict, dict[str, bytes]]:
        values = {
            "one.json": b'{"number":1.2345678901234567,"minus_zero":-0.0}\n',
            "code.py": b'raise RuntimeError("restoration never executes this")\n',
            "historic.txt": b"published immutable Git fixture\n",
        }
        pool = values["one.json"]
        zipped = lzma.compress(pool)
        (directory / "pool.xz").write_bytes(zipped)
        pool_entry = {
            "file": "pool.xz",
            "bytes": len(zipped),
            "sha256": r.sha(zipped),
            "raw_bytes": len(pool),
            "raw_sha256": r.sha(pool),
        }
        objects = {}
        for name, raw in values.items():
            entry = {"bytes": len(raw), "sha256": r.sha(raw)}
            if name == "one.json":
                entry.update(encoding="byte-pool", offset=0)
            elif name == "code.py":
                parts = []
                for i, chunk in enumerate((raw[:20], raw[20:])):
                    data = lzma.compress(chunk)
                    member = f"part{i}.xz"
                    (directory / member).write_bytes(data)
                    parts.append(
                        {
                            "file": member,
                            "bytes": len(data),
                            "sha256": r.sha(data),
                            "raw_bytes": len(chunk),
                            "raw_sha256": r.sha(chunk),
                        }
                    )
                entry.update(encoding="xz-parts", parts=parts)
            else:
                entry.update(
                    encoding="published-git",
                    revision="1" * 40,
                    path="historical/file.txt",
                    mode="100644",
                    blob="2" * 40,
                )
            objects[r.sha(raw)] = entry
        capsule = {
            "schema": "generativeqc.h100-pbe0-failed-pilot-originals.v1",
            "pool": pool_entry,
            "objects": objects,
            "files": [
                {
                    "path": name,
                    "sha256": r.sha(raw),
                    "bytes": len(raw),
                    "linux_mode": 0o644,
                    "role": "ORIGINAL",
                }
                for name, raw in values.items()
            ],
        }
        return capsule, values

    def recover(
        self, directory: Path, capsule: dict, values: dict[str, bytes]
    ) -> dict[str, bytes]:
        return r.recover(
            directory, capsule, resolver=lambda entry: values["historic.txt"]
        )

    def test_pool_parts_and_published_git_restore_exact_original_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            capsule, values = self.fixture(path)
            self.assertEqual(self.recover(path, capsule, values), values)

    def test_missing_or_corrupt_part_and_wrong_object_identity_fail(self) -> None:
        for mutation in ("missing", "content", "size", "digest"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                path = Path(d)
                capsule, values = self.fixture(path)
                if mutation == "missing":
                    (path / "part0.xz").unlink()
                if mutation == "content":
                    (path / "part0.xz").write_bytes(b"corrupt")
                if mutation == "size":
                    capsule["files"][0]["bytes"] += 1
                if mutation == "digest":
                    capsule["objects"][r.sha(values["one.json"])]["sha256"] = "0" * 64
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.recover(path, capsule, values)

    def test_bounded_decode_and_json_refuse_ambiguous_or_extra_data(self) -> None:
        data = lzma.compress(b"exact")
        self.assertEqual(r.decode(data, 5), b"exact")
        for bad, limit in (
            (data[:-2], 5),
            (data + b"tail", 5),
            (data + data, 10),
            (data, 4),
            (b"x" * ((1 << 20) + 1), 10),
        ):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                r.decode(bad, limit)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                r.read_json(raw)

    def test_unsafe_duplicate_modes_and_file_directory_collision_fail_before_restore(
        self,
    ) -> None:
        for mutation in ("unsafe", "duplicate", "mode", "collision"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as d:
                path = Path(d)
                capsule, values = self.fixture(path)
                if mutation == "unsafe":
                    capsule["files"][0]["path"] = "../escape"
                if mutation == "duplicate":
                    capsule["files"].append(copy.deepcopy(capsule["files"][0]))
                if mutation == "mode":
                    capsule["files"][0]["linux_mode"] = True
                if mutation == "collision":
                    capsule["files"][0]["path"] = "dir"
                    capsule["files"][1]["path"] = "dir/item"
                with self.assertRaises(ValueError):
                    self.recover(path, capsule, values)

    def test_inventory_only_and_unreferenced_objects_are_not_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            capsule, values = self.fixture(path)
            changed = copy.deepcopy(capsule)
            changed["files"][0]["role"] = "CACHE_DATA_INVENTORY_ONLY"
            with self.assertRaises(ValueError):
                self.recover(path, changed, values)
            capsule["files"] = capsule["files"][:-1]
            with self.assertRaises(ValueError):
                self.recover(path, capsule, values)

    def test_git_reference_never_fetches_or_accepts_unfixed_revision(self) -> None:
        entry = {
            "revision": "master",
            "path": "file",
            "mode": "100644",
            "blob": "2" * 40,
        }
        with patch("reader.subprocess.run") as command:
            with self.assertRaises(ValueError):
                r.git_blob(None, entry)
            with self.assertRaises(ValueError):
                r.git_blob(Path.cwd(), entry)
            command.assert_not_called()

    def test_existing_valid_pin_is_readable_with_all_transports_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, entry, raw = self.local_pinned_repo(Path(directory))
            with patch.dict(
                os.environ, {"GIT_ALLOW_PROTOCOL": "", "GIT_NO_LAZY_FETCH": "1"}
            ):
                self.assertEqual(r.git_blob(repo, entry), raw)

    def test_valid_pin_missing_promisor_blob_never_attempts_transport(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo, entry, _ = self.local_pinned_repo(root)
            marker = root / "transport-attempt.txt"
            sentinel = root / "local-transport-sentinel.py"
            sentinel.write_text(
                "import pathlib,sys\nwith pathlib.Path(sys.argv[1]).open('ab') as f:f.write(b'ATTEMPT\\n')\nraise SystemExit(97)\n",
                encoding="utf-8",
            )

            def ext_arg(value: str) -> str:
                return value.replace("%", "%%").replace(" ", "% ")

            endpoint = "ext::" + " ".join(
                ext_arg(str(value)) for value in (sys.executable, sentinel, marker)
            )
            for key, value in (
                ("core.repositoryformatversion", "1"),
                ("extensions.partialClone", "origin"),
                ("remote.origin.promisor", "true"),
                ("remote.origin.partialclonefilter", "blob:none"),
                ("remote.origin.url", endpoint),
                ("protocol.ext.allow", "always"),
            ):
                subprocess.run(
                    ["git", "config", "--local", key, value],
                    cwd=repo,
                    capture_output=True,
                    timeout=5,
                    check=True,
                )
            obj = repo / ".git" / "objects" / entry["blob"][:2] / entry["blob"][2:]
            obj.resolve().relative_to(repo.resolve())
            obj.chmod(0o600)
            obj.unlink()
            # Only a local sentinel helper could run. It has no networking code
            # and never contacts or supplies data from an external repository.
            with (
                patch.dict(
                    os.environ,
                    {"GIT_ALLOW_PROTOCOL": "ext", "GIT_NO_LAZY_FETCH": "0"},
                ),
                self.assertRaises(ValueError),
            ):
                r.git_blob(repo, entry)
            self.assertFalse(
                marker.exists(),
                "valid missing-object read attempted a transport helper",
            )

    def test_unsupported_git_option_fails_closed_without_unprotected_fallback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo, entry, _ = self.local_pinned_repo(Path(directory))
            failure = subprocess.CalledProcessError(
                129, ["git"], stderr=b"unknown option: --no-lazy-fetch"
            )
            with patch("reader.subprocess.run", side_effect=failure) as command:
                with self.assertRaisesRegex(
                    ValueError, "no-lazy-fetch support required"
                ):
                    r.git_blob(repo, entry)
                self.assertTrue(command.call_args_list)
                for invocation in command.call_args_list:
                    self.assertIn("--no-lazy-fetch", invocation.args[0])
                    self.assertEqual(invocation.kwargs["env"]["GIT_ALLOW_PROTOCOL"], "")
                    self.assertEqual(invocation.kwargs["env"]["GIT_NO_LAZY_FETCH"], "1")
                    self.assertEqual(invocation.kwargs["timeout"], 15)

    def test_restoration_does_not_execute_code_or_overwrite_existing_data(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            capsule, values = self.fixture(path)
            output = path / "restored"
            r.restore(output, values, capsule["files"])
            self.assertEqual((output / "code.py").read_bytes(), values["code.py"])
            with self.assertRaises(ValueError):
                r.restore(output, values, capsule["files"])
            destination = path / "never-created"
            with self.assertRaises(ValueError):
                r.restore(destination, {"a": b"1", "a-b": b"2", "a/x": b"3"}, [])
            self.assertFalse(destination.exists())

    def test_unbound_case_cannot_claim_zero_science_or_success(self) -> None:
        with self.assertRaises(ValueError):
            r.replay(
                {
                    "audit/root-terminal-failure-audit.json": json.dumps(
                        {"status": "PASS", "actual_scientific_request_markers": 0}
                    ).encode()
                },
                {"files": []},
            )


if __name__ == "__main__":
    unittest.main()
