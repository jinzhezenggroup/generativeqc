"""Exercise raw retention and fail-closed restoration without scientific fixtures."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "force_pack", Path(__file__).with_name("pack.py")
)
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


class PackTests(unittest.TestCase):
    def test_collection_task_path_is_not_trusted_from_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = {
                "schema": "generativeqc.independent-cbs-force-collection.v1",
                "tasks": {"../escape": {"task": "../escape", "files": []}},
            }
            with self.assertRaises(ValueError):
                pack.restore_collection(bundle, root / "restored", {"../escape"})
            self.assertFalse((root / "restored").exists())
            self.assertFalse((root / "escape").exists())

    def test_collection_validates_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "outcome.json").write_bytes(b'{"exit_code":124}\n')
            task = {
                "task": "C4-h0-minus",
                "files": pack.inventory(source, pack.TASK_FILES, {}),
            }
            bundle = {
                "schema": "generativeqc.independent-cbs-force-collection.v1",
                "tasks": {"C4-h0-minus": task},
            }
            pack.restore_collection(bundle, root / "valid", {"C4-h0-minus"})
            self.assertEqual(
                (root / "valid/C4-h0-minus/outcome.json").read_bytes(),
                b'{"exit_code":124}\n',
            )
            bad = copy.deepcopy(bundle)
            bad["tasks"]["C4-h0-minus"]["files"][0]["path"] = "../escape"
            with self.assertRaises(ValueError):
                pack.restore_collection(bad, root / "invalid", {"C4-h0-minus"})
            self.assertFalse((root / "invalid").exists())
            with self.assertRaises(ValueError):
                pack.restore_collection(bundle, root / "unknown", {"C7-h0-minus"})
            self.assertFalse((root / "unknown").exists())

    def test_roundtrip_and_no_clobber(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            raw = b"failed scientific attempt\r\n-0.0\n"
            (source / "stdout.log").write_bytes(raw)
            files = pack.inventory(source, pack.TASK_FILES, {"stdout.log": "100644"})
            bundle = {"files": files}
            pack.restore(bundle, root / "restored", pack.TASK_FILES)
            self.assertEqual((root / "restored/stdout.log").read_bytes(), raw)
            with self.assertRaises(FileExistsError):
                pack.restore(bundle, root / "restored", pack.TASK_FILES)
            for mutation in ("content", "path", "mode", "duplicate"):
                bad = copy.deepcopy(bundle)
                if mutation == "content":
                    bad["files"][0]["utf8"] += "x"
                elif mutation == "path":
                    bad["files"][0]["path"] = "../escape"
                elif mutation == "mode":
                    bad["files"][0]["original_mode"] = "120000"
                else:
                    bad["files"].append(bad["files"][0])
                destination = root / mutation
                with self.assertRaises(ValueError):
                    pack.restore(bad, destination, pack.TASK_FILES)
                self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
