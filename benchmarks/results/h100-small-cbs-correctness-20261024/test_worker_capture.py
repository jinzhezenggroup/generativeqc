"""Synthetic actual-worker control-path tests; no native library is loaded."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import native_worker
from replay import recover
from transport import can_continue_scientific_failure, read, retain_return


class WorkerReturnCapture(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "python/generativeqc").mkdir(parents=True)
        library = self.source / "synthetic-library"
        library.write_bytes(b"SYNTHETIC; never loaded")
        self.output = self.root / "output"
        self.output.mkdir()
        self.process = "conventional-h2-dz-tz"
        reference = {
            "low_basis": "low.json",
            "high_basis": "high.json",
            "basis_identities": ["low", "high"],
            "atoms": [[1, [0.0, 0.0, -0.4]], [1, [0.0, 0.0, 0.4]]],
        }
        text = json.dumps(reference)
        bundle = {
            "conventional": {
                "tests/reference_data/cbs/h2-dz-tz.json": {
                    "content_utf8": text,
                    "sha256": hashlib.sha256(text.encode()).hexdigest(),
                }
            }
        }
        path = self.root / "inputs.json"
        path.write_text(json.dumps(bundle))
        self.plan = {
            "processes": [{"id": self.process, "calls": 1}],
            "calls": [
                {
                    "index": 0,
                    "process": self.process,
                    "fixture": "h2-dz-tz",
                    "kind": "EF",
                    "step_bohr": None,
                    "policy": "standard",
                    "numeric_budget_bytes": 34359738368,
                }
            ],
            "output_root": str(self.output),
            "caps": {"process_log_max_bytes": 65536},
            "science_source": {
                "root": str(self.source),
                "library": str(library),
                "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            },
            "input_bundle_file": path.name,
            "input_bundle_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "runtime_controls": {},
        }
        self.requests = []
        self.metadata_saw_ready_originals = []

    def invoke(
        self,
        *,
        metadata_error: bool = False,
        io_error: bool = False,
        native_error: bool = False,
        missing_forces: bool = False,
    ) -> None:
        requests = self.requests
        checks = self.metadata_saw_ready_originals
        observable_path = self.output / self.process / "call-000/observables"
        components = []
        for i in range(5):
            result = SimpleNamespace(
                energy=-1.0 - i,
                forces=[[0.0, 0.0, 0.0], [0.0, 0.0, -0.0]],
                converged=True,
                iterations=1,
                executed_backend="cuda",
                correlation=None,
                cc_performance=None,
            )
            components.append(
                SimpleNamespace(
                    result=result,
                    roles=["SYNTHETIC"],
                    method="SYNTHETIC",
                    basis_identity="SYNTHETIC",
                    coefficient=1.0,
                    elapsed_seconds=1.0,
                )
            )

        class Returned:
            energy = -1.0
            converged = True
            executed_backend = "cuda"

            def __init__(self) -> None:
                self.forces = [[0.0, 0.0, 0.0], [0.0, 0.0, -0.0]]

            def __getattr__(self, field: str) -> object:
                if field == "hf_energy":
                    ready = all(
                        (observable_path / name).is_file()
                        for name in ("values.bin", "values.index.json", "return.json")
                    )
                    checks.append(ready)
                    if ready:
                        recover(observable_path)
                if field == "basis_metadata":
                    if metadata_error:
                        raise RuntimeError("SYNTHETIC post-return metadata failure")
                    return {}
                return None

        returned = Returned()
        if missing_forces:
            returned.forces = None
        returned.components = tuple(components)

        class Calculator:
            def __init__(self, *a: object, **k: object) -> None:
                pass

            def singlepoint(self, *a: object, **k: object) -> Returned:
                requests.append(True)
                if native_error:
                    raise ValueError("SYNTHETIC native refusal")
                return returned

        native = SimpleNamespace(
            __file__=str(self.source / "python/generativeqc/__init__.py"),
            CBSBasis=lambda basis, n: SimpleNamespace(basis=basis),
            load_basis=lambda p: SimpleNamespace(identity=p.stem),
            CCSDTCBSCalculator=Calculator,
        )

        def persist(path: Path, sample: dict) -> dict:
            if io_error and path.name == "enriched":
                raise OSError("SYNTHETIC enrichment IO failure")
            return retain_return(path, sample)

        original = sys.path[:]
        try:
            with (
                patch.object(native_worker, "release", return_value=(self.plan, {})),
                patch.object(native_worker, "device", return_value={}),
                patch.object(
                    native_worker, "os", SimpleNamespace(name="posix", environ={})
                ),
                patch.dict(
                    sys.modules,
                    {
                        "resource": SimpleNamespace(
                            RLIMIT_FSIZE=1, setrlimit=lambda *a: None
                        )
                    },
                ),
                patch.object(
                    native_worker.importlib, "import_module", return_value=native
                ),
                patch.object(native_worker, "retain_return", side_effect=persist),
            ):
                native_worker.run(
                    self.root / "plan.json", self.root / "release.json", self.process
                )
        finally:
            sys.path[:] = original

    def test_metadata_failure_preserves_returned_EF_and_classifies_transport(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "SYNTHETIC post-return metadata"):
            self.invoke(metadata_error=True)
        outcome = read(self.output / self.process / "worker-outcome.json")
        self.assertEqual(self.requests, [True])
        self.assertEqual(outcome["calls_returned"], 1)
        self.assertEqual(self.metadata_saw_ready_originals, [True])
        self.assertEqual(outcome["failure_kind"], "TRANSPORT")
        self.assertFalse(
            can_continue_scientific_failure(
                {"status": "FAILED", "returncode": 1}, outcome
            )
        )
        core = recover(self.output / self.process / "call-000/observables")
        self.assertEqual(core["energy"], -1.0)
        self.assertEqual(len(core["components"]), 5)
        self.assertEqual(core["components"][4]["energy"], -5.0)
        self.assertGreater(core["complete_seconds"], 0.0)

    def test_unavailable_returned_force_is_not_zero_filled(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SYNTHETIC post-return metadata"):
            self.invoke(metadata_error=True, missing_forces=True)
        core = recover(self.output / self.process / "call-000/observables")
        self.assertIsNone(core["forces"])
        self.assertEqual(core["observable_availability"]["forces"], "RETURNED_NONE")
        self.assertEqual(self.requests, [True])

    def test_enrichment_IO_failure_keeps_minimal_return_and_stops_transport(
        self,
    ) -> None:
        with self.assertRaisesRegex(OSError, "SYNTHETIC enrichment IO"):
            self.invoke(io_error=True)
        outcome = read(self.output / self.process / "worker-outcome.json")
        self.assertEqual(self.requests, [True])
        self.assertEqual(outcome["calls_returned"], 1)
        self.assertEqual(outcome["failure_kind"], "TRANSPORT")
        self.assertFalse(
            can_continue_scientific_failure(
                {"status": "FAILED", "returncode": 1}, outcome
            )
        )
        self.assertEqual(
            recover(self.output / self.process / "call-000/observables")["energy"], -1.0
        )

    def test_native_refusal_consumes_one_slot_without_inventing_return(self) -> None:
        with self.assertRaisesRegex(ValueError, "SYNTHETIC native refusal"):
            self.invoke(native_error=True)
        outcome = read(self.output / self.process / "worker-outcome.json")
        self.assertEqual(self.requests, [True])
        self.assertEqual(outcome["calls_attempted"], 1)
        self.assertEqual(outcome["calls_returned"], 0)
        self.assertEqual(outcome["failure_kind"], "SCIENTIFIC_REQUEST")
        self.assertTrue(
            can_continue_scientific_failure(
                {"status": "FAILED", "returncode": 1}, outcome
            )
        )
        self.assertFalse(list(self.output.rglob("values.bin")))


if __name__ == "__main__":
    unittest.main()
