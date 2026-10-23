"""Executable isolation and fail-closed contracts for the #2680 driver."""

from __future__ import annotations

import copy
import json
import lzma
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks import runtime_cache_state as driver


class CacheStateTests(unittest.TestCase):
    def test_lossless_capsule_retains_config_order_signed_zero_and_text(self) -> None:
        codec = runpy.run_path(
            str(
                driver.ROOT
                / "benchmarks/results/runtime-cache-state-20261023/capsule.py"
            )
        )
        config = {"second": 2, "first": 1, "independent_reference": {"energy": -1.25}}
        value = {
            "before": "α\r\n",
            "config": {"first": 1, "second": 2},
            "signed_zero": -0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "request.json"
            raw = codec["pretty"](value)
            path.write_bytes(raw)
            stored = codec["encode_member"](path, "process/request.json", config)
            self.assertEqual(codec["decode_member"](stored, config), raw)
            crlf = raw.replace(b"\n", b"\r\n")
            path.write_bytes(crlf)
            crlf_member = codec["encode_member"](path, "process/request.json", config)
            self.assertEqual(codec["decode_member"](crlf_member, config), crlf)
            stored["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "raw recovery"):
                codec["decode_member"](stored, config)

    def test_capsule_rejects_host_independent_path_escapes(self) -> None:
        codec = runpy.run_path(
            str(
                driver.ROOT
                / "benchmarks/results/runtime-cache-state-20261023/capsule.py"
            )
        )
        for name in (
            "../escape",
            "C:\\escape",
            "a\\..\\escape",
            "/absolute",
            ".",
            "a//b",
        ):
            with self.subTest(path=name), self.assertRaises(ValueError):
                codec["relative_name"](name)

    def test_cache_gate_rejects_missing_stats_and_shared_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary)
            folder = campaign / "process-00"
            folder.mkdir()
            row = {"state": "empty", "cache": "water-12-pbe0/empty-0"}
            owned = campaign / "caches" / row["cache"]
            driver.new_cache(owned)
            recorded_output = "/original-run/formal"
            recorded_cache = recorded_output + "/caches/" + row["cache"]
            driver.save(folder / "cache-before.json", driver.census(owned))
            driver.save(
                folder / "request.json",
                {
                    "cache_before_sha256": driver.digest(folder / "cache-before.json"),
                    "output_root": recorded_output,
                    "cache_root": recorded_cache,
                },
            )
            paths = {
                name: recorded_cache + "/" + name
                for name in (
                    "SCCACHE_DIR",
                    "SCCACHE_CONF",
                    "SCCACHE_CACHED_CONF",
                    "CCACHE_DIR",
                    "CCACHE_CONFIGPATH",
                )
            }
            event = {
                "kind": "compiler-process",
                "phase": "compile",
                "command": ["/tools/sccache", "nvcc", "-c", "unit.cu"],
                "cache_paths": paths,
                "stats_before": None,
                "stats_after": None,
            }
            record = {
                "request_sha256": driver.digest(folder / "request.json"),
                "observation": {"events": [event]},
            }
            self.assertEqual(
                driver.cache_gate(folder, row, record)["status"], "INCOMPLETE"
            )
            event.update(stats_before={"returncode": 0}, stats_after={"returncode": 0})
            self.assertEqual(
                driver.cache_gate(folder, row, record)["status"], "INCOMPLETE"
            )
            stats = {
                "returncode": 0,
                "stdout": json.dumps(
                    {
                        "stats": {
                            "compile_requests": 1,
                            "cache_hits": {"counts": {}},
                            "cache_misses": {"counts": {}},
                        },
                        "cache_location": f'Local disk: "{recorded_cache}"',
                    }
                ),
            }
            event.update(stats_before=stats, stats_after=stats)
            self.assertEqual(driver.cache_gate(folder, row, record)["status"], "PASS")
            paths["SCCACHE_DIR"] = "/other-owner/cache"
            with self.assertRaisesRegex(ValueError, "private root"):
                driver.cache_gate(folder, row, record)

    def test_formal_run_cannot_create_output_without_pilot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "formal"
            with self.assertRaisesRegex(ValueError, "pilot"):
                driver.run(
                    Path(temporary) / "config.json", output, pilot=False, timeout=1
                )
            self.assertFalse(output.exists())

    def test_artifact_observer_preserves_shared_library_result(self) -> None:
        from generativeqc_compiler.common.cuda_runtime import CudaArtifact

        artifact = CudaArtifact(Path("cached.so"), {"compile_seconds": 123.0})
        observer = driver.Observer()
        observed = observer.wrap(lambda: artifact, "artifact")()
        self.assertIs(observed, artifact)
        self.assertEqual(observer.receipt()["cached_driver_wall_seconds"], 0)
        self.assertEqual(
            observer.events[0]["artifact_metadata"]["compile_seconds"], 123.0
        )

    def test_observer_reraises_original_failure(self) -> None:
        failure = RuntimeError("compiler rejected input")

        def fail(command: list[str]) -> None:
            raise failure

        observer = driver.Observer()
        with self.assertRaises(RuntimeError) as caught:
            observer.wrap(fail, "cached-driver")(["nvcc", "input.cu", "-c"])
        self.assertIs(caught.exception, failure)
        self.assertEqual(observer.events[0]["error_type"], "RuntimeError")

    def test_retained_water_oracle_gate_rejects_changed_observables(self) -> None:
        root = driver.ROOT / "benchmarks/results/readme-final-20261022"
        references = json.loads(
            lzma.decompress((root / "oracles.json.xz").read_bytes())
        )
        native = json.loads(lzma.decompress((root / "standard.json.xz").read_bytes()))
        original = next(
            r
            for r in native
            if r["method"] == "pbe0" and r["mode"] == "direct" and r["atoms"] == 12
        )
        config = {
            "water_references": references,
            "reference_reuse": json.loads((root / "reference-reuse.json").read_text()),
        }
        record = {
            "environment": original["environment"],
            "endpoint": {
                "method": "pbe0",
                "atoms": 12,
                "geometry_bohr": original["geometry_bohr"],
                "sample": copy.deepcopy(original["samples"][0]),
            },
        }
        self.assertEqual(driver.water_gate(record, config)["status"], "PASS")
        record["endpoint"]["sample"]["energy"] += 1
        self.assertEqual(driver.water_gate(record, config)["status"], "FAIL")
        config["water_references"] = [r for r in references if r["backend"] != "cpu"]
        self.assertEqual(driver.water_gate(record, config)["status"], "INCOMPLETE")

    def test_cbs_gate_retains_original_independent_directional_limits(self) -> None:
        path = (
            driver.ROOT / "benchmarks/results/cbs-fast-lambda-20261021/summary.json.xz"
        )
        point = json.loads(lzma.decompress(path.read_bytes()))["points"][1]
        record = {
            "endpoint": {
                "protocol": point["native"]["protocol"],
                "sample": copy.deepcopy(point["native"]["samples"][0]),
            }
        }
        config = {
            "cbs_reference": point["reference"],
            "cbs_reference_sha256": driver.identity(point["reference"]),
        }
        self.assertEqual(driver.cbs_gate(record, config)["status"], "PASS")
        record["endpoint"]["sample"]["energy"] += 1e-5
        self.assertEqual(driver.cbs_gate(record, config)["status"], "FAIL")
        record["endpoint"]["protocol"] = {
            **record["endpoint"]["protocol"],
            "frozen_core": True,
        }
        with self.assertRaisesRegex(ValueError, "different scientific input"):
            driver.cbs_gate(record, config)
        config = copy.deepcopy(config)
        config["cbs_reference"]["direction"][0][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "seed 709"):
            driver.cbs_gate(record, config)

    def test_direction_identity_rejects_seed_sign_scale_shape_and_unknown_fields(
        self,
    ) -> None:
        import numpy as np

        path = (
            driver.ROOT / "benchmarks/results/cbs-fast-lambda-20261021/summary.json.xz"
        )
        original = json.loads(lzma.decompress(path.read_bytes()))["points"][1][
            "reference"
        ]
        mutations = []
        for scale in (-1, 1.000001, 0):
            reference = copy.deepcopy(original)
            reference["direction"] = (
                np.asarray(reference["direction"]) * scale
            ).tolist()
            mutations.append(reference)
        reference = copy.deepcopy(original)
        other = np.random.default_rng(710).normal(size=(8, 3))
        reference["direction"] = (other / np.linalg.norm(other)).tolist()
        mutations.append(reference)
        reference = copy.deepcopy(original)
        reference["direction"] = reference["direction"][:-1]
        mutations.append(reference)
        reference = copy.deepcopy(original)
        reference["unreviewed_direction_model"] = True
        mutations.append(reference)
        for index, reference in enumerate(mutations):
            with self.subTest(mutation=index), self.assertRaises(ValueError):
                driver.cbs_direction(reference)

    def test_schedule_is_thirty_fresh_measurements_plus_five_primes(self) -> None:
        rows = driver.schedule()
        self.assertEqual(len(rows), 35)
        self.assertEqual(sum(r["state"] == "prime" for r in rows), 5)
        measured = [r for r in rows if r["state"] != "prime"]
        self.assertEqual(
            len({(r["case"], r["state"], r["replicate"]) for r in measured}), 30
        )
        empty = [r["cache"] for r in measured if r["state"] == "empty"]
        self.assertEqual(len(set(empty)), 15)
        for case in driver.CASES:
            self.assertEqual(
                len(
                    {
                        r["cache"]
                        for r in rows
                        if r["case"] == case and r["state"] != "empty"
                    }
                ),
                1,
            )

    def test_environment_does_not_inherit_secrets_or_remote_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"library": "/frozen/lib.so", "python_paths": ["/frozen/python"]}
            env = driver.cache_environment(
                root,
                config,
                {
                    "PATH": "/tools",
                    "CUDA_VISIBLE_DEVICES": "GPU-real",
                    "SCCACHE_BUCKET": "secret",
                    "SCCACHE_REDIS": "secret",
                    "SCCACHE_CONF": "/shared/config",
                    "CCACHE_REMOTE_STORAGE": "secret",
                    "NVCC_APPEND_FLAGS": "--use_fast_math",
                    "GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE": "1",
                    "API_TOKEN": "secret",
                },
            )
            self.assertFalse(any("secret" in value for value in env.values()))
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-real")
            self.assertNotIn("GENERATIVEQC_KS_GEOMETRY_RESPONSE_REUSE", env)
            self.assertNotIn("NVCC_APPEND_FLAGS", env)
            for key in (
                "SCCACHE_DIR",
                "SCCACHE_CONF",
                "GENERATIVEQC_JIT_SCCACHE_ROOT",
                "CUDA_CACHE_PATH",
            ):
                self.assertTrue(Path(env[key]).is_relative_to(root))
            self.assertNotEqual(
                env["SCCACHE_DIR"], env["GENERATIVEQC_JIT_SCCACHE_ROOT"]
            )

    def test_private_empty_cache_cannot_clobber_existing_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "empty"
            driver.new_cache(root)
            sentinel = root / "do-not-delete"
            sentinel.write_bytes(b"retained")
            with self.assertRaises(FileExistsError):
                driver.new_cache(root)
            self.assertEqual(sentinel.read_bytes(), b"retained")

    def test_primed_integrity_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "artifact").write_bytes(b"complete")
            driver.mark_primed(root, "a" * 64)
            driver.check_primed(root)
            (root / "artifact").write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "primed cache"):
                driver.check_primed(root)

    def test_invalid_sdk_never_starts_production(self) -> None:
        with self.assertRaisesRegex(ValueError, "Standard"):
            driver.validate_config({"schema": driver.SCHEMA, "policy": "fast"})

    def test_failed_fresh_children_keep_raw_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pids = []
            for index in range(2):
                folder = Path(temporary) / str(index)
                folder.mkdir()
                request = folder / "request.json"
                driver.save(request, {"config": {}})
                outcome = driver.launch(request, dict(os.environ), folder, 20)
                pids.append(outcome["pid"])
                self.assertNotEqual(outcome["returncode"], 0)
                self.assertTrue((folder / "stderr.log").read_bytes())
                self.assertTrue((folder / "process.json").is_file())
                self.assertGreater(outcome["launch_to_exit_wall_seconds"], 0)
            self.assertNotEqual(*pids)

    def test_observer_preserves_original_semantic_cache_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            observer = driver.Observer()
            with observer:
                from generativeqc_compiler.common.semantic_source_cache import (
                    cached_sources,
                )

                first = cached_sources(
                    Path(temporary), {"input": "one"}, lambda: ("line1\nline2\n",)
                )
                second = cached_sources(
                    Path(temporary),
                    {"input": "one"},
                    lambda: self.fail("cache hit must replay"),
                )
            self.assertEqual(first[0], second[0])
            events = observer.receipt()["events"]
            self.assertEqual(len(events), 2)
            self.assertEqual({e["kind"] for e in events}, {"semantic-source"})
            self.assertIsNone(sys.getprofile())


if __name__ == "__main__":
    unittest.main()
