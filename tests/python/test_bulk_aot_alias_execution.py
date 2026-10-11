"""Real registration aliases reuse a closed object without sharing provenance."""

from __future__ import annotations

import ctypes
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from generativeqc_compiler.common.compiler_process import run_compiler
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.xc import bulk_aot, bulk_aot_closure, bulk_aot_store
from generativeqc_compiler.xc.libxc_bulk import BULK_SEMANTICS, build_bulk_program
from test_bulk_aot_closure import compile_controlled

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    import pytest

pytest_plugins = ("test_bulk_aot_closure",)


def _variant(name: str, spin: str = "unpolarized") -> bulk_aot.SourceVariant:
    """Use real imported registrations and preserve each registration identity."""
    return bulk_aot.inspect_program(
        build_bulk_program(name, spin=spin), 1, "cpu", domain=BULK_SEMANTICS
    )


def test_real_lda_alias_executes_one_reused_closed_object(
    tmp_path: Path, real_gcc: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CHACHIYO aliases coincide only in this spin/output specialization.

    One canonical source path and one immutable toolchain snapshot bracket the
    actual object build. The alias performs a fresh closure check and loads
    the stored object, not a second object build or a copied scientific receipt.
    Both real names must independently agree with pinned Libxc E/vxc.
    """
    from pyscf import __version__ as pyscf_version
    from pyscf.dft import libxc

    assert pyscf_version == "2.14.0" and libxc.__version__ == "7.0.0"
    variants = [_variant(name) for name in ("LDA_C_CHACHIYO", "LDA_C_CHACHIYO_MOD")]
    assert variants[0].import_identity != variants[1].import_identity
    assert variants[0].source == variants[1].source
    plan = bulk_aot.plan_package(variants)
    assert plan["summary"]["unique_artifacts"] == 1
    assert plan["summary"]["reused_build_tasks"] == 1
    assert len(plan["artifacts"][0]["registrations"]) == 2
    assert plan["capability_claims"] == []

    wrapper = "\nvoid evaluate(const double* features, double* outputs) { bulk_xc_point(features, outputs); }\n"
    observed = [
        replace(variant, source=variant.source + wrapper) for variant in variants
    ]
    source = tmp_path / "shared-point.c"
    source.write_text(observed[0].source)
    compiler, manifest = real_gcc

    def collect(variant: bulk_aot.SourceVariant) -> Any:
        result = bulk_aot_closure.collect_cpu_closure(
            variant, source, compiler=compiler, toolchain=manifest
        )
        assert result.reasons == () and result.closure is not None
        assert result.closure.reusable
        return result

    before = collect(observed[0])
    object_path = tmp_path / "shared-point.o"
    compiled = compile_controlled(
        [compiler, *before.recipe["flags"], "-c", str(source), "-o", str(object_path)],
        monkeypatch,
    )
    assert compiled.returncode == 0 and not compiled.timed_out, compiled.stderr
    after = collect(observed[0])
    assert before.closure.cache_key == after.closure.cache_key
    stored = bulk_aot_store.store_artifact(
        tmp_path / "store", after.closure, object_path
    )
    assert stored.status == "hit" and stored.object_sha256 == file_hash(object_path)
    object_path.unlink()

    alias = collect(observed[1])
    assert alias.closure.cache_key == after.closure.cache_key
    reused = bulk_aot_store.lookup_artifact(tmp_path / "store", alias.closure)
    assert reused.status == "hit" and reused.artifact_path == stored.artifact_path
    assert reused.object_sha256 == stored.object_sha256
    binary = tmp_path / "shared-point.so"
    linked = run_compiler(
        [compiler, "-shared", str(reused.artifact_path), "-lm", "-o", str(binary)],
        60,
        label="reused alias object link",
        environment=bulk_aot_closure.CPU_ENVIRONMENT,
    )
    assert linked.returncode == 0 and not linked.timed_out, linked.stderr
    library = ctypes.CDLL(str(binary))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (pointer, pointer)
    library.evaluate.restype = None
    densities = np.array([1e-5, 0.001, 0.1, 1.0, 4.0])
    actual = np.empty((len(densities), 2))
    for index, density in enumerate(densities):
        inputs = np.array([density])
        library.evaluate(
            inputs.ctypes.data_as(pointer), actual[index].ctypes.data_as(pointer)
        )
    for variant in variants:
        energy, potential, _, _ = libxc.eval_xc(
            variant.name, densities, spin=0, deriv=1
        )
        expected = np.column_stack((energy * densities, potential[0]))
        np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=2e-9)


def test_alias_reuse_does_not_escape_its_spin_specialization() -> None:
    """Sharing the unpolarized object never merges distinct polarized programs."""
    original = _variant("LDA_C_CHACHIYO", "polarized")
    modified = _variant("LDA_C_CHACHIYO_MOD", "polarized")
    assert original.import_identity != modified.import_identity
    assert original.emission_identity != modified.emission_identity
    assert (
        bulk_aot.plan_package((original, modified))["summary"]["unique_artifacts"] == 2
    )
