"""Semantic source reuse must preserve the ordinary object/link cache boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_runtime import CudaArtifact
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.integral import first_derivative_schedule as primitives
from generativeqc_compiler.integral.first_derivative_schedule import derivative_requests
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.method import stationary_cuda as lowering
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def compile_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., CudaArtifact]:
    """Replace only binary compilation; recipe generation stays production-owned."""
    monkeypatch.setattr(
        lowering,
        "compile_cuda_object",
        lambda compiler, cache, source, **kwargs: source.read_text(),
    )
    monkeypatch.setattr(
        lowering,
        "link_cuda_objects",
        lambda compiler, cache, objects, **kwargs: CudaArtifact(
            tmp_path / "not-loaded.so", {"key": canonical_hash(objects)}
        ),
    )
    monkeypatch.setattr(
        primitives,
        "emit_first_derivative_cuda",
        lambda requests: "// primitive " + repr(requests),
    )
    monkeypatch.setattr(
        primitives,
        "derivative_cuda_sources",
        lambda domain: ((derivative_requests(domain), "// shard"),),
    )
    monkeypatch.setattr(
        lowering,
        "emit_stationary_wrapper_cuda",
        lambda **kwargs: "// wrapper " + kwargs["plan"].identity,
    )
    plan = StationaryGradientPlan(
        resolve_method("PBE0", spin="unpolarized"), StationaryMeanField(SCF_POINT_MODEL)
    )
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))

    def compile(**kwargs: Any) -> CudaArtifact:
        requests = kwargs.pop("primitive_requests", (("nuclear", ()),))
        domain = kwargs.pop("component_domain", None)
        explicit = kwargs.pop("primitive_source", None)
        selected_compiler = kwargs.get("compiler", compiler)
        enabled = kwargs.get("cache_generated_sources", True)
        provider = (
            explicit
            if explicit is not None
            else lambda: primitives.cached_derivative_cuda_source(
                requests,
                component_domain=domain,
                cache=tmp_path,
                target=selected_compiler.target,
                enabled=enabled,
            )
        )
        return lowering.compile_stationary_cuda(
            provider,
            **{
                "primitive_shard_width": 16 if domain is not None else None,
                "functional": 1,
                "plan": plan,
                "iterations": 3,
                "compiler": compiler,
                "cache": tmp_path,
                **kwargs,
            },
        )

    return compile


def test_hit_skips_both_producers_and_keeps_binary_identity(
    compile_recipe: Callable[..., CudaArtifact],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = compile_recipe()

    def forbidden(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("source producer reached on a semantic hit")

    monkeypatch.setattr(primitives, "emit_first_derivative_cuda", forbidden)
    monkeypatch.setattr(lowering, "emit_stationary_wrapper_cuda", forbidden)
    replay = compile_recipe()
    assert replay.metadata["key"] == first.metadata["key"]
    work = replay.metadata["source_cache"]
    assert work["primitive"]["hit"] and work["wrapper"]["hit"]
    assert work["primitive"]["requests"] == work["primitive"]["units"] == 1
    assert work["recipe_seconds"] >= 0 and work["binary_cache_seconds"] >= 0


@pytest.mark.parametrize(
    "change",
    ("requests", "domain", "iterations", "method", "target", "dependency", "abi"),
)
def test_semantic_changes_invalidate(
    compile_recipe: Callable[..., CudaArtifact],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    first = compile_recipe().metadata["source_cache"]
    kwargs = {}
    if change == "requests":
        kwargs["primitive_requests"] = (("overlap", ("", "")), ("nuclear", ()))
    elif change == "domain":
        kwargs.update(
            primitive_requests=derivative_requests(("",)), component_domain=("",)
        )
    elif change == "iterations":
        kwargs["iterations"] = 2
    elif change == "method":
        kwargs["plan"] = StationaryGradientPlan(
            resolve_method("PBE", spin="unpolarized"),
            StationaryMeanField(SCF_POINT_MODEL),
        )
    elif change == "target":
        kwargs["compiler"] = CudaCompilerAdapter(
            Path("nvcc"), cuda_target_info("sm_100")
        )
    else:
        original = lowering.source_hashes
        monkeypatch.setattr(
            lowering,
            "source_hashes",
            lambda *args, **kwargs: {
                **original(*args, **kwargs),
                "derivative-abi" if change == "abi" else "generator": "changed",
            },
        )
    changed = compile_recipe(**kwargs).metadata["source_cache"]
    product = "primitive" if change in ("requests", "domain") else "wrapper"
    assert not changed[product]["hit"]
    assert changed[product]["key"] != first[product]["key"]
    if change in ("method", "iterations"):
        assert changed["primitive"]["hit"]


def test_incompatible_demand_is_rejected(
    compile_recipe: Callable[..., CudaArtifact],
) -> None:
    with pytest.raises(ValueError, match="differs from primitive demand"):
        compile_recipe(component_domain=("",))
    with pytest.raises(ValueError, match="unique nonempty"):
        compile_recipe(primitive_requests=())


def test_no_cache_preserves_bytes_and_legacy_sources(
    compile_recipe: Callable[..., CudaArtifact],
) -> None:
    first = compile_recipe()
    disabled = compile_recipe(cache_generated_sources=False)
    assert first.metadata["key"] == disabled.metadata["key"]
    assert not disabled.metadata["source_cache"]["primitive"]["enabled"]
    assert not disabled.metadata["source_cache"]["wrapper"]["enabled"]
    legacy = compile_recipe(
        primitive_requests=None, primitive_source="// explicit source"
    )
    assert legacy.metadata["source_cache"]["primitive"] is None
    assert legacy.metadata["source_cache"]["wrapper"]["hit"]


def test_fresh_process_real_pbe0_sources_skip_ir_and_ad(tmp_path: Path) -> None:
    """Separate owners/processes use actual lowering once, never fake recipes."""
    script = r"""
import json, sys
from pathlib import Path
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_runtime import CudaArtifact
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.method import stationary_cuda as lowering, resolve_method
from generativeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL, StationaryGradientPlan, StationaryMeanField
from generativeqc_compiler.integral.first_derivative_schedule import derivative_requests
from generativeqc_compiler.integral import first_derivative_schedule as primitives
from generativeqc_compiler.method import stationary_becke_phased as becke
from generativeqc_compiler.integral.expr import Graph
def forbidden(*args, **kwargs):
    raise AssertionError('fresh process hit reconstructed source/IR/AD')
if sys.argv[2] == 'hit':
    primitives.emit_first_derivative_cuda = forbidden
    primitives.derivative_cuda_sources = forbidden
    lowering.emit_stationary_wrapper_cuda = forbidden
    becke.grid_partition_domain_program = forbidden
    Graph.differentiate = forbidden
lowering.compile_cuda_object = lambda compiler, cache, source, **kwargs: source.read_text()
lowering.link_cuda_objects = lambda compiler, cache, objects, **kwargs: CudaArtifact(Path('unused.so'), {'key': canonical_hash(objects)})
plan = StationaryGradientPlan(resolve_method('PBE0', spin='unpolarized'), StationaryMeanField(SCF_POINT_MODEL))
artifact = lowering.compile_stationary_cuda(
    lambda: primitives.cached_derivative_cuda_source(
        derivative_requests(('',)), component_domain=('',),
        cache=Path(sys.argv[1]), target=cuda_target_info('sm_120'),
    ), primitive_shard_width=16,
    functional=1, plan=plan, iterations=3,
    compiler=CudaCompilerAdapter(Path('nvcc'), cuda_target_info('sm_120')), cache=Path(sys.argv[1]),
)
print(json.dumps(artifact.metadata))
"""
    results = []
    for mode in ("miss", "hit"):
        run = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
            env={**os.environ, "PYTHONPATH": str(Path("python").resolve())},
        )
        results.append(json.loads(run.stdout))
    assert results[0]["key"] == results[1]["key"]
    assert not results[0]["source_cache"]["primitive"]["hit"]
    assert not results[0]["source_cache"]["wrapper"]["hit"]
    assert results[1]["source_cache"]["primitive"]["hit"]
    assert results[1]["source_cache"]["wrapper"]["hit"]
