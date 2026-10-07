"""Exact package coverage and cold loader admission require no TensorIR/AD or NVCC."""

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest
from generativeqc_compiler.common.provenance import file_hash
from generativeqc_compiler.method import resolve_method, stationary_cuda
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize("name", ["M06-2X", "MN15"])
def test_unrepresented_hybrids_have_no_exact_package_coverage(
    name: str, spin: str
) -> None:
    from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS

    plan = StationaryGradientPlan(
        resolve_method(name, spin=spin), StationaryMeanField(SCF_POINT_MODEL)
    )
    assert (
        stationary_cuda.stationary_aot_profile_for_plan(
            SPLIT_HYBRIDS[name]["functional_code"], spin, plan
        )
        is None
    )


def test_altered_semilocal_composition_does_not_alias_a_packaged_plan() -> None:
    method = resolve_method("PBE0")
    changed = replace(
        method,
        primitives=tuple(
            replace(
                node,
                functional=replace(
                    node.functional,
                    components=(
                        ("GGA_X_PBE", Fraction(2, 3)),
                        ("GGA_C_PBE", Fraction(1)),
                    ),
                ),
            )
            if node.kind == "semilocal_xc"
            else node
            for node in method.primitives
        ),
    )
    plan = StationaryGradientPlan(changed, StationaryMeanField(SCF_POINT_MODEL))
    assert (
        stationary_cuda.stationary_aot_profile_for_plan(1, "unpolarized", plan) is None
    )


@pytest.mark.parametrize(
    "profile_name", ["pbe0_rks", "pbe0_uks", "b3lyp_rks", "b3lyp_uks"]
)
@pytest.mark.parametrize("component", [False, True])
@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing",
        "json",
        "plan",
        "contract",
        "precision",
        "target",
        "binary",
        "weights",
        "weight_hex",
        "weight_profile",
        "weight_spin",
        "missing_integrity",
        "integrity",
        "source",
        "domain",
        "iterations",
    ],
)
def test_hybrid_aot_admission_without_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_name: str,
    component: bool,
    failure: str | None,
) -> None:
    """A cold valid load uses hashes only; declared broken artifacts fail closed."""
    profile = stationary_cuda._qualified_aot_profile(profile_name)
    domain = stationary_cuda.QUALIFIED_SPD_COMPONENTS if component else None
    name = f"{profile_name}_spd" if component else profile_name
    library = tmp_path / f"libgenerativeqc_stationary_{name}.so"
    library.write_bytes(b"opaque binary, checked but never executed")
    manifest = tmp_path / f"generativeqc_stationary_{name}.json"
    metadata = {
        "schema": f"generativeqc.stationary-cuda-aot.v{3 if component else 2}",
        "functional": profile.functional,
        "spin": profile.spin,
        "plan_identity": profile.plan.identity,
        "partition_iterations": 3,
        "contract_identity": stationary_cuda.stationary_aot_profile_contract_identity(
            profile_name, component_domain=domain
        ),
        "weight_programs": stationary_cuda.stationary_aot_profile_weight_programs(
            profile_name
        ),
        "compile_contract": {"fp64": True, "fmad": False},
        "architectures": ["sm_120"],
        "code_objects": [{"architecture": "sm_120", "kind": "cubin"}],
        "source_identity": "build-time-source",
        "binary_sha256": file_hash(library),
        "binary_bytes": library.stat().st_size,
    }
    if component:
        metadata.update(
            component_domain=list(domain),
            primitive_shard_width=stationary_cuda.QUALIFIED_SPD_AOT_SHARD_WIDTH,
            primitive_shards=stationary_cuda.QUALIFIED_SPD_AOT_SHARDS,
        )
    metadata["manifest_integrity_sha256"] = (
        stationary_cuda.stationary_aot_manifest_integrity(metadata)
    )
    if failure == "plan":
        metadata["plan_identity"] = "wrong plan"
    elif failure == "contract":
        metadata["contract_identity"] = "stale contract"
    elif failure == "precision":
        metadata["compile_contract"]["fmad"] = True
    elif failure == "binary":
        library.write_bytes(b"replaced binary")
    elif failure == "weights":
        metadata["weight_programs"] = {}
    elif failure == "weight_hex":
        original = metadata["weight_programs"]["exact_exchange"]
        metadata["weight_programs"]["exact_exchange"] = (
            "0" if original[0] != "0" else "1"
        ) + original[1:]
    elif failure in {"weight_profile", "weight_spin"}:
        other = (
            profile_name.replace("pbe0", "b3lyp")
            if profile_name.startswith("pbe0")
            else profile_name.replace("b3lyp", "pbe0")
        )
        if failure == "weight_spin":
            other = profile_name[:-3] + (
                "uks" if profile_name.endswith("rks") else "rks"
            )
        replacement = stationary_cuda.stationary_aot_profile_weight_programs(other)
        assert replacement != metadata["weight_programs"]
        metadata["weight_programs"] = replacement
    elif failure == "missing_integrity":
        del metadata["manifest_integrity_sha256"]
    elif failure == "integrity":
        metadata["manifest_integrity_sha256"] = "0" * 64
    elif failure == "source":
        metadata["source_identity"] = "different-build-source"
    manifest.write_text("{" if failure == "json" else json.dumps(metadata))
    if failure == "missing":
        manifest.unlink()

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("cold AOT loading regenerated mathematical IR/AD or source")

    stationary_cuda.stationary_aot_profile_contract_identity.cache_clear()
    monkeypatch.setattr(StationaryGradientPlan, "integral_block", forbidden)
    monkeypatch.setattr(StationaryGradientPlan, "reduction_program", forbidden)
    monkeypatch.setattr(stationary_cuda, "emit_stationary_wrapper_cuda", forbidden)
    monkeypatch.setattr(stationary_cuda, "emit_stationary_cuda", forbidden)
    load = lambda: stationary_cuda.load_stationary_aot_artifact(
        tmp_path,
        functional=profile.functional,
        spin=profile.spin,
        plan=profile.plan,
        architecture="sm_80" if failure == "target" else "sm_120",
        component_domain=("s",) if failure == "domain" else domain,
        iterations=2 if failure == "iterations" else 3,
    )
    if failure is None:
        artifact = load()
        assert artifact.library == library
        assert artifact.metadata["weight_programs"] == metadata["weight_programs"]
    else:
        errors = {
            "missing": (FileNotFoundError, "missing packaged"),
            "json": (json.JSONDecodeError, None),
            "plan": (ValueError, "plan_identity"),
            "contract": (ValueError, "contract_identity"),
            "precision": (ValueError, "precision contract"),
            "target": (NotImplementedError, "sm_80"),
            "binary": (ValueError, "binary integrity"),
            "weights": (ValueError, "weight-program provenance"),
            **{
                name: (ValueError, "manifest integrity")
                for name in (
                    "weight_hex",
                    "weight_profile",
                    "weight_spin",
                    "missing_integrity",
                    "integrity",
                    "source",
                )
            },
            "domain": (NotImplementedError, "s/p/d domain"),
            "iterations": (NotImplementedError, "partition_iterations"),
        }
        error, message = errors[failure]
        with pytest.raises(error, match=message):
            load()


@pytest.mark.parametrize("combined", [False, True])
def test_native_source_coverage_gate_is_strict_without_reduction_ir(
    monkeypatch: pytest.MonkeyPatch, combined: bool
) -> None:
    from generativeqc_compiler.method import stationary_gradient

    plan = StationaryGradientPlan(
        resolve_method("PBE0"), StationaryMeanField(SCF_POINT_MODEL)
    )
    expected = tuple(
        "two_electron" if combined and name == "coulomb" else name
        for name in plan.source_names
        if not (combined and name == "exact_exchange")
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("native coverage gate constructed a TensorIR input")

    monkeypatch.setattr(stationary_gradient, "_input", forbidden)
    assert (
        plan.validate_source_coverage(
            sources=tuple(reversed(expected)), combined_two_electron=combined
        )
        == expected
    )
    with pytest.raises(ValueError, match="incomplete or unknown"):
        plan.validate_source_coverage(
            sources=expected[:-1], combined_two_electron=combined
        )
    with pytest.raises(ValueError, match="duplicate"):
        plan.validate_source_coverage(
            sources=(*expected, expected[0]), combined_two_electron=combined
        )
