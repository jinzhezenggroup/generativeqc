"""Test emitted worker adaptation independently of the functional algebra."""

import re
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from generativeqc_compiler.method import resolve_method
from generativeqc_compiler.xc import libxc_bulk
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS
from generativeqc_compiler.xc.geometry_cuda import _emit_composed_point
from generativeqc_compiler.xc.libxc_maple import MapleImportError
from generativeqc_compiler.xc.scalar_schedule import (
    XcScalarConsumer,
    select_xc_scalar_schedule,
)
from generativeqc_compiler.xc.split_hybrid_domain import stabilize_m06_2x_stoll

from tools.generate_xc_split_hybrid_cuda import (
    SplitHybridWorkPolicy,
    _component_work_policy,
    _emit_work_wrapper,
    emit_split_hybrid_device,
    emit_split_hybrid_device_body,
)
from tools.libxc_split_hybrid import (
    _bound_component,
    _catalog,
    _root,
    build_split_global_hybrid,
)


def test_pinned_mgga_components_use_global_fhc_and_direct_spin_screens() -> None:
    for identifier in ("M06-2X", "MN15"):
        program = build_split_global_hybrid(identifier)
        for component in (program.exchange, program.correlation):
            assert _component_work_policy(component).enforce_fhc
        source = emit_split_hybrid_device(identifier)
        assert "if (rho_a <= " in source
        assert "if (rho_b <= " in source
        assert "work_sigma_aa = fmin(work_sigma_aa" in source


@pytest.mark.parametrize("identifier", ("M06-2X", "MN15"))
@pytest.mark.parametrize("lower_integer_powers", (False, True))
def test_split_mgga_components_lower_bounded_integer_powers_only_when_requested(
    identifier: str, lower_integer_powers: bool
) -> None:
    """Qualify FP64 lowering without changing the unqualified float schedule."""
    source, _, features = emit_split_hybrid_device_body(
        identifier, lower_integer_powers=lower_integer_powers
    )
    assert len(features) == 7
    exponents = re.findall(
        r"\bpowf?\([^,\n]+,\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)f?\)",
        source,
    )
    assert exponents
    has_bounded_integer_power = any(
        float(exponent).is_integer() and 1 <= abs(float(exponent)) <= 4
        for exponent in exponents
    )
    assert has_bounded_integer_power != lower_integer_powers


@pytest.mark.parametrize("identifier", ("M06-2X", "MN15"))
@pytest.mark.parametrize("scalar_type", ("double", "float"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_split_consumers_qualify_integer_power_schedule_separately(
    identifier: str, scalar_type: str, spin: str
) -> None:
    """Every FP64 force graph lowers powers; every SCF/float graph stays put."""
    semilocal = resolve_method(identifier, spin=spin).primitives[0].functional
    code = SPLIT_HYBRIDS[identifier]["functional_code"]
    force_source = _emit_composed_point(code, semilocal, scalar_type=scalar_type)
    renamed = replace(semilocal, identifier="structural-functional-alias")
    assert _emit_composed_point(code, renamed, scalar_type=scalar_type) == force_source
    point_source, _, _ = emit_split_hybrid_device_body(
        identifier, scalar_type=scalar_type
    )
    pattern = r"\bpowf?\([^,\n]+,\s*[-+]?(?:2|3|4)(?:\.0)?f?\)"
    assert re.search(pattern, point_source)
    assert bool(re.search(pattern, force_source)) == (scalar_type != "double")


@pytest.mark.parametrize("scalar_type", ("double", "float"))
@pytest.mark.parametrize(
    "consumer", (XcScalarConsumer.SCF_POINT, XcScalarConsumer.STATIONARY_FORCE)
)
def test_scalar_schedule_depends_only_on_consumer_and_precision(
    scalar_type: str, consumer: XcScalarConsumer
) -> None:
    schedule = select_xc_scalar_schedule(consumer=consumer, scalar_type=scalar_type)
    assert schedule.lower_integer_powers is (
        scalar_type == "double" and consumer is XcScalarConsumer.STATIONARY_FORCE
    )


def test_unknown_or_invalid_scalar_schedule_requests_fail_closed() -> None:
    with pytest.raises(TypeError, match="consumer"):
        select_xc_scalar_schedule(
            consumer="stationary_force",
            scalar_type="double",
        )
    with pytest.raises(ValueError, match="precision"):
        select_xc_scalar_schedule(
            consumer=XcScalarConsumer.STATIONARY_FORCE,
            scalar_type="half",
        )


def test_integer_power_lowering_rejects_unqualified_precision_and_policy() -> None:
    with pytest.raises(ValueError, match="FP64 qualification"):
        emit_split_hybrid_device_body(
            "M06-2X", scalar_type="float", lower_integer_powers=True
        )
    with pytest.raises(TypeError, match="boolean"):
        emit_split_hybrid_device_body("M06-2X", lower_integer_powers="true")


def test_stoll_rewrite_fails_closed_on_changed_modified_pw_parameters() -> None:
    record = _bound_component("MGGA_C_M06_2X", allow_hybrid_exchange=False)
    catalog, source_root = _catalog(), _root()
    raw = libxc_bulk.build_record(record, catalog["source_files"], source_root)
    module = libxc_bulk.import_record(record, catalog["source_files"], source_root)
    altered = replace(
        module,
        assignments=tuple(
            (name, "[0, 0, 0]" if name == "params_a_a" else value)
            for name, value in module.assignments
        ),
    )
    with pytest.raises(MapleImportError, match="modified PW parameter policy"):
        stabilize_m06_2x_stoll(raw, altered, record)


@pytest.mark.parametrize("identifier", ("M06-2X", "MN15"))
@pytest.mark.parametrize("consumer", ("scf_point", "stationary_force"))
def test_acceptance_header_binds_both_production_spins_without_patching_scf(
    identifier: str, consumer: str
) -> None:
    from tools.validate_split_hybrid_cuda import validation_header

    baseline = emit_split_hybrid_device(identifier)
    header, hashes = validation_header(identifier, consumer)
    assert emit_split_hybrid_device(identifier) == baseline
    if consumer == "scf_point":
        assert header == baseline
        assert not hashes
    else:
        assert header != baseline
        assert set(hashes) == {"unpolarized", "polarized"}


def test_stoll_rewrite_uses_physical_spin_nodes_and_keeps_source_guard() -> None:
    record = _bound_component("MGGA_C_M06_2X", allow_hybrid_exchange=False)
    catalog, source_root = _catalog(), _root()
    raw = libxc_bulk.build_record(record, catalog["source_files"], source_root)
    module = libxc_bulk.import_record(record, catalog["source_files"], source_root)
    stabilized = stabilize_m06_2x_stoll(raw, module, record)
    assert stabilized.energy.identifier != raw.energy.identifier
    assert stabilized.identity != raw.identity
    altered = replace(raw, energy=raw.variables[0] + raw.variables[1])
    with pytest.raises(MapleImportError, match="perpendicular PW terms changed"):
        stabilize_m06_2x_stoll(altered, module, record)


def test_emitted_mgga_work_adapter(tmp_path: Path) -> None:
    plain = SplitHybridWorkPolicy(1e-15, 1e-40, 1e-20, False, "test")
    fhc = SplitHybridWorkPolicy(1e-15, 1e-40, 1e-20, True, "test")
    source = r"""
#include <cmath>
#include <cassert>
#define __device__
struct Value {double energy_density{}; double feature_derivative[7]{};};
Value point_interior(double a,double b,double c,double d,double e,double f,double g) {
  return {2*(a+b),{a,b,c,d,e,f,g}};
}
Value fhc_interior(double a,double b,double c,double d,double e,double f,double g) {
  return point_interior(a,b,c,d,e,f,g);
}
"""
    source += _emit_work_wrapper("Value", "point", plain)
    source += _emit_work_wrapper("Value", "fhc", fhc)
    source += r"""
int main() {
  auto a = point(0,0,0,0,0,0,0);
  assert(a.energy_density == 0);
  for(double value : a.feature_derivative) assert(value == 0);
  a = point(0.1,0.2,0,0,0,0,0);
  assert(std::abs(a.energy_density-0.6)<1e-15);
  assert(a.feature_derivative[2]==1e-40);
  assert(a.feature_derivative[5]==1e-20);
  a = point(0.1,0,0.2,0.2,0,0.3,0);
  assert(std::abs(a.energy_density-0.2)<1e-15);
  assert(a.feature_derivative[1]==1e-15);
  assert(a.feature_derivative[3]==0.1);
  a = fhc(0.1,0.2,0.02,0,0.03,0.001,0.002);
  assert(std::abs(a.feature_derivative[2]-0.0008)<1e-15);
  assert(std::abs(a.feature_derivative[4]-0.0032)<1e-15);
}
"""
    path = tmp_path / "adapter.cpp"
    path.write_text(source, encoding="utf-8")
    executable = tmp_path / "adapter"
    compiler = shutil.which("c++")
    assert compiler is not None
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(path),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=30)
