"""Value-only Direct Rys lowering and independent contracted Libcint gates."""

import ctypes
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from generativeqc_compiler.integral import (
    FUSED_SHELL_SPEC_BY_NAME,
    FusedShellPlan,
    KernelConsumer,
    KernelSelection,
    ScheduleKind,
    build_fused_shell_plan,
    cuda_target_info,
    emit_shell_class_fused_cuda,
)
from generativeqc_compiler.integral.capabilities import CAPABILITY_STREAMING_FOCK
from generativeqc_compiler.integral.lowering.fock_component import (
    _emit_rys_component_lane_fock_consumer_cuda,
    emit_rys_value_support_cuda,
)
from generativeqc_compiler.integral.production_emission import emit_production_shard

TARGET = cuda_target_info("sm_120")
CLASSES = ("ppps", "dpss", "ddss", "pppp", "ddpp", "dddp")


def value_plan(name: str) -> FusedShellPlan:
    """Use only value angular order to select the exact fixed-root identity."""
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    return build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK,),
        recurrence=f"rys{sum(spec.angular) // 2 + 1}",
        target=TARGET,
    )


@pytest.mark.parametrize("name", CLASSES)
def test_value_only_rys_has_its_own_roots_and_state_bounds(name: str) -> None:
    """Odd angular orders must not inherit the extra force quadrature root."""
    plan = value_plan(name)
    integral = plan.kernel.integral
    assert integral.derivative is None
    assert integral.maximum_coulomb_order == sum(plan.spec.angular)
    assert plan.schedule.kind == ScheduleKind.COMPONENT_LANES
    assert plan.schedule.block_threads >= plan.spec.component_count
    source = emit_shell_class_fused_cuda(plan.spec, plan)
    roots = integral.required_rys_roots
    assert f"generated_{name}_rys{roots}_value_axis" in source
    assert (
        f"volatile double trr[{sum(plan.spec.angular[:2]) + 1}][{sum(plan.spec.angular[2:]) + 1}]"
        in source
    )
    assert "shell_class_force_task" not in source
    assert "shell_class_force_rhf_kernel" not in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source
    assert f"generated_{name}_shell_class_fock_uhf_kernel" in source
    assert source == emit_shell_class_fused_cuda(plan.spec, plan)
    wider = replace(plan.schedule, block_threads=plan.schedule.block_threads + 32)
    explicit = emit_shell_class_fused_cuda(plan.spec, plan, fock_schedule=wider)
    class_name = name[0].upper() + name[1:]
    assert (
        f"kGenerated{class_name}FockBlockThreads = {wider.block_threads}U" in explicit
    )


def test_value_rys_rejects_unimplemented_schedule_and_decoder() -> None:
    """A candidate must not emit Cartesian math under a Rys artifact identity."""
    plan = value_plan("pppp")
    with pytest.raises(ValueError, match="component lane"):
        build_fused_shell_plan(
            plan.spec,
            integral=plan.kernel.integral,
            schedule=replace(plan.schedule, block_threads=32),
            target=TARGET,
        )
    with pytest.raises(ValueError, match="fourth center"):
        value_plan("dddd")


@pytest.mark.parametrize("name", CLASSES)
def test_value_rys_emits_normal_production_streaming_artifacts(name: str) -> None:
    """Use the ordinary registry ABI without defining absent force consumers."""
    plan = value_plan(name)
    selection = KernelSelection(
        architecture=TARGET.architecture,
        spec=plan.spec,
        consumers=(KernelConsumer.FOCK,),
        schedule=plan.schedule,
        recurrence=plan.kernel.integral.recurrence,
        integral=plan.kernel.integral,
        capabilities=frozenset((CAPABILITY_STREAMING_FOCK,)),
        fock_route="streaming",
        tuned=False,
    )
    source = emit_production_shard((selection,))
    assert f"generated_{name}_shell_class_fock_rhf_streaming_kernel" in source
    assert f"generated_{name}_shell_class_fock_uhf_streaming_kernel" in source
    assert "shell_class_force_rhf_persistent_kernel" not in source


@pytest.fixture(scope="module")
def compiled_axis_probe(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    """Compile the emitted value arithmetic, independently of native CUDA."""
    compiler, cache = shutil.which("c++"), shutil.which("ccache")
    if compiler is None or cache is None:
        pytest.skip("requires a host C++ compiler and ccache")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    directory = tmp_path_factory.mktemp("direct_rys_values")
    source = [
        "#include <cmath>\n#include <cstddef>\n#include <cstdint>\n",
        "#define __device__\n#define __constant__\n#define __forceinline__ inline\n",
        "#define __noinline__ __attribute__((noinline))\n",
    ]
    # Every probe executes the emitted TRR/HRR and strict root tables. Only
    # geometry packing is a host adapter; it uses the runtime primitive pairs.
    for name in CLASSES:
        plan = value_plan(name)
        roots = plan.kernel.integral.required_rys_roots
        axis = _emit_rys_component_lane_fock_consumer_cuda(plan.spec, plan, 1)
        axis = axis.split("template <bool Unrestricted>", 1)[0]
        source.append(f"namespace {name} {{\n")
        source.append(emit_rys_value_support_cuda(plan.spec, plan.kernel.integral))
        source.append(axis)
        source.append("}\n")
        source.append(
            f"""
extern "C" double probe_{name}(const double* bra, const double* ket,
                              const double* centers, const unsigned* powers) {{
  const double p=bra[0], q=ket[0], rho=p*q/(p+q);
  double difference[3], argument=0.0, rw[{2 * roots}];
  for(unsigned axis=0;axis<3;++axis) {{
    difference[axis]=bra[2+axis]-ket[2+axis];
    argument+=rho*difference[axis]*difference[axis];
  }}
  {name}::generated_dppp_rys{roots}_roots(argument,rw,1U);
  const double prefactor=34.986836655249725*bra[5]*ket[5]/(p*q*sqrt(p+q));
  double result=0;
  for(unsigned root=0;root<{roots};++root) {{
    const double u=rw[2*root]/(p+q), rb=u*q, rk=u*p;
    const double b10=.5/p*(1-rb), b00=.5*u, b01=.5/q*(1-rk);
    double value=rw[2*root+1]*prefactor;
    for(unsigned axis=0;axis<3;++axis) {{
      value*={name}::generated_dppp_rys{roots}_value_axis(
          powers[axis],powers[3+axis],powers[6+axis],powers[9+axis],
          bra[2+axis]-centers[axis]-difference[axis]*rb,
          ket[2+axis]-centers[6+axis]+difference[axis]*rk,
          centers[3+axis]-centers[axis],centers[9+axis]-centers[6+axis],
          b10,b00,b01,1.0);
    }}
    result+=value;
  }}
  return result;
}}
"""
        )
    cpp, output = directory / "probe.cpp", directory / "probe.so"
    cpp.write_text("".join(source))
    subprocess.run(
        [
            cache,
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(cpp),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(output))
    double = ctypes.POINTER(ctypes.c_double)
    uint = ctypes.POINTER(ctypes.c_uint)
    for name in CLASSES:
        function = getattr(library, f"probe_{name}")
        function.argtypes = (double, double, double, uint)
        function.restype = ctypes.c_double
    return library


@pytest.mark.parametrize("name", CLASSES)
@pytest.mark.parametrize("variant", ("cartesian", "coincident", "reversed_pairs"))
def test_emitted_value_axis_matches_independent_libcint(
    compiled_axis_probe: ctypes.CDLL,
    name: str,
    variant: str,
) -> None:
    """Contract signed unequal primitive lists and all Cartesian components."""
    pytest.importorskip("pyscf")
    from tools.generativeqc_validation.f_shell_numerics import (
        _reference_integrals,
        make_fixture,
    )

    fixture = make_fixture(name, variant)
    _, _, expected, _ = _reference_integrals(fixture.inputs)
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    function = getattr(compiled_axis_probe, f"probe_{name}")
    double = ctypes.POINTER(ctypes.c_double)
    uint = ctypes.POINTER(ctypes.c_uint)
    centers = np.ascontiguousarray(fixture.positions[list(fixture.atom_indices)])
    actual = np.empty(expected.shape)
    for component_index, component in zip(
        np.ndindex(expected.shape), spec.components, strict=True
    ):
        powers = np.array(
            [[label.count(axis) for axis in "xyz"] for label in component],
            dtype=np.uint32,
        )
        value = 0.0
        for bra in fixture.pairs[: fixture.pair_split]:
            for ket in fixture.pairs[fixture.pair_split :]:
                value += function(
                    bra.ctypes.data_as(double),
                    ket.ctypes.data_as(double),
                    centers.ctypes.data_as(double),
                    powers.ctypes.data_as(uint),
                )
        coefficient = np.prod(
            [
                fixture.ao_coefficients[offset + index]
                for offset, index in zip(
                    fixture.ao_offsets, component_index, strict=True
                )
            ]
        )
        actual[component_index] = value * coefficient
    np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-10)
