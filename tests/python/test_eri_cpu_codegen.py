"""Production CPU ERI domain, symmetry and independent compiled numerical gates."""

from __future__ import annotations

import ctypes
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.eri_cpu import (
    AXIS_PERMUTATIONS,
    CENTER_PERMUTATIONS,
    COMPONENTS,
    emit_eri_cpu,
    eri_cpu_component_map,
    eri_cpu_coulomb_layout,
    eri_cpu_inventory,
)


def test_complete_spd_value_inventory_and_lossless_symmetry_map() -> None:
    inventory = eri_cpu_inventory()
    assert inventory["version"] == 3
    assert inventory["maximum_angular"] == 2
    assert inventory["maximum_coulomb_order"] == 8
    assert inventory["derivative_order"] == 0
    assert inventory["ordered_component_count"] == 10000
    assert inventory["representative_count"] == 313
    assert inventory["primitive_geometry"] == {
        "abi_version": 1,
        "reuse_scope": "primitive_quartet",
        "maximum_order": 8,
        "required_order": "sum_shell_angular_momenta",
        "scalar_count": 29,
        "boys_value_count": 9,
    }
    assert inventory["component_schedule"] == {
        "kind": "prepared_geometry_coulomb_symmetry_representatives",
        "record_bits": 16,
        "maximum_shell_component_count": 1296,
        "component_scratch_owner": "native_caller",
    }
    assert inventory["coulomb_reuse"] == {
        "abi_version": 1,
        "reuse_scope": "primitive_quartet",
        "maximum_order": 8,
        "scalar_count": 165,
        "readiness": "matching_maximum_order",
        "order_prefix_counts": [1, 4, 10, 20, 35, 56, 84, 120, 165],
        "scratch_owner": "native_caller",
        "initialization": "requested_graded_prefix_only",
        "compatibility_schedule": "component_local_factored_dag",
    }
    assert len(inventory["values"]) == 81
    assert all(item["derivative"] is None for item in inventory["values"])
    assert all(
        item["contractions"][0]["consumer"] == "raw_block"
        for item in inventory["values"]
    )
    representatives, records = eri_cpu_component_map()
    for components, record in zip(product(COMPONENTS, repeat=4), records, strict=True):
        centers = CENTER_PERMUTATIONS[(record >> 3) & 7]
        axes = AXIS_PERMUTATIONS[record & 7]
        transformed = tuple(
            "".join(
                "xyz"[axis] * components[center].count("xyz"[original])
                for axis, original in enumerate(axes)
            )
            for center in centers
        )
        assert transformed == tuple(COMPONENTS[i] for i in representatives[record >> 6])


def test_coulomb_axis_map_preserves_every_graded_prefix() -> None:
    orders, mappings = eri_cpu_coulomb_layout()
    assert len(orders) == 165
    assert [sum(order) for order in orders] == sorted(sum(order) for order in orders)
    for axes, mapping in zip(AXIS_PERMUTATIONS, mappings, strict=True):
        assert sorted(mapping) == list(range(165))
        for order, mapped in zip(orders, mapping, strict=True):
            assert tuple(orders[mapped][original] for original in axes) == order
            assert sum(orders[mapped]) == sum(order)


@pytest.fixture(scope="module")
def generated_source() -> str:
    return emit_eri_cpu()


def test_deterministic_bounded_host_emission(generated_source: str) -> None:
    assert generated_source == emit_eri_cpu()
    # Two schedules share 313 generated mathematical representatives. The
    # component-local schedule prevents eager-table work in f+ compatibility.
    assert len(generated_source.encode()) < 6 * 1024 * 1024
    assert generated_source.count("static inline double component_") == 313
    assert "__device__" not in generated_source
    assert "__forceinline__" not in generated_source
    assert "cuda_runtime" not in generated_source
    assert "Jet" not in generated_source
    assert "double boys[9]" in generated_source
    assert "double boys[10]" not in generated_source
    assert generated_source.count("boys_values<") == 9
    assert "double values[165]; unsigned maximum_order = 9U;" in generated_source
    assert "primitive_dispatch<false>(geometry, nullptr, record)" in generated_source
    assert "primitive_dispatch<true>(geometry, &coulomb, record)" in generated_source
    preparation = generated_source.split("inline void prepare_coulomb", 1)[1]
    preparation = preparation.split("inline constexpr std::uint8_t", 1)[0]
    assert preparation.count("coulomb.values[") >= 165
    assert preparation.index("maximum_order == 0U") < preparation.index(
        "coulomb.values[1]"
    )
    assert "boys_values<" not in preparation
    components = generated_source.split("static inline double component_", 1)[1]
    components = components.split("inline constexpr unsigned center_permutations", 1)[0]
    assert "boys_values<" not in components
    assert "exponents[" not in components
    assert "centers[" not in components
    assert "exp(" not in components
    assert "sqrt(" not in components
    for component in components.split("static inline double component_"):
        local = component.split("  } else {", 1)[1]
        assert "coulomb->" not in local


@pytest.fixture(scope="module")
def compiled_evaluator(
    generated_source: str,
    tmp_path_factory: pytest.TempPathFactory,
    native_cxx,
) -> ctypes.CDLL:
    directory = tmp_path_factory.mktemp("eri_cpu")
    (directory / "generated_eri_cpu.hpp").write_text(generated_source)
    source = directory / "probe.cpp"
    evaluators = []
    for name, extra_argument, preparation, expression in (
        ("evaluate", "", "", "primitive(e,c,components)"),
        (
            "evaluate_prepared",
            ", unsigned maximum_order",
            (
                "  const auto geometry = make_geometry(e,c,maximum_order);\n"
                "  CoulombValues coulomb;\n"
                "  for (double& value : coulomb.values) value = NAN;\n"
                "  prepare_coulomb(geometry,coulomb);\n"
            ),
            "prepared_primitive(geometry,coulomb,component_record(components))",
        ),
    ):
        evaluators.append(
            f'extern "C" void {name}(const double* exponents, const double* centers, '
            f"double* output{extra_argument}) {{\n"
            "  double e[4], c[4][3];\n"
            "  for (unsigned i=0; i<4; ++i) { e[i]=exponents[i];\n"
            "    for (unsigned a=0; a<3; ++a) c[i][a]=centers[3*i+a]; }\n"
            f"{preparation}"
            "  for (unsigned a=0; a<10; ++a) for (unsigned b=0; b<10; ++b)\n"
            "    for (unsigned d=0; d<10; ++d) for (unsigned f=0; f<10; ++f) {\n"
            "      const unsigned components[4]{a,b,d,f};\n"
            f"      output[((a*10+b)*10+d)*10+f] = {expression};\n"
            "    }\n"
            "}\n"
        )
    source.write_text(
        '#include "generated_eri_cpu.hpp"\n'
        "using namespace generativeqc::integrals::generated_eri_cpu;\n"
        + "".join(evaluators)
        + 'extern "C" void evaluate_reused(const double* exponents, const double* centers, double* output) {\n'
        "  const unsigned orders[]{8,0,4,1,7,2,8,3,6,5,0};\n"
        "  CoulombValues coulomb;\n"
        "  for (unsigned step=0; step<11; ++step) {\n"
        "    double e[4], c[4][3];\n"
        "    for (unsigned i=0; i<4; ++i) {\n"
        "      e[i] = exponents[i] * (1.0 + 0.013*(step+1)*(i+1));\n"
        "      for (unsigned a=0; a<3; ++a)\n"
        "        c[i][a] = centers[3*i+a] + 0.007*(step+1)*(i+1)*(a+1);\n"
        "    }\n"
        "    const auto geometry = make_geometry(e,c,orders[step]);\n"
        "    prepare_coulomb(geometry,coulomb);\n"
        "    for (unsigned a=0; a<10; ++a) for (unsigned b=0; b<10; ++b)\n"
        "      for (unsigned d=0; d<10; ++d) for (unsigned f=0; f<10; ++f) {\n"
        "        const unsigned components[4]{a,b,d,f};\n"
        "        output[step*10000+((a*10+b)*10+d)*10+f] =\n"
        "          prepared_primitive(geometry,coulomb,component_record(components));\n"
        "      }\n"
        "  }\n"
        "}\n"
        + 'extern "C" void coulomb_prefix(unsigned maximum_order, double* output) {\n'
        "  const double e[4]{0.6,0.8,1.1,0.9};\n"
        "  const double c[4][3]{{0.13,-0.31,0.24},{-0.43,0.27,0.51},\n"
        "                       {0.68,-0.14,-0.22},{-0.21,0.48,-0.63}};\n"
        # Hold Boys inputs fixed: this gate checks the requested Coulomb prefix,
        # not bitwise equality between distinct-order Boys evaluation schedules.
        "  auto geometry = make_geometry(e,c,8);\n"
        "  geometry.maximum_order = maximum_order;\n"
        "  CoulombValues coulomb;\n"
        "  for (double& value : coulomb.values) value = NAN;\n"
        "  prepare_coulomb(geometry,coulomb);\n"
        "  for (unsigned i=0; i<165; ++i) output[i] = coulomb.values[i];\n"
        "}\n" + 'extern "C" double unsupported(unsigned scenario) {\n'
        "  const double e[4]{1,1,1,1}, c[4][3]{};\n"
        "  const unsigned components[4]{10,0,0,0}, dddd[4]{4,4,4,4};\n"
        "  const auto geometry = make_geometry(e,c,8);\n"
        "  CoulombValues coulomb;\n"
        "  if (scenario != 6) prepare_coulomb(geometry,coulomb);\n"
        "  switch (scenario) {\n"
        "    case 0: return primitive(e,c,components);\n"
        "    case 1: return prepared_primitive(geometry,component_record(components));\n"
        "    case 2: return prepared_primitive(geometry,313U << 6);\n"
        "    case 3: return prepared_primitive(geometry,6U);\n"
        "    case 4: return prepared_primitive(make_geometry(e,c,7),component_record(dddd));\n"
        "    case 5: return prepared_primitive(make_geometry(e,c,9),0U);\n"
        "    case 6: return prepared_primitive(geometry,coulomb,0U);\n"
        "    case 7: return prepared_primitive(make_geometry(e,c,7),coulomb,0U);\n"
        "    case 8: return prepared_primitive(geometry,coulomb,313U << 6);\n"
        "    case 9: return prepared_primitive(geometry,coulomb,6U);\n"
        "    case 10: return prepared_primitive(make_geometry(e,c,9),coulomb,0U);\n"
        "  }\n"
        "  return 0.0;\n"
        "}\n"
    )
    library = directory / "probe.so"
    native_cxx.build_shared(
        [source],
        library,
        compile_args=("-std=c++20", "-O2", "-ffp-contract=off"),
        compile_timeout=180,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.evaluate.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
    loaded.evaluate.restype = None
    loaded.evaluate_prepared.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [
        ctypes.c_uint
    ]
    loaded.evaluate_prepared.restype = None
    loaded.evaluate_reused.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
    loaded.evaluate_reused.restype = None
    loaded.unsupported.argtypes = [ctypes.c_uint]
    loaded.unsupported.restype = ctypes.c_double
    loaded.coulomb_prefix.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_double)]
    loaded.coulomb_prefix.restype = None
    return loaded


_CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)
_EXPONENTS = np.array([0.6, 0.8, 1.1, 0.9])


def test_coulomb_scratch_reuse_with_changed_geometry_and_order(
    compiled_evaluator: ctypes.CDLL,
) -> None:
    pointer = ctypes.POINTER(ctypes.c_double)
    orders = [8, 0, 4, 1, 7, 2, 8, 3, 6, 5, 0]
    actual = np.empty((len(orders), 10000))
    compiled_evaluator.evaluate_reused(
        _EXPONENTS.ctypes.data_as(pointer),
        _CENTERS.ctypes.data_as(pointer),
        actual.ctypes.data_as(pointer),
    )
    component_orders = np.array(
        [sum(map(len, components)) for components in product(COMPONENTS, repeat=4)]
    )
    for step, maximum_order in enumerate(orders):
        exponents = _EXPONENTS * (1 + 0.013 * (step + 1) * np.arange(1, 5))
        centers = _CENTERS + 0.007 * (step + 1) * np.outer(
            np.arange(1, 5), np.arange(1, 4)
        )
        expected = np.empty(10000)
        compiled_evaluator.evaluate(
            exponents.ctypes.data_as(pointer),
            centers.ctypes.data_as(pointer),
            expected.ctypes.data_as(pointer),
        )
        supported = component_orders <= maximum_order
        assert np.all(np.isnan(actual[step, ~supported]))
        np.testing.assert_allclose(
            actual[step, supported], expected[supported], rtol=2e-11, atol=3e-12
        )


@pytest.mark.parametrize("scenario", range(11))
def test_invalid_prepared_inputs_return_nan(
    compiled_evaluator: ctypes.CDLL, scenario: int
) -> None:
    assert np.isnan(compiled_evaluator.unsupported(scenario))


@pytest.mark.parametrize("maximum_order", range(9))
def test_coulomb_preparation_initializes_only_requested_prefix(
    compiled_evaluator: ctypes.CDLL, maximum_order: int
) -> None:
    pointer = ctypes.POINTER(ctypes.c_double)
    complete, actual = np.empty(165), np.empty(165)
    compiled_evaluator.coulomb_prefix(8, complete.ctypes.data_as(pointer))
    compiled_evaluator.coulomb_prefix(maximum_order, actual.ctypes.data_as(pointer))
    count = (maximum_order + 1) * (maximum_order + 2) * (maximum_order + 3) // 6
    np.testing.assert_array_equal(actual[:count], complete[:count])
    assert np.all(np.isnan(actual[count:]))


@pytest.mark.parametrize("maximum_order", range(9))
def test_shared_boys_order_covers_exactly_requested_components(
    compiled_evaluator: ctypes.CDLL, maximum_order: int
) -> None:
    pointer = ctypes.POINTER(ctypes.c_double)
    expected, actual = np.empty(10000), np.empty(10000)
    arguments = (
        _EXPONENTS.ctypes.data_as(pointer),
        _CENTERS.ctypes.data_as(pointer),
    )
    compiled_evaluator.evaluate(*arguments, expected.ctypes.data_as(pointer))
    compiled_evaluator.evaluate_prepared(
        *arguments, actual.ctypes.data_as(pointer), maximum_order
    )
    supported = np.array(
        [
            sum(map(len, components)) <= maximum_order
            for components in product(COMPONENTS, repeat=4)
        ]
    )
    assert np.all(np.isnan(actual[~supported]))
    np.testing.assert_allclose(
        actual[supported], expected[supported], rtol=2e-11, atol=3e-12
    )


@pytest.mark.parametrize(
    "case",
    ["generic", "coincident", "large_boys", "diffuse", "below_cutoff", "above_cutoff"],
)
def test_all_ordered_spd_components_match_independent_libcint(
    compiled_evaluator: ctypes.CDLL, case: str
) -> None:
    gto = pytest.importorskip("pyscf.gto")
    centers, exponents = _CENTERS.copy(), _EXPONENTS.copy()
    if case == "coincident":
        centers[:] = 0
    elif case == "large_boys":
        centers[2:] += [9, 2, -1]
    elif case == "diffuse":
        exponents[:] = [0.0003, 0.04, 0.002, 0.007]
    elif case in ("below_cutoff", "above_cutoff"):
        p, q = exponents[:2].sum(), exponents[2:].sum()
        argument = 20 + (-1 if case == "below_cutoff" else 1) * 1e-8
        centers[:] = 0
        centers[2:, 0] = np.sqrt(argument / (p * q / (p + q)))
    molecule = gto.M(
        atom=[(f"H{slot}", center) for slot, center in enumerate(centers)],
        basis={
            f"H{slot}": [[angular, [exponent, 1.0]] for angular in range(3)]
            for slot, exponent in enumerate(exponents)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    # libcint's primitive radial coefficients include normalization. Its s/p
    # common angular factors are explicit; d Cartesian functions use factor 1.
    factors = []
    for slot in range(4):
        row = []
        for angular in range(3):
            shell = molecule._bas[3 * slot + angular]
            common = (1 / np.sqrt(4 * np.pi), np.sqrt(3 / (4 * np.pi)), 1)[angular]
            coefficient = molecule._env[shell[gto.PTR_COEFF]] * common
            row.extend([coefficient] * ((angular + 1) * (angular + 2) // 2))
        factors.append(row)
    expected = molecule.intor("int2e_cart", shls_slice=(0, 3, 3, 6, 6, 9, 9, 12))
    actual = np.empty((10, 10, 10, 10))
    prepared = np.empty_like(actual)
    pointer = ctypes.POINTER(ctypes.c_double)
    compiled_evaluator.evaluate(
        exponents.ctypes.data_as(pointer),
        centers.ctypes.data_as(pointer),
        actual.ctypes.data_as(pointer),
    )
    compiled_evaluator.evaluate_prepared(
        exponents.ctypes.data_as(pointer),
        centers.ctypes.data_as(pointer),
        prepared.ctypes.data_as(pointer),
        8,
    )
    f = np.asarray(factors)
    normalization = np.einsum("i,j,k,l->ijkl", f[0], f[1], f[2], f[3])
    np.testing.assert_allclose(actual * normalization, expected, rtol=2e-11, atol=3e-12)
    np.testing.assert_allclose(
        prepared * normalization, expected, rtol=2e-11, atol=3e-12
    )


def test_native_ao_component_indices_reuse_generated_authority(
    generated_source: str, tmp_path: Path, native_cxx
) -> None:
    """Run the actual AO construction/preparation against generated metadata.

    Instrument only classification calls; no recurrence is compiled or mocked.
    Native basis.cpp owns component enumeration and normalization, including f/g.
    """
    root = Path(__file__).resolve().parents[2]
    source = (root / "src/integrals/s_integrals.cpp").read_text()
    ao_helpers = (
        "struct AoView"
        + source.split("struct AoView", 1)[1].split("struct GlobalExpansionTerm", 1)[0]
    )
    preparation = (
        "struct ValueEriComponent"
        + source.split("struct ValueEriComponent", 1)[1].split(
            "std::size_t build_value_eri_shell_quartet(", 1
        )[0]
    )
    classification = (
        "constexpr unsigned component_index"
        + generated_source.split("constexpr unsigned component_index", 1)[1].split(
            "// Cartesian representative", 1
        )[0]
    )
    records = (
        "inline constexpr std::uint16_t component_map"
        + generated_source.split("inline constexpr std::uint16_t component_map", 1)[
            1
        ].split("inline constexpr unsigned representative_orders", 1)[0]
    )
    lookup = (
        "inline unsigned component_record"
        + generated_source.split("inline unsigned component_record", 1)[1].split(
            "/** Unnormalized, unscreened component", 1
        )[0]
    )
    # Keep generated classification as the only mapping authority. This wrapper
    # observes placement/count without changing the production implementation.
    classification = classification.replace(
        "component_index(", "classification_authority("
    )
    text = (
        "#include <array>\n#include <bit>\n#include <cstddef>\n#include <cstdint>\n"
        "#include <iostream>\n#include <limits>\n#include <stdexcept>\n#include <vector>\n"
        '#include "molecule/basis.hpp"\n'
        "namespace generativeqc::integrals::generated_eri_cpu {\n"
        + classification
        + "std::size_t classifications = 0;\n"
        + "unsigned component_index(unsigned x, unsigned y, unsigned z) {\n"
        + "  ++classifications; return classification_authority(x, y, z);\n}\n"
        + records
        + lookup
        + "}\nnamespace generativeqc::integrals {\n"
        + ao_helpers
        + preparation
        + "}\nusing namespace generativeqc;\nusing namespace generativeqc::integrals;\n"
        + r"""
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
struct PreviousAoView {
  const core::Shell* shell{};
  molecule::CartesianComponent angular{};
  double component_normalization{};
};
void check_layout() {
  const auto old_normalization = offsetof(PreviousAoView, component_normalization);
  const auto angular_end = offsetof(PreviousAoView, angular) + sizeof(molecule::CartesianComponent);
  const auto index_offset = offsetof(AoView, eri_component_index);
  require(offsetof(AoView, shell) == offsetof(PreviousAoView, shell) &&
          offsetof(AoView, angular) == offsetof(PreviousAoView, angular), "AO prefix changed");
  // The optimization must use existing padding on ABIs where it fits. Other
  // ABIs remain supported: their measured layout is reported, not assumed.
  if (old_normalization - angular_end >= sizeof(unsigned) &&
      angular_end % alignof(unsigned) == 0) {
    require(sizeof(AoView) == sizeof(PreviousAoView) &&
            offsetof(AoView, component_normalization) == old_normalization &&
            index_offset == angular_end, "AO cache failed to reuse available padding");
  }
  std::cout << "old_size=" << sizeof(PreviousAoView) << " size=" << sizeof(AoView)
            << " shell_offset=" << offsetof(AoView, shell)
            << " angular_offset=" << offsetof(AoView, angular)
            << " index_offset=" << index_offset
            << " normalization_offset=" << offsetof(AoView, component_normalization) << '\n';
}
void check_expansion(const core::System& system, const std::vector<AoView>& aos) {
  std::size_t offset = 0;
  for (const auto& shell : system.shells) {
    for (const auto& angular : molecule::cartesian_components(shell.angular_momentum)) {
      const auto& ao = aos[offset++];
      require(ao.shell == &shell && ao.angular == angular, "AO identity/order changed");
      require(ao.eri_component_index == generated_eri_cpu::classification_authority(
                  angular[0], angular[1], angular[2]), "AO cached index differs from authority");
      require(std::bit_cast<std::uint64_t>(ao.component_normalization) ==
                  std::bit_cast<std::uint64_t>(molecule::cartesian_component_normalization(angular)),
              "AO normalization changed");
    }
  }
  require(offset == aos.size(), "AO expansion count changed");
}
int main() {
  check_layout();
  require(expand_cartesian_aos(core::System{}).empty(), "empty expansion changed");
  core::System all_angular;
  for (unsigned l : {0U, 1U, 2U, 3U, 4U, 4U, 3U, 2U, 1U, 0U})
    all_angular.shells.push_back({0, l, {{0.7, 1.0}}});
  auto aos = expand_cartesian_aos(all_angular);
  check_expansion(all_angular, aos);
  require(generated_eri_cpu::classifications == aos.size(), "classification is not per AO");
  // Unsupported cache entries must survive unchanged, and lookup must still
  // reject them in every quartet position, rather than indexing out of range.
  const unsigned valid = generated_eri_cpu::classification_authority(0, 0, 0);
  for (unsigned l : {3U, 4U}) {
    const unsigned unsupported = generated_eri_cpu::classification_authority(l, 0, 0);
    for (unsigned slot = 0; slot < 4; ++slot) {
      unsigned indices[4]{valid, valid, valid, valid};
      indices[slot] = unsupported;
      require(generated_eri_cpu::component_record(indices) == 0xffffffffU,
              "unsupported generated index accepted");
      indices[slot] = std::numeric_limits<unsigned>::max();
      require(generated_eri_cpu::component_record(indices) == 0xffffffffU,
              "out-of-range generated index accepted");
    }
  }
  ValueEriComponents components;
  for (const auto& ao : aos) {
    if (ao.shell->angular_momentum < 3) continue;
    const std::vector<AoView> unsupported_ao{ao};
    const auto before = generated_eri_cpu::classifications;
    require(prepare_value_eri_components(unsupported_ao, {0, 1}, {0, 0, 0, 0}, components) == 1 &&
            components[0].record == 0xffffffffU, "cached unsupported sentinel was not validated");
    require(generated_eri_cpu::classifications == before, "unsupported AO was reclassified");
  }
  std::size_t blocks = 0, prepared = 0, expanded = 0;
  for (std::size_t nshell = 1; nshell <= 4; ++nshell) {
    std::size_t layouts = 1;
    for (std::size_t s = 0; s < nshell; ++s) layouts *= 3;
    for (std::size_t layout = 0; layout < layouts; ++layout) {
      core::System system;
      std::vector<std::size_t> offsets{0};
      auto code = layout;
      for (std::size_t s = 0; s < nshell; ++s) {
        const unsigned l = code % 3;
        code /= 3;
        system.shells.push_back({0, l, {{0.7, 1.0}}});
        offsets.push_back(offsets.back() + molecule::cartesian_count(l));
      }
      const auto before = generated_eri_cpu::classifications;
      aos = expand_cartesian_aos(system);
      expanded += aos.size();
      check_expansion(system, aos);
      const auto after = generated_eri_cpu::classifications;
      require(after - before == aos.size(), "classification count differs from AO count");
      for (std::size_t si = 0; si < nshell; ++si)
        for (std::size_t sj = 0; sj <= si; ++sj)
          for (std::size_t sk = 0; sk <= si; ++sk)
            for (std::size_t sl = 0; sl <= sk; ++sl) {
              if (si == sk && sj < sl) continue;
              const auto count = prepare_value_eri_components(aos, offsets, {si, sj, sk, sl}, components);
              require(generated_eri_cpu::classifications == after, "quartet reclassified an AO");
              std::size_t item = 0;
              for (auto i = offsets[si]; i < offsets[si + 1]; ++i)
                for (auto j = offsets[sj]; j < offsets[sj + 1]; ++j)
                  for (auto k = offsets[sk]; k < offsets[sk + 1]; ++k)
                    for (auto l = offsets[sl]; l < offsets[sl + 1]; ++l) {
                      if (si == sj && j > i) continue;
                      if (sk == sl && l > k) continue;
                      if (si == sk && sj == sl && i*(i+1)/2+j < k*(k+1)/2+l) continue;
                      const std::array<std::size_t, 4> indices{i, j, k, l};
                      unsigned direct[4];
                      for (unsigned slot = 0; slot < 4; ++slot) {
                        const auto& a = aos[indices[slot]].angular;
                        direct[slot] = generated_eri_cpu::classification_authority(a[0], a[1], a[2]);
                      }
                      const double normalization = aos[i].component_normalization *
                          aos[j].component_normalization * aos[k].component_normalization *
                          aos[l].component_normalization;
                      require(item < count, "component missing");
                      const auto& component = components[item++];
                      require(component.indices == indices && component.record ==
                                  generated_eri_cpu::component_record(direct), "component identity changed");
                      require(std::bit_cast<std::uint64_t>(component.normalization) ==
                                  std::bit_cast<std::uint64_t>(normalization), "normalization association changed");
                      require(std::bit_cast<std::uint64_t>(component.value) == 0, "nonzero accumulator");
                    }
              require(item == count, "extra prepared component");
              ++blocks; prepared += count;
            }
    }
  }
  require(blocks == 5079, "shell-layout coverage changed");
  std::cout << "blocks=" << blocks << " prepared=" << prepared << " expanded=" << expanded << '\n';
}
"""
    )
    probe = tmp_path / "ao_component_indices.cpp"
    probe.write_text(text)
    binary = tmp_path / "ao_component_indices"
    native_cxx.build_executable(
        [probe, root / "src/molecule/basis.cpp"],
        binary,
        compile_args=(
            "-std=c++20",
            "-O1",
            "-ffp-contract=off",
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
        ),
    )
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    print(result.stdout, end="")
    assert "blocks=5079" in result.stdout
