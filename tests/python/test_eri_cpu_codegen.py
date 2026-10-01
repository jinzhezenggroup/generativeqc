"""Production CPU ERI domain, symmetry and independent compiled numerical gates."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from itertools import product

import numpy as np
import pytest
from generativeqc_compiler.integral.eri_cpu import (
    AXIS_PERMUTATIONS,
    CENTER_PERMUTATIONS,
    COMPONENTS,
    emit_eri_cpu,
    eri_cpu_component_map,
    eri_cpu_inventory,
)


def test_complete_spd_value_inventory_and_lossless_symmetry_map() -> None:
    inventory = eri_cpu_inventory()
    assert inventory["version"] == 2
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
        "kind": "prepared_geometry_symmetry_representatives",
        "record_bits": 16,
        "maximum_shell_component_count": 1296,
        "component_scratch_owner": "native_caller",
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


@pytest.fixture(scope="module")
def generated_source() -> str:
    return emit_eri_cpu()


def test_deterministic_bounded_host_emission(generated_source: str) -> None:
    assert generated_source == emit_eri_cpu()
    assert len(generated_source.encode()) < 4 * 1024 * 1024
    assert generated_source.count("static inline double component_") == 313
    assert "__device__" not in generated_source
    assert "__forceinline__" not in generated_source
    assert "cuda_runtime" not in generated_source
    assert "Jet" not in generated_source
    assert "double boys[9]" in generated_source
    assert "double boys[10]" not in generated_source
    assert generated_source.count("boys_values<") == 9
    components = generated_source.split("static inline double component_", 1)[1]
    components = components.split("inline constexpr unsigned center_permutations", 1)[0]
    assert "boys_values<" not in components
    assert "exponents[" not in components
    assert "centers[" not in components
    assert "exp(" not in components
    assert "sqrt(" not in components


@pytest.fixture(scope="module")
def compiled_evaluator(
    generated_source: str, tmp_path_factory: pytest.TempPathFactory
) -> ctypes.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("compiled ERI gate requires a C++ compiler")
    directory = tmp_path_factory.mktemp("eri_cpu")
    (directory / "generated_eri_cpu.hpp").write_text(generated_source)
    source = directory / "probe.cpp"
    evaluators = []
    for name, extra_argument, preparation, expression in (
        ("evaluate", "", "", "primitive(e,c,components)"),
        (
            "evaluate_prepared",
            ", unsigned maximum_order",
            "  const auto geometry = make_geometry(e,c,maximum_order);\n",
            "prepared_primitive(geometry,component_record(components))",
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
        + 'extern "C" double unsupported(unsigned scenario) {\n'
        "  const double e[4]{1,1,1,1}, c[4][3]{};\n"
        "  const unsigned components[4]{10,0,0,0}, dddd[4]{4,4,4,4};\n"
        "  const auto geometry = make_geometry(e,c,8);\n"
        "  switch (scenario) {\n"
        "    case 0: return primitive(e,c,components);\n"
        "    case 1: return prepared_primitive(geometry,component_record(components));\n"
        "    case 2: return prepared_primitive(geometry,313U << 6);\n"
        "    case 3: return prepared_primitive(geometry,6U);\n"
        "    case 4: return prepared_primitive(make_geometry(e,c,7),component_record(dddd));\n"
        "    case 5: return prepared_primitive(make_geometry(e,c,9),0U);\n"
        "  }\n"
        "  return 0.0;\n"
        "}\n"
    )
    library = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    loaded = ctypes.CDLL(str(library))
    loaded.evaluate.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3
    loaded.evaluate.restype = None
    loaded.evaluate_prepared.argtypes = [ctypes.POINTER(ctypes.c_double)] * 3 + [
        ctypes.c_uint
    ]
    loaded.evaluate_prepared.restype = None
    loaded.unsupported.argtypes = [ctypes.c_uint]
    loaded.unsupported.restype = ctypes.c_double
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


@pytest.mark.parametrize("scenario", range(6))
def test_invalid_prepared_inputs_return_nan(
    compiled_evaluator: ctypes.CDLL, scenario: int
) -> None:
    assert np.isnan(compiled_evaluator.unsupported(scenario))


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
