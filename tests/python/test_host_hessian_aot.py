"""Execute linked CPU scalar leaves independently of the molecular runtime."""

from __future__ import annotations

import ctypes as ct
import json
import subprocess
import sys
from dataclasses import replace
from fractions import Fraction
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.dft.native_semilocal import native_semilocal_record
from generativeqc_compiler.integral.expr import Graph
from generativeqc_compiler.method.host_hessian_aot import (
    HostArrayProgram,
    emit_registry,
    parse_inventory,
    programs,
)
from generativeqc_compiler.method.spec import resolve_method
from generativeqc_compiler.xc.program import output_set
from generativeqc_compiler.xc.semilocal_codegen import build_roots
from generativeqc_compiler.xc.spec import FunctionalSpec

ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "manifests/host_hessian_aot_b3lyp.json"
POINTER = ct.POINTER(ct.c_double)


@pytest.fixture(scope="module")
def linked(tmp_path_factory: Any, required_native_cxx: Any) -> tuple:
    """Compile the actual selected leaves and registry, with no runtime JIT."""
    directory = tmp_path_factory.mktemp("host-hessian-aot")
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    selected = programs(inventory)
    graph = Graph()
    argument = graph.variable("argument")
    lazy = HostArrayProgram(
        "lazy_probe",
        graph,
        (graph.select_le(argument, 0, 0, argument.pow(Fraction(1, 2))),),
        "test-lazy-branch",
    )
    constant = HostArrayProgram(
        "constant_probe", graph, (graph.constant(3),), "test-no-input"
    )
    spec = FunctionalSpec("B88 control", (("GGA_X_B88", Fraction(1)),))
    b88_graph, b88_roots, identity = build_roots(
        spec, output_set(spec, 2), production=True
    )
    b88 = HostArrayProgram("b88_probe", b88_graph, tuple(b88_roots), identity)
    entries = {entry.name: entry for entry in (*selected, lazy, constant, b88)}
    sources = []
    for entry in entries.values():
        source = directory / (entry.name + ".cpp")
        source.write_text(entry.emit())
        sources.append(source)
    registry = directory / "registry.cpp"
    registry.write_text(emit_registry(inventory))
    probe = directory / "probe.cpp"
    probe.write_text(
        '#include "dft/host_hessian_aot.hpp"\n'
        "using namespace generativeqc::dft::host_hessian_aot;\n"
        'extern "C" std::size_t count() { return programs().size(); }\n'
        'extern "C" int lookup(const char* name, const char* identity) {\n'
        "  const auto* entry = find(name, identity);\n"
        "  return entry ? static_cast<int>(entry->outputs) : -1;\n}\n"
        'extern "C" const char* input_name(const char* name, const char* identity, std::size_t index) {\n'
        "  const auto* entry = find(name, identity);\n"
        "  return entry && index < entry->inputs.size() ? entry->inputs[index].data() : nullptr;\n}\n"
        'extern "C" std::size_t binding_count() { return method_bindings().size(); }\n'
        'extern "C" std::size_t binding_field(const char* identity, unsigned field) {\n'
        "  const auto* entry = find_method(identity);\n"
        "  if (!entry) return 0;\n"
        "  return field == 0 ? entry->functional : field == 1 ? entry->scf_domain : entry->spins;\n"
        "}\n"
        'extern "C" double binding_exchange(const char* identity) {\n'
        "  const auto* entry = find_method(identity);\n"
        "  return entry ? entry->exact_exchange : -1;\n"
        "}\n"
        'extern "C" const char* binding_leaf(const char* identity, unsigned field) {\n'
        "  const auto* entry = find_method(identity);\n"
        "  if (!entry) return nullptr;\n"
        "  return (field == 3 ? entry->energy_mixed_identity : field == 2 ? entry->field_response_identity :\n"
        "          field == 1 ? entry->coefficient_response_identity : entry->xc_identity).data();\n"
        "}\n"
    )
    library = required_native_cxx.build_shared(
        [*sources, registry, probe],
        directory / "leaves.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(ROOT / "src"),
        ],
    )
    loaded = ct.CDLL(str(library))
    for name in entries:
        call = getattr(loaded, "gqc_host_hessian_" + name)
        call.argtypes = [POINTER, ct.c_size_t, POINTER, ct.c_size_t, ct.c_size_t]
        call.restype = ct.c_int
    loaded.count.restype = ct.c_size_t
    loaded.lookup.argtypes = [ct.c_char_p, ct.c_char_p]
    loaded.lookup.restype = ct.c_int
    loaded.input_name.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t]
    loaded.input_name.restype = ct.c_char_p
    loaded.binding_count.restype = ct.c_size_t
    loaded.binding_field.argtypes = [ct.c_char_p, ct.c_uint]
    loaded.binding_field.restype = ct.c_size_t
    loaded.binding_exchange.argtypes = [ct.c_char_p]
    loaded.binding_exchange.restype = ct.c_double
    loaded.binding_leaf.argtypes = [ct.c_char_p, ct.c_uint]
    loaded.binding_leaf.restype = ct.c_char_p
    return entries, loaded


def _evaluate(linked: tuple, name: str, values: dict) -> np.ndarray:
    """Bind independent point lanes in the registry's advertised input order."""
    entries, library = linked
    entry = entries[name]
    panel = np.ascontiguousarray([np.atleast_1d(values[key]) for key in entry.inputs])
    size = panel.shape[1]
    output = np.full((len(entry.roots), size), np.nan)
    call = getattr(library, "gqc_host_hessian_" + name)
    assert (
        call(
            panel.ctypes.data_as(POINTER),
            panel.size,
            output.ctypes.data_as(POINTER),
            output.size,
            size,
        )
        == 0
    )
    return output


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "unknown"},
        {"method": "PBE0"},
        {"runtime_python": True},
        {"spins": []},
        {"spins": ["polarized", "polarized"]},
        {"spins": [True]},
        {"arrays": ["xc", "xc"]},
        {"arrays": ["finite_difference"]},
        {"arrays": []},
    ],
)
def test_inventory_refuses_ignored_or_unsupported_controls(change: dict) -> None:
    with pytest.raises(ValueError):
        parse_inventory({**json.loads(INVENTORY.read_text()), **change})


def test_exact_identity_and_reachable_input_metadata(linked: tuple) -> None:
    entries, library = linked
    selected = programs(parse_inventory(json.loads(INVENTORY.read_text())))
    assert library.count() == len(selected)
    for entry in selected:
        name, identity = entry.name.encode(), entry.identity.encode()
        assert library.lookup(name, identity) == len(entry.roots)
        assert library.lookup(name, b"wrong identity") == -1
        assert library.lookup(b"absent", identity) == -1
        assert [
            library.input_name(name, identity, index).decode()
            for index in range(len(entry.inputs))
        ] == list(entry.inputs)
        assert library.input_name(name, identity, len(entry.inputs)) is None
    assert entries["xc_polarized"].identity != entries["xc_unpolarized"].identity


def test_exact_method_binding_authenticates_both_linked_programs(linked: tuple) -> None:
    entries, library = linked
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert library.binding_count() == len(inventory.spins)
    for spin in inventory.spins:
        method = resolve_method(inventory.method, spin=spin)
        identity = method.identity.encode()
        functional = next(
            primitive.functional
            for primitive in method.primitives
            if primitive.kind == "semilocal_xc"
        )
        record = native_semilocal_record(functional, spin=spin)
        assert [library.binding_field(identity, field) for field in range(3)] == [
            record["code"],
            record["domain_version"],
            1 if spin == "unpolarized" else 2,
        ]
        assert library.binding_exchange(identity) == float(
            Fraction(record["cuda_global_hybrid_exact_exchange"])
        )
        for field, prefix in enumerate(
            ("xc_", "coefficient_response_", "gga_field_response_", "gga_energy_mixed_")
        ):
            assert (
                library.binding_leaf(identity, field).decode()
                == entries[prefix + spin].identity
            )
    assert library.binding_field(b"absent", 0) == 0
    assert library.binding_exchange(b"absent") == -1
    assert library.binding_leaf(b"absent", 0) is None
    assert library.binding_leaf(b"B3LYP", 0) is None


def test_checked_abi_refuses_before_pointer_access(linked: tuple) -> None:
    entries, library = linked
    for name, entry in entries.items():
        call = getattr(library, "gqc_host_hessian_" + name)
        assert call(None, 0, None, 0, 0) == 0
        assert call(None, 1, None, 0, 0) < 0
        assert call(None, len(entry.inputs), None, len(entry.roots), 1) < 0
        sentinel = (ct.c_double * 1)(-9182.5)
        assert call(None, 0, sentinel, len(entry.roots) + 1, 1) < 0
        if entry.inputs:
            assert call(None, len(entry.inputs), sentinel, len(entry.roots), 1) < 0
        size = (ct.c_size_t(-1).value // ct.sizeof(ct.c_double)) // max(
            1, len(entry.inputs), len(entry.roots)
        ) + 1
        assert call(None, 0, sentinel, 1, size) < 0
        assert (
            call(
                sentinel,
                size * len(entry.inputs),
                sentinel,
                size * len(entry.roots),
                size,
            )
            < 0
        )
        assert sentinel[0] == -9182.5


def test_lazy_branches_input_free_roots_and_nonfinite_failure(linked: tuple) -> None:
    np.testing.assert_array_equal(
        _evaluate(linked, "lazy_probe", {"argument": [-1.0, 0.0, 4.0]}),
        [[0.0, 0.0, 2.0]],
    )
    _, library = linked
    output = (ct.c_double * 3)(0, 0, 0)
    assert library.gqc_host_hessian_constant_probe(None, 0, output, 3, 3) == 0
    assert list(output) == [3.0] * 3
    argument = (ct.c_double * 1)(float("inf"))
    assert library.gqc_host_hessian_lazy_probe(argument, 1, output, 1, 1) > 0


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_physical_b3lyp_roots_match_independent_libxc(linked: tuple, spin: str) -> None:
    """Use Libxc only as an independent test oracle, never in source generation."""
    libxc = pytest.importorskip("pyscf.dft.libxc")
    alpha = np.array(
        [[0.4, 0.7, 0.2], [0.03, 0.01, -0.02], [0.06, -0.04, 0.03], [0.02, 0.05, 0.07]]
    )
    beta = np.array(
        [[0.2, 0.1, 0.3], [-0.02, 0.04, 0.01], [0.04, 0.03, -0.02], [0.01, -0.01, 0.06]]
    )
    polarized = spin == "polarized"
    rho = np.array([alpha, beta]) if polarized else alpha + beta
    exc, vxc, fxc, _ = libxc.eval_xc(
        ".08*LDA_X+.72*GGA_X_B88+.19*LDA_C_VWN_RPA+.81*GGA_C_LYP",
        rho,
        spin=int(polarized),
        deriv=2,
    )
    if polarized:
        values = {
            "rho_a": alpha[0],
            "rho_b": beta[0],
            "sigma_aa": np.sum(alpha[1:] ** 2, axis=0),
            "sigma_ab": np.sum(alpha[1:] * beta[1:], axis=0),
            "sigma_bb": np.sum(beta[1:] ** 2, axis=0),
        }
        gradient = np.concatenate((vxc[0].T, vxc[1].T, np.zeros((2, 3))))
        hessian = np.zeros((7, 7, 3))
        for index, (left, right) in enumerate(
            combinations_with_replacement(range(2), 2)
        ):
            hessian[left, right] = hessian[right, left] = fxc[0][:, index]
        for index, (left, right) in enumerate(
            combinations_with_replacement(range(2, 5), 2)
        ):
            hessian[left, right] = hessian[right, left] = fxc[2][:, index]
        for left in range(2):
            for right in range(3):
                hessian[left, right + 2] = hessian[right + 2, left] = fxc[1][
                    :, 3 * left + right
                ]
        density = alpha[0] + beta[0]
    else:
        values = {"rho": rho[0], "sigma": np.sum(rho[1:] ** 2, axis=0)}
        gradient = np.array([vxc[0], vxc[1], np.zeros(3)])
        hessian = np.zeros((3, 3, 3))
        hessian[0, 0], hessian[0, 1], hessian[1, 0], hessian[1, 1] = (
            fxc[0],
            fxc[1],
            fxc[1],
            fxc[2],
        )
        density = rho[0]
    expected = np.array(
        [
            exc * density,
            *gradient,
            *(
                hessian[left, right]
                for left, right in combinations_with_replacement(
                    range(len(gradient)), 2
                )
            ),
        ]
    )
    np.testing.assert_allclose(
        _evaluate(linked, "xc_" + spin, values), expected, atol=2e-12, rtol=2e-11
    )


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_coefficient_pullback_and_response_chain_rule(linked: tuple, spin: str) -> None:
    """Check both spin layouts against the elementary sigma Jacobian."""
    generator = np.random.default_rng(2844)
    channels = 2 if spin == "polarized" else 1
    gradient = generator.normal(size=(channels, 3, 4))
    moving = generator.normal(size=gradient.shape)
    feature_gradient = generator.normal(size=(5 if channels == 2 else 2, 4))
    feature_response = generator.normal(size=feature_gradient.shape)
    values = {f"v{index}": row for index, row in enumerate(feature_gradient)}
    values.update({f"dv{index}": row for index, row in enumerate(feature_response)})
    for channel in range(channels):
        for axis in range(3):
            values[f"g{channel}_{axis}"] = gradient[channel, axis]
            values[f"dg{channel}_{axis}"] = moving[channel, axis]
    expected, response = [], []
    for channel in range(channels):
        sigma_index = 1 if channels == 1 else 2 + 2 * channel
        vector = 2 * feature_gradient[sigma_index] * gradient[channel]
        delta = 2 * (
            feature_response[sigma_index] * gradient[channel]
            + feature_gradient[sigma_index] * moving[channel]
        )
        if channels == 2:
            vector += feature_gradient[3] * gradient[1 - channel]
            delta += (
                feature_response[3] * gradient[1 - channel]
                + feature_gradient[3] * moving[1 - channel]
            )
        expected.extend((feature_gradient[channel], *vector))
        response.extend((feature_response[channel], *delta))
    np.testing.assert_allclose(
        _evaluate(linked, "coefficients_" + spin, values), expected, atol=2e-14
    )
    np.testing.assert_allclose(
        _evaluate(linked, "coefficient_response_" + spin, values), response, atol=2e-14
    )


def test_integral_direction_keeps_independent_slots_and_projection(
    linked: tuple,
) -> None:
    """Ordered mathematical centers are projected only after their AD gradients."""
    rng = np.random.default_rng(2844)
    gradients = rng.normal(size=(12, 5))
    directions = rng.normal(size=(12, 5))
    projection = np.array([0.0, -0.17, 0.43, 1.0, 2.1])
    values = {f"g{index}": value for index, value in enumerate(gradients)}
    values.update({f"u{index}": value for index, value in enumerate(directions)})
    values["projection"] = projection
    expected = projection * np.sum(gradients * directions, axis=0)
    np.testing.assert_allclose(
        _evaluate(linked, "integral_direction", values)[0],
        expected,
        atol=3e-15,
        rtol=3e-15,
    )


def test_nuclear_pair_hvp_matches_coulomb_tensor(linked: tuple) -> None:
    displacement = np.array([[1.3, 0.2], [-0.4, 1.7], [0.9, -0.8]])
    direction = np.array([[0.2, -0.7], [0.1, 0.3], [-0.3, 0.8]])
    charge = np.array([4.0, -2.0])
    distance = np.linalg.norm(displacement, axis=0)
    expected = charge * (
        3 * displacement * np.sum(displacement * direction, axis=0) / distance**5
        - direction / distance**3
    )
    values = {"charge_product": charge}
    for axis in range(3):
        values[f"r{axis}"] = displacement[axis]
        values[f"dr{axis}"] = direction[axis]
    np.testing.assert_allclose(
        _evaluate(linked, "nuclear_pair_hvp", values), expected, atol=2e-14
    )


@pytest.mark.parametrize("kind", ["norm", "ratio", "log", "becke"])
def test_grid_mixed_roots_match_independent_primal(linked: tuple, kind: str) -> None:
    """Include upstream mixed tangents, not merely two independent JVPs."""
    precision = pytest.importorskip("mpmath").mp.clone()
    precision.dps = 60
    names = {
        "norm": ("x", "y", "z"),
        "ratio": ("a", "b"),
        "log": ("p",),
        "becke": ("mu",),
    }[kind]
    values = {}
    for index, name in enumerate(names):
        values[name] = np.array([0.3 * (index + 1), 0.2 * (index + 2)])
        values["l" + name] = np.array([-0.2 * (index + 1) ** 2, 0.1 / (index + 2)])
        values["r" + name] = np.array([0.15 / (index + 1), -0.17 * (index + 1)])
        values["lr" + name] = np.array(
            [0.07 * (index + 1) ** 3, -0.03 * (index + 2) ** 2]
        )
    expected = []
    for lane in range(2):

        def independent(left: object, right: object, lane: int = lane) -> object:
            arguments = [
                precision.mpf(values[name][lane])
                + left * precision.mpf(values["l" + name][lane])
                + right * precision.mpf(values["r" + name][lane])
                + left * right * precision.mpf(values["lr" + name][lane])
                for name in names
            ]
            if kind == "norm":
                return precision.sqrt(sum(value**2 for value in arguments))
            if kind == "ratio":
                return arguments[0] / arguments[1]
            if kind == "log":
                return precision.log(arguments[0])
            coordinate = arguments[0]
            for _ in range(3):
                coordinate = coordinate * (3 - coordinate**2) / 2
            return (1 - coordinate) / 2

        expected.append(
            [
                float(precision.diff(independent, (0, 0), orders))
                for orders in ((0, 0), (1, 0), (0, 1), (1, 1))
            ]
        )
    np.testing.assert_allclose(
        _evaluate(linked, "grid_mixed_" + kind, values),
        np.asarray(expected).T,
        atol=2e-14,
        rtol=2e-13,
    )


@pytest.mark.parametrize(
    "density,sigma",
    [(7.4225911645623816e-9, 5.024e-11), (7.957986101616474e-8, 5.395e-10)],
)
def test_minority_b88_feature_derivatives_match_independent_mp(
    linked: tuple, density: float, sigma: float
) -> None:
    """Differentiate the original B88 primal at 80 digits, not emitted AD."""
    precision = pytest.importorskip("mpmath").mp.clone()
    precision.dps = 80
    values = {
        "rho_a": 0.11,
        "rho_b": density,
        "sigma_aa": 0.01,
        "sigma_ab": 0.0,
        "sigma_bb": sigma,
    }
    arguments = [precision.mpf(value) for value in values.values()]

    def independent(*features: object) -> object:
        exchange = precision.mpf(3) / 4 * (6 / precision.pi) ** (precision.mpf(1) / 3)
        beta = precision.mpf(".0042")
        result = precision.mpf(0)
        for rho_index, sigma_index in ((0, 2), (1, 4)):
            rho, invariant = features[rho_index], features[sigma_index]
            reduced = precision.sqrt(invariant) / rho ** (precision.mpf(4) / 3)
            result -= rho ** (precision.mpf(4) / 3) * (
                exchange
                + beta
                * reduced**2
                / (1 + 6 * beta * reduced * precision.asinh(reduced))
            )
        return result

    spec = FunctionalSpec("B88 control", (("GGA_X_B88", Fraction(1)),))
    expected = []
    for output in output_set(spec, 2):
        if any(index >= 5 for index in output):
            expected.append(0.0)
        else:
            orders = tuple(output.count(index) for index in range(5))
            expected.append(
                float(precision.diff(independent, tuple(arguments), orders))
            )
    np.testing.assert_allclose(
        _evaluate(linked, "b88_probe", values)[:, 0], expected, atol=2e-12, rtol=8e-13
    )


def test_generation_blocks_runtime_and_reference_imports(tmp_path: Path) -> None:
    """Run the real generator in a process refusing forbidden module imports."""
    script = (
        "import importlib.abc, runpy, sys\n"
        "class Guard(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in {'generativeqc', 'pyscf', 'cupy', 'torch'}:\n"
        "   raise RuntimeError('forbidden build-time dependency: ' + fullname)\n"
        "sys.meta_path.insert(0, Guard())\n"
        f"sys.argv = ['generator', '--inventory', {str(INVENTORY)!r}, '--all', '--output-dir', {str(tmp_path)!r}]\n"
        f"runpy.run_path({str(ROOT / 'tools/generate_host_hessian_aot.py')!r}, run_name='__main__')\n"
    )
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    inventory = parse_inventory(json.loads(INVENTORY.read_text()))
    assert {path.name for path in tmp_path.glob("*.cpp")} == {
        entry.name + ".cpp" for entry in programs(inventory)
    } | {"registry.cpp"}
    reversed_inventory = replace(inventory, arrays=tuple(reversed(inventory.arrays)))
    assert emit_registry(reversed_inventory) == emit_registry(inventory)
