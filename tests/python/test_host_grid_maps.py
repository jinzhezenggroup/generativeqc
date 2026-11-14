"""Compiled shared contractions against direct TensorIR and scalar stencils."""

from __future__ import annotations

import ctypes as ct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc_compiler.method.stationary_grid_second import (
    feature_mixed_program,
    feature_programs,
)
from generativeqc_compiler.method.wb97mv_nuclear_grid import nuclear_grid_maps
from generativeqc_compiler.tensor import Program, linearize, transpose_program
from generativeqc_compiler.tensor.host_affine import emit_affine_host
from generativeqc_compiler.tensor.interpreter import execute

from tools.generate_host_grid_maps import generate, generate_curvature

POINTER = ct.POINTER(ct.c_double)
LABELS = ("rho", "g0", "g1", "g2", "tau")
AOS = ("ao0", "ao1", "ao2", "ao3")


class Array(ct.Structure):
    """Borrow one contiguous, explicitly sized FP64 input or output."""

    _fields_ = [("data", POINTER), ("elements", ct.c_size_t)]


def _view(array: np.ndarray) -> Array:
    return Array(array.ctypes.data_as(POINTER), array.size)


def _direct_programs() -> tuple[dict[str, Program], dict[str, Program]]:
    primal, first = feature_programs(3, 2)
    fields = Program({name: primal.outputs[name] for name in LABELS[:4]})
    geometric = linearize(primal, AOS).program
    pullback = transpose_program(fields, LABELS[:4], inputs=AOS).program
    relaxed = linearize(fields, ("density", *AOS)).program
    mixed = feature_mixed_program(3, 2, density_response=False)
    return (
        {
            "features": primal,
            "feature_direction": Program(
                {name: first.outputs[f"d_{name}"] for name in LABELS}
            ),
            "potential_direction": nuclear_grid_maps()["potential_direction"],
        },
        {
            "features": fields,
            "first": Program(
                {name: geometric.outputs[f"d_{name}"] for name in LABELS[:4]}
            ),
            "mixed": Program(
                {name: mixed.outputs[f"d_d_{name}"] for name in LABELS[:4]}
            ),
            "pullback": pullback,
            "pullback_direction": linearize(pullback, AOS).program,
            "first_relaxed": Program(
                {name: relaxed.outputs[f"d_{name}"] for name in LABELS[:4]}
            ),
            "pullback_direction_relaxed": linearize(
                pullback, ("density", *AOS)
            ).program,
        },
    )


@pytest.fixture(scope="module", params=[False, True], ids=["native", "scalar-fallback"])
def native(tmp_path_factory: Any, required_native_cxx: Any, request: Any) -> Any:
    """Compile actual generated kernels in isolated test-only namespaces."""
    directory = tmp_path_factory.mktemp("host-grid-maps")
    direct = _direct_programs()
    sources = []
    variants = {}
    for family, generator in enumerate((generate, generate_curvature)):
        for variant in ("direct", "scalar", "shared"):
            name = f"{variant}_{family}"
            namespace = "test_" + name
            source = (
                generator(ordered_binary_contractions=variant == "shared").replace(
                    "generativeqc_host_grid_maps"
                    if family == 0
                    else "generativeqc_host_curvature_maps",
                    namespace,
                )
                if variant != "direct"
                else emit_affine_host(
                    direct[family],
                    namespace=namespace,
                    extents={"point": "points", "ao": "nao"},
                )
            )
            header = directory / (name + ".hpp")
            header.write_text(source)
            probe = directory / (name + ".cpp")
            probe.write_text(f'''
#include "{header.name}"
#include <vector>
namespace maps = {namespace};
struct Array {{ double* data; std::size_t elements; }};
static const maps::Program* entry(const char* name) {{
  for (const auto& program : maps::programs()) if (program.name == name) return &program;
  return nullptr;
}}
extern "C" int {name}_query(const char* name, std::size_t basis, std::size_t points,
                            std::size_t* bytes, std::size_t* terms) {{
  const auto* program = entry(name);
  return program && program->workspace({{basis, points}}, *bytes) &&
         program->work({{basis, points}}, *terms) ? 0 : -1;
}}
extern "C" const char* {name}_label(const char* name, int output, std::size_t index) {{
  const auto* program = entry(name);
  if (!program) return nullptr;
  const auto labels = output ? program->outputs : program->inputs;
  return index < labels.size() ? labels[index].data() : nullptr;
}}
extern "C" int {name}_apply(const char* name, std::size_t basis, std::size_t points,
                            const Array* inputs, std::size_t input_count,
                            const Array* outputs, std::size_t output_count,
                            double* arena, std::size_t arena_count) {{
  const auto* program = entry(name);
  if (!program) return -1;
  std::vector<maps::Input> source;
  std::vector<maps::Output> target;
  for (std::size_t index = 0; index < input_count; ++index)
    source.push_back({{inputs[index].data, inputs[index].elements}});
  for (std::size_t index = 0; index < output_count; ++index)
    target.push_back({{outputs[index].data, outputs[index].elements}});
  return static_cast<int>(program->evaluate({{basis, points}}, source, target,
                                           {{arena, arena_count}}));
}}
''')
            sources.append(probe)
            variants[name] = direct[family]
    library = required_native_cxx.build_shared(
        sources,
        directory / "maps.so",
        compile_args=[
            "-std=c++20",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
            *(["-DGQC_ORDERED_BINARY_FORCE_SCALAR=1"] if request.param else []),
        ],
        compile_timeout=300,
    )
    loaded = ct.CDLL(str(library))
    for name in variants:
        getattr(loaded, name + "_query").argtypes = [
            ct.c_char_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(ct.c_size_t),
            ct.POINTER(ct.c_size_t),
        ]
        getattr(loaded, name + "_label").argtypes = [ct.c_char_p, ct.c_int, ct.c_size_t]
        getattr(loaded, name + "_label").restype = ct.c_char_p
        getattr(loaded, name + "_apply").argtypes = [
            ct.c_char_p,
            ct.c_size_t,
            ct.c_size_t,
            ct.POINTER(Array),
            ct.c_size_t,
            ct.POINTER(Array),
            ct.c_size_t,
            POINTER,
            ct.c_size_t,
        ]
    return loaded, direct


@pytest.mark.parametrize(
    "basis,points", [(0, 5), (1, 0), (1, 1), (4, 8), (7, 11), (17, 19), (81, 257)]
)
@pytest.mark.parametrize("family", [0, 1])
def test_ordered_maps_preserve_serial_bits_and_resources(
    native: Any, family: int, basis: int, points: int
) -> None:
    """Compare identical shared graphs, including signed nonsymmetric D and D'."""
    loaded, direct = native
    for name, program in direct[family].items():
        feeds = _feeds(program, basis, points)
        status, expected, terms = _call(
            native, f"scalar_{family}", name, program, feeds, basis, points
        )
        assert status == 0
        status, actual, scheduled_terms = _call(
            native, f"shared_{family}", name, program, feeds, basis, points
        )
        assert status == 0 and terms == scheduled_terms
        for output in expected:
            np.testing.assert_array_equal(
                actual[output].view(np.uint64), expected[output].view(np.uint64)
            )
        sizes = []
        for variant in ("scalar", "shared"):
            byte_count, work = ct.c_size_t(), ct.c_size_t()
            assert (
                getattr(loaded, f"{variant}_{family}_query")(
                    name.encode(), basis, points, ct.byref(byte_count), ct.byref(work)
                )
                == 0
            )
            sizes.append(byte_count.value)
        assert sizes[0] == sizes[1]


@pytest.mark.parametrize("basis", [4, 17, 81])
def test_ordered_primal_preserves_vacuum_boundary(native: Any, basis: int) -> None:
    """A SIMD schedule must not change the scalar closure's 1e-18 branch."""
    _, direct = native
    program = direct[0]["features"]
    for seed in range(4):
        feeds = _feeds(program, basis, 5)
        rng = np.random.default_rng(2848 + seed)
        factor = rng.normal(size=(basis, basis))
        density = factor @ factor.T / basis + np.eye(basis) * 0.4
        feeds["density"] = np.ascontiguousarray(density)
        status, initial, _ = _call(
            native, "scalar_0", "features", program, feeds, basis, 5
        )
        assert status == 0
        central = 1e-18 / initial["rho"][0]
        scales = [central]
        left = right = central
        for _ in range(16):
            left, right = np.nextafter(left, 0), np.nextafter(right, np.inf)
            scales.extend((left, right))
        for scale in scales:
            feeds["density"] = np.ascontiguousarray(scale * density)
            status, expected, _ = _call(
                native, "scalar_0", "features", program, feeds, basis, 5
            )
            scheduled_status, actual, _ = _call(
                native, "shared_0", "features", program, feeds, basis, 5
            )
            assert status == scheduled_status == 0
            for output in expected:
                np.testing.assert_array_equal(
                    actual[output].view(np.uint64), expected[output].view(np.uint64)
                )
            np.testing.assert_array_equal(
                actual["rho"] <= 1e-18, expected["rho"] <= 1e-18
            )


def _shape(node: Any, basis: int, points: int) -> tuple[int, ...]:
    extents = {"point": points, "ao": basis}
    return tuple(extents[axis.space.name] for axis in node.spec.indices)


def _feeds(program: Program, basis: int, points: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(28442848 + basis + points)
    return {
        node.attrs["name"]: np.ascontiguousarray(
            rng.normal(size=_shape(node, basis, points))
        )
        for node in program.nodes
        if node.op == "input"
    }


def _call(
    native: Any,
    variant: str,
    name: str,
    program: Program,
    feeds: dict[str, np.ndarray],
    basis: int,
    points: int,
    shortage: bool = False,
) -> tuple[int, dict[str, np.ndarray], int]:
    loaded, _ = native
    encoded = name.encode()
    sizes = ct.c_size_t(), ct.c_size_t()
    assert (
        getattr(loaded, variant + "_query")(
            encoded, basis, points, *(ct.byref(size) for size in sizes)
        )
        == 0
    )
    arena = np.empty(sizes[0].value // 8)
    outputs = {
        label: np.full(_shape(node, basis, points), -781.25)
        for label, node in program.outputs.items()
    }
    label = getattr(loaded, variant + "_label")
    source = []
    for index in range(len(feeds)):
        found = label(encoded, 0, index)
        if found is None:
            break
        source.append(_view(feeds[found.decode()]))
    target = []
    for index in range(len(outputs)):
        target.append(_view(outputs[label(encoded, 1, index).decode()]))
    status = getattr(loaded, variant + "_apply")(
        encoded,
        basis,
        points,
        (Array * len(source))(*source),
        len(source),
        (Array * len(target))(*target),
        len(target),
        arena.ctypes.data_as(POINTER),
        max(0, arena.size - int(shortage)),
    )
    return status, outputs, sizes[1].value


@pytest.mark.parametrize(
    "basis,points", [(0, 0), (0, 5), (1, 0), (1, 1), (3, 5), (7, 11)]
)
@pytest.mark.parametrize("family", [0, 1])
def test_compiled_shared_maps_match_direct_contractions(
    native: Any, family: int, basis: int, points: int
) -> None:
    """All maps retain signed D/D', cotangents and bounded empty/small layouts."""
    _, direct = native
    for name, program in direct[family].items():
        feeds = _feeds(program, basis, points)
        status, expected, _ = _call(
            native, f"direct_{family}", name, program, feeds, basis, points
        )
        assert status == 0
        status, actual, _ = _call(
            native, f"shared_{family}", name, program, feeds, basis, points
        )
        assert status == 0
        for output in expected:
            np.testing.assert_allclose(
                actual[output], expected[output], atol=3e-12, rtol=3e-12
            )


@pytest.mark.parametrize("family", [0, 1])
def test_compiled_shared_map_caps_and_nonfinite_inputs_are_transactional(
    native: Any, family: int
) -> None:
    _, direct = native
    for name, program in direct[family].items():
        feeds = _feeds(program, 3, 5)
        status, outputs, _ = _call(
            native, f"shared_{family}", name, program, feeds, 3, 5, shortage=True
        )
        assert status != 0
        for output in outputs.values():
            np.testing.assert_array_equal(output, -781.25)
        for panel in feeds.values():
            original = panel.flat[0]
            panel.flat[0] = np.nan
            status, outputs, _ = _call(
                native, f"shared_{family}", name, program, feeds, 3, 5
            )
            assert status != 0
            for output in outputs.values():
                np.testing.assert_array_equal(output, -781.25)
            panel.flat[0] = original


def test_runtime_work_bound_drops_without_a_relaxed_cap(native: Any) -> None:
    """Price the real six-atom shape without allocating AO/point numerical data."""
    loaded, _ = native
    scheduled = (
        ("features", "feature_direction", "potential_direction"),
        (
            "features",
            "first_relaxed",
            "pullback",
            "pullback_direction_relaxed",
            "pullback",
        ),
    )
    totals = []
    for variant in ("direct", "shared"):
        total = 0
        for family, names in enumerate(scheduled):
            query = getattr(loaded, f"{variant}_{family}_query")
            for begin in range(0, 87458, 256):
                for name in names:
                    byte_count, terms = ct.c_size_t(), ct.c_size_t()
                    assert (
                        query(
                            name.encode(),
                            81,
                            min(256, 87458 - begin),
                            ct.byref(byte_count),
                            ct.byref(terms),
                        )
                        == 0
                    )
                    total += terms.value
        totals.append(18 * total)
    assert totals[1] < (1 << 40) < totals[0]


def test_shared_primal_and_jvp_match_independent_scalar_stencils() -> None:
    """The optimized compiler map is not its own numerical derivative oracle."""
    shared, first = feature_programs(5, 7, shared_contractions=True)
    rng = np.random.default_rng(2844)
    density = rng.normal(size=(7, 7))
    density = density + density.T
    jets = rng.normal(size=(4, 5, 7))
    delta_density = rng.normal(size=(7, 7))
    delta_jets = rng.normal(size=jets.shape)
    feeds = {"density": density, "d_density": delta_density}
    feeds.update({name: jets[index] for index, name in enumerate(AOS)})
    feeds.update({"d_" + name: delta_jets[index] for index, name in enumerate(AOS)})

    def scalar(motion: float) -> dict[str, np.ndarray]:
        moved, matrix = jets + motion * delta_jets, density + motion * delta_density
        bilinear = lambda left, right: np.einsum(
            "pi,ij,pj->p", left, matrix, right, optimize=False
        )
        return {
            "rho": bilinear(moved[0], moved[0]),
            **{
                f"g{axis}": bilinear(moved[axis + 1], moved[0])
                + bilinear(moved[0], moved[axis + 1])
                for axis in range(3)
            },
            "tau": sum(bilinear(panel, panel) for panel in moved[1:]) / 2,
        }

    primal = execute(shared, feeds).outputs
    tangent = execute(first, feeds).outputs
    for name, expected in scalar(0).items():
        np.testing.assert_allclose(primal[name], expected, atol=3e-12, rtol=3e-12)
    errors = []
    for step in (1e-2, 3e-3, 1e-3):
        plus, minus = scalar(step), scalar(-step)
        errors.append(
            max(
                float(
                    np.max(
                        abs(
                            (plus[name] - minus[name]) / (2 * step)
                            - tangent["d_" + name]
                        )
                    )
                )
                for name in LABELS
            )
        )
    assert errors[-1] < errors[0] / 50
    assert errors[-1] < 1e-4


@pytest.mark.parametrize("policy", [None, 0, "shared"])
def test_shared_feature_schedule_requires_an_explicit_boolean(policy: object) -> None:
    with pytest.raises(TypeError, match="shared_contractions"):
        feature_programs(3, 2, shared_contractions=policy)
