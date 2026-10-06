"""Independent runtime axes must stay local without Cartesian map expansion."""

from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.dft.cuda import CudaGrid, DeviceGridTask, GridTaskView
from generativeqc_compiler.dft.indexed_layout import AoGridBlockLayout
from generativeqc_compiler.method.indexed_grid import AoGridBlockProgram
from generativeqc_compiler.tensor import (
    Index,
    IndexedTensorLayout,
    IndexSpace,
    PrecisionDirective,
    Program,
    TensorSpec,
    dot_test,
    execute,
    input_tensor,
    linearize,
    lower_precision,
    runtime_cartesian_select,
    transpose_program,
)
from generativeqc_compiler.tensor.cuda_emit import emit_cuda
from generativeqc_compiler.tensor.cuda_plan import plan_cuda

if TYPE_CHECKING:
    from pathlib import Path


def _layout(
    active: int = 3, indexed: bool = True, order: int = 2, map_order: int | None = 2
) -> AoGridBlockLayout:
    return AoGridBlockLayout(5, active, 2, order, "basis-geometry", indexed, map_order)


def _density() -> np.ndarray:
    return np.arange(50, dtype=np.float64).reshape(2, 5, 5)


def _scatter_oracle(local: np.ndarray, ids: tuple[int, ...] | np.ndarray) -> np.ndarray:
    result = np.zeros((2, 5, 5))
    for spin in range(2):
        for row, global_row in enumerate(ids):
            for column, global_column in enumerate(ids):
                result[spin, global_row, global_column] += local[spin, row, column]
    return result


@pytest.mark.parametrize("ids", [(), (2,), (4, 1, 3), (4, 1, 4), (0, 1, 2, 3, 4)])
def test_cartesian_density_gather_and_scatter_share_one_runtime_map(
    ids: tuple[int, ...],
) -> None:
    layout = _layout(len(ids))
    mapping = np.asarray(ids, dtype=np.int64)
    density = _density()
    program = AoGridBlockProgram(layout).density_program()
    gathered = execute(program, {"density": density, "ao_ids": mapping}).outputs[
        "local_density"
    ]
    expected = density[:, mapping[:, None], mapping[None, :]]
    np.testing.assert_array_equal(gathered, expected)
    assert gathered.shape == (2, len(ids), len(ids))
    assert AoGridBlockProgram(layout).density_layout.map_bytes == len(ids) * 8
    plan = plan_cuda(program, cuda_target_info("sm_120"))
    assert len(plan.inputs) == 2
    assert plan.index_tables == ()
    assert len(plan.steps) == 3
    replay = Program.loads(program.dumps())
    assert replay.logical_hash == program.logical_hash
    np.testing.assert_array_equal(
        execute(replay, {"density": density, "ao_ids": mapping}).outputs[
            "local_density"
        ],
        expected,
    )
    local = np.arange(2 * len(ids) ** 2, dtype=np.float64).reshape(
        2, len(ids), len(ids)
    )
    scattered = execute(
        AoGridBlockProgram(layout).scatter_program(),
        {"local_potential": local, "ao_ids": mapping},
    ).outputs["potential"]
    np.testing.assert_array_equal(scattered, _scatter_oracle(local, ids))


@pytest.mark.parametrize("jets", [1, 4])
@pytest.mark.parametrize("ids", [(), (2,), (4, 1, 3), (4, 1, 4)])
def test_local_projection_matches_independent_numpy_contraction(
    jets: int, ids: tuple[int, ...]
) -> None:
    layout = _layout(len(ids))
    mapping = np.asarray(ids, dtype=np.int64)
    density = _density() / 17
    ao = np.arange(layout.ao_jet_values, dtype=np.float64).reshape(10, 2, len(ids)) / 31
    result = execute(
        AoGridBlockProgram(layout).projection_program(jets),
        {
            "density": density,
            "ao_ids": mapping,
            "ao_jets": ao,
        },
    ).outputs["projected"]
    expected = np.einsum(
        "jpm,smn->sjpn", ao[:jets], density[:, mapping[:, None], mapping[None, :]]
    )
    np.testing.assert_allclose(result, expected, atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("scatter", [False, True])
def test_cartesian_reference_and_generated_adjoints_accumulate_repeats(
    scatter: bool,
) -> None:
    layout = _layout()
    ids = np.array([4, 1, 4], dtype=np.int64)
    if scatter:
        program, source_name, output_name = (
            AoGridBlockProgram(layout).scatter_program(),
            "local_potential",
            "potential",
        )
        source = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 17
        cotangent = _density() / 31
    else:
        program, source_name, output_name = (
            AoGridBlockProgram(layout).density_program(),
            "density",
            "local_density",
        )
        source = _density() / 17
        cotangent = np.arange(18, dtype=np.float64).reshape(2, 3, 3) / 31
    feeds = {source_name: source, "ao_ids": ids}
    tangent = np.linspace(-1, 1, source.size).reshape(source.shape)
    result = dot_test(program, feeds, {source_name: tangent}, {output_name: cotangent})
    assert result.passed, result
    forward = linearize(program, [source_name])
    actual_forward = execute(
        forward.program, {**feeds, "d_" + source_name: tangent}
    ).outputs["d_" + output_name]
    expected_forward = (
        _scatter_oracle(tangent, ids)
        if scatter
        else tangent[:, ids[:, None], ids[None, :]]
    )
    np.testing.assert_array_equal(actual_forward, expected_forward)
    reverse = transpose_program(program, [output_name], inputs=[source_name])
    actual_reverse = execute(
        reverse.program, {**feeds, "bar_" + output_name: cotangent}
    ).outputs["bar_" + source_name]
    expected_reverse = (
        cotangent[:, ids[:, None], ids[None, :]]
        if scatter
        else _scatter_oracle(cotangent, ids)
    )
    np.testing.assert_array_equal(actual_reverse, expected_reverse)


def test_cartesian_independent_extents_and_unselected_domains() -> None:
    global_axes = tuple(
        Index(name, IndexSpace(name + "_space", "matrix", size))
        for name, size in zip(("batch", "row", "column"), (2, 5, 4), strict=True)
    )
    row = Index("local_row", IndexSpace("row_local", "matrix", 3))
    column = Index("local_column", IndexSpace("column_local", "matrix", 2))
    row_map = input_tensor(
        "row_ids", TensorSpec((row,), dtype="int64", role="input", differentiable=False)
    )
    column_map = input_tensor(
        "column_ids",
        TensorSpec((column,), dtype="int64", role="input", differentiable=False),
    )
    spec = TensorSpec(global_axes, role="input")
    layout = IndexedTensorLayout(spec, ((1, row_map, row), (2, column_map, column)))
    source = input_tensor("source", spec)
    program = Program({"local": layout.select(source)})
    feeds = {
        "source": np.arange(40, dtype=np.float64).reshape(2, 5, 4),
        "row_ids": np.array([4, 1, 4]),
        "column_ids": np.array([3, 0]),
    }
    expected = feeds["source"][
        :, feeds["row_ids"][:, None], feeds["column_ids"][None, :]
    ]
    np.testing.assert_array_equal(execute(program, feeds).outputs["local"], expected)
    assert layout.map_bytes == (3 + 2) * 8
    assert layout.local_spec.indices[0].domain == global_axes[0].domain


def test_cartesian_layout_fail_closed_and_emitted_index_checks() -> None:
    layout = _layout()
    program = AoGridBlockProgram(layout).density_program()
    for ids in ([0, 1, 5], [0, -1, 2]):
        with pytest.raises(ValueError, match="outside"):
            execute(
                program,
                {"density": _density(), "ao_ids": np.array(ids, dtype=np.int64)},
            )
        with pytest.raises(ValueError, match="outside"):
            execute(
                AoGridBlockProgram(layout).scatter_program(),
                {
                    "local_potential": np.zeros((2, 3, 3)),
                    "ao_ids": np.array(ids, dtype=np.int64),
                },
            )
    with pytest.raises(ValueError, match="int64"):
        execute(program, {"density": _density(), "ao_ids": np.array([0.0, 1.0, 2.0])})
    selected = AoGridBlockProgram(layout).density_layout
    source = input_tensor("wrong", replace(selected.global_spec, dtype="float32"))
    with pytest.raises(ValueError, match="different global domain"):
        selected.select(source)
    wrong = input_tensor(
        "wrong_local",
        TensorSpec((Index("bad", IndexSpace("bad_space", "ao", 18)),), role="input"),
    )
    with pytest.raises(ValueError, match="different local domain"):
        selected.scatter_add(wrong)
    for selections in (
        (),
        selected.selections[::-1],
        (selected.selections[0], selected.selections[0]),
    ):
        with pytest.raises(ValueError):
            runtime_cartesian_select(
                input_tensor("density", selected.global_spec), selections
            )
    for candidate in (program, AoGridBlockProgram(layout).scatter_program()):
        source_text = emit_cuda(plan_cuda(candidate, cuda_target_info("sm_120")))
        assert "runtime_index_0" in source_text and "runtime_index_1" in source_text
        assert "runtime index out of bounds" in source_text


def test_ao_layout_capabilities_dense_fallback_and_lease_lifetime() -> None:
    layout = _layout()
    layout.require_derivative_order(2)
    for unsupported in (
        replace(layout, map_derivative_order=1),
        replace(layout, map_derivative_order=None),
        replace(layout, derivative_order=1),
    ):
        with pytest.raises(ValueError, match="derivative"):
            unsupported.require_derivative_order(2)
    dense = _layout(5, indexed=False, map_order=None)
    dense.require_derivative_order(2)
    assert AoGridBlockProgram(dense).density_layout is None
    np.testing.assert_array_equal(
        execute(
            AoGridBlockProgram(dense).density_program(), {"density": _density()}
        ).outputs["local_density"],
        _density(),
    )
    assert dense.logical_work(4)["density_gather_values"] == 0
    assert dense.logical_work(4)["ao_map_bytes"] == 0
    assert layout.logical_work(4)["projection_fma_pairs"] == 2 * 4 * 2 * 3**2
    view = GridTaskView(nao=5, nactive=5, npoint=2, jets=10)
    task = DeviceGridTask(
        SimpleNamespace(plan=SimpleNamespace(order=2), basis_identity="basis"), view
    )
    assert task.layout is task.layout
    assert not task.layout.indexed
    task._active = False
    with pytest.raises(RuntimeError, match="expired"):
        _ = task.layout


def test_typed_block_admission_checks_shape_basis_and_epochs_before_launch() -> None:
    owner = object.__new__(CudaGrid)
    owner.plan = SimpleNamespace(nao=5, order=2)
    owner.basis_identity = "basis-geometry"
    owner.basis_generation = 7
    owner._geometry_generation = 12
    layout = replace(_layout(), basis_generation=7, geometry_generation=12)
    owner._admit_block_layout(layout, 3, 2)
    owner._admit_block_layout(layout, 3, 2, indexed=True)
    with pytest.raises(ValueError, match="current owner"):
        owner._admit_block_layout(layout, 3, 2, indexed=False)
    owner._admit_block_layout(None, 3, 2)
    for changes in (
        {"basis_identity": "other-basis"},
        {"basis_generation": 6},
        {"geometry_generation": 11},
        {"npoint": 1},
        {"nactive": 2},
    ):
        with pytest.raises(ValueError, match="current owner"):
            owner._admit_block_layout(replace(layout, **changes), 3, 2)
    with pytest.raises(ValueError, match="capability"):
        owner._admit_block_layout(replace(layout, map_derivative_order=1), 3, 2)
    assert (
        AoGridBlockProgram(layout).density_program().logical_hash
        == AoGridBlockProgram(
            replace(layout, basis_generation=1, geometry_generation=2)
        )
        .density_program()
        .logical_hash
    )


def test_cartesian_precision_preserves_int64_controls_and_qualifies_scatter() -> None:
    program = AoGridBlockProgram(_layout()).density_program()
    name = program.debug_names[program.outputs["local_density"]]
    lowered = lower_precision(
        program, {name: PrecisionDirective("float32", "float32", "float32")}
    )
    maps = [
        node
        for node in lowered.live_nodes
        if node.op == "input" and node.attrs["name"] == "ao_ids"
    ]
    assert len(maps) == 1 and maps[0].spec.dtype == "int64"
    ids = np.array([4, 1, 3], dtype=np.int64)
    np.testing.assert_array_equal(
        execute(lowered, {"density": _density(), "ao_ids": ids}).outputs[
            "local_density"
        ],
        _density()[:, ids[:, None], ids[None, :]],
    )
    scatter = AoGridBlockProgram(_layout()).scatter_program()
    with pytest.raises(ValueError, match="qualification"):
        lower_precision(
            scatter,
            {
                scatter.debug_names[scatter.outputs["potential"]]: PrecisionDirective(
                    "float32", "float32", "float32"
                )
            },
        )


def test_ao_layout_uses_explicit_point_interval_and_rejects_bad_domains() -> None:
    layout = replace(_layout(), point_start=17)
    point = AoGridBlockProgram(layout)._indices[5]
    assert (point.start, point.stop, point.extent) == (17, 19, 2)
    for changes in (
        {"nao": 0},
        {"nactive": 6},
        {"derivative_order": True},
        {"indexed": False},
        {"map_derivative_order": 4},
        {"basis_identity": ""},
    ):
        with pytest.raises(ValueError):
            replace(layout, **changes)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit finite Slurm GPU allocation",
)
@pytest.mark.parametrize("consumer", ["select", "scatter", "projection"])
def test_cartesian_cuda_replays_maps_and_recovers_bounds(
    tmp_path: Path, consumer: str
) -> None:
    """Qualify generated and resident backends against independent local math."""
    from generativeqc.profiles import find_nvcc
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from generativeqc_compiler.tensor.cuda_resident import (
        PreparedResident,
        compile_resident,
    )

    assert os.environ.get("SLURM_JOB_ID"), (
        "GPU execution must retain its Slurm allocation"
    )
    nvcc = find_nvcc()
    assert nvcc is not None, "allocated GPU tests require NVCC"
    compiler = CudaCompilerAdapter(nvcc, cuda_target_info("sm_120"))
    layout = _layout()
    if consumer == "select":
        program, output_name = (
            AoGridBlockProgram(layout).density_program(),
            "local_density",
        )
        feeds = {"density": _density()}
        oracle = lambda ids: feeds["density"][:, ids[:, None], ids[None, :]]
    elif consumer == "scatter":
        program, output_name = AoGridBlockProgram(layout).scatter_program(), "potential"
        feeds = {"local_potential": np.arange(18, dtype=np.float64).reshape(2, 3, 3)}
        oracle = lambda ids: _scatter_oracle(feeds["local_potential"], ids)
    else:
        program, output_name = (
            AoGridBlockProgram(layout).projection_program(4),
            "projected",
        )
        feeds = {
            "density": _density(),
            "ao_jets": np.arange(60, dtype=np.float64).reshape(10, 2, 3),
        }
        oracle = lambda ids: np.einsum(
            "jpm,smn->sjpn",
            feeds["ao_jets"][:4],
            feeds["density"][:, ids[:, None], ids[None, :]],
        )
    plan = plan_cuda(program, compiler.target)
    artifact = compile_cuda(plan, compiler, tmp_path)
    resident_artifact = compile_resident(plan, compiler, tmp_path)
    initial = np.array([4, 1, 3], dtype=np.int64)
    changed = np.array([4, 1, 4], dtype=np.int64)
    invalid = np.array([4, 1, 5], dtype=np.int64)
    with (
        PreparedCuda(plan, artifact) as prepared,
        PreparedResident(plan, resident_artifact) as resident,
    ):
        for ids in (initial, changed):
            actual = prepared.execute({**feeds, "ao_ids": ids}).outputs[output_name]
            np.testing.assert_allclose(actual, oracle(ids), rtol=1e-13, atol=1e-13)
            resident.upload({**feeds, "ao_ids": ids})
            leases, _ = resident.run()
            np.testing.assert_allclose(
                resident.download(leases[output_name]),
                oracle(ids),
                rtol=1e-13,
                atol=1e-13,
            )
        with pytest.raises(RuntimeError, match="runtime index out of bounds"):
            prepared.execute({**feeds, "ao_ids": invalid})
        np.testing.assert_allclose(
            prepared.execute({**feeds, "ao_ids": initial}).outputs[output_name],
            oracle(initial),
            rtol=1e-13,
            atol=1e-13,
        )
        resident.upload({**feeds, "ao_ids": invalid})
        with pytest.raises(RuntimeError, match="runtime index out of bounds"):
            resident.run()
        resident.upload({**feeds, "ao_ids": initial})
        leases, _ = resident.run()
        np.testing.assert_allclose(
            resident.download(leases[output_name]),
            oracle(initial),
            rtol=1e-13,
            atol=1e-13,
        )
