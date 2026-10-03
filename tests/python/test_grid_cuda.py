"""Opt-in real-device gates; callers must use a finite Slurm GPU allocation."""

import os
import typing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.evidence import block_error
from generativeqc_compiler.dft import GridSpec, NativeAO
from generativeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from generativeqc_compiler.dft.fixtures import NAMES, basis_arguments, load_fixture
from generativeqc_compiler.dft.prepared import PreparedGrid, PreparedGridBatch

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_GRID_CUDA_TEST") != "1",
    reason="opt-in Slurm CUDA gate",
)


@pytest.fixture(scope="module")
def artifact() -> typing.Any:
    return compile_cuda(
        CudaCompilerAdapter(
            Path(
                os.environ.get(
                    "GENERATIVEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc"
                )
            ),
            cuda_target_info(os.environ.get("GENERATIVEQC_GRID_CUDA_ARCH", "sm_120")),
        ),
        Path("/tmp/dft160-cuda-cache"),
    )


def check(actual: typing.Any, expected: typing.Any) -> None:
    result = block_error(actual, expected, atol=1e-11, rtol=1e-10)
    assert result["passed"], result


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tile_points", [7, 31])
def test_all_jets_features_partial_tiles_and_resident_density(
    artifact: typing.Any, name: typing.Any, tile_points: typing.Any
) -> None:
    meta, arrays = load_fixture(name)
    with NativeAO(**basis_arguments(meta)) as basis:
        with CudaGrid(basis, artifact, order=3, tile_points=tile_points) as cuda:
            with pytest.raises(ValueError, match="supplied density"):
                cuda.evaluate(arrays["points"][:1])
            cuda.set_density(arrays["density"])
            for begin in range(0, len(arrays["points"]), tile_points):
                end = begin + tile_points
                result = cuda.evaluate(arrays["points"][begin:end], download_jets=True)
                for key, value in result.items():
                    expected = arrays[key][:, begin:end]
                    check(value, expected)
            first = cuda.evaluate(arrays["points"][-7:])
            cuda.set_density(2 * arrays["density"])
            second = cuda.evaluate(arrays["points"][-7:])
            for key in first:
                check(second[key], first[key] * (4 if key == "sigma" else 2))
            metrics = cuda.metrics()
            assert metrics["owned_device_bytes"] == cuda.plan.allocation_bytes
            assert metrics["provider_retained_bytes"] <= cuda.plan.provider_bytes
            assert metrics["kernel_ms"] > 0 and metrics["library_ms"] > 0
            assert metrics["input_ms"] > 0 and metrics["output_ms"] > 0
            assert cuda.evaluate(np.empty((0, 3)), download_jets=True)[
                "ao_jets"
            ].shape == (20, 0, basis.nao)
        with pytest.raises(RuntimeError, match="closed"):
            cuda.evaluate(arrays["points"][:1])


def test_feature_task_with_features_reuses_one_evaluated_tile(
    artifact: typing.Any,
) -> None:
    meta, arrays = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        ids = np.arange(basis.nao, dtype=np.uintp)
        with CudaGrid(
            basis,
            artifact,
            order=2,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=("rho", "gradient", "tau"),
        ) as cuda:
            cuda.set_density(arrays["density"])
            with cuda.feature_task_with_features(
                arrays["points"][:7], ids, ("rho", "gradient", "tau")
            ) as (
                features,
                task,
            ):
                assert set(features) == {"rho", "gradient", "tau"}
                check(features["rho"], arrays["rho"][:, :7])
                check(features["gradient"], arrays["gradient"][:, :7])
                check(features["tau"], arrays["tau"][:, :7])
                assert task.view.npoint == 7
                assert task.view.nactive == basis.nao
            with pytest.raises(RuntimeError, match="expired"):
                _ = task.view

            # The legacy functional wrapper is only a compatibility adapter;
            # publication remains backed by the same generic resident lease.
            with cuda.xc_task_with_features(arrays["points"][:3], ids, "PBE") as (
                pbe_features,
                pbe_task,
            ):
                assert set(pbe_features) == {"rho", "gradient"}
                check(pbe_features["rho"], arrays["rho"][:, :3])
                check(pbe_features["gradient"], arrays["gradient"][:, :3])
                assert pbe_task.view.npoint == 3

            with (
                pytest.raises(ValueError, match="ingredient contract"),
                cuda.feature_task_with_features(
                    arrays["points"][:1], ids, ("gradient",)
                ),
            ):
                pass


def test_full_identity_map_matches_explicit_local_map(artifact: typing.Any) -> None:
    meta, arrays = load_fixture("water")
    with NativeAO(**basis_arguments(meta)) as basis:
        ids = np.arange(basis.nao, dtype=np.uintp)
        points = arrays["points"][:7]
        weights = np.ones(len(points))
        with CudaGrid(
            basis,
            artifact,
            order=1,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=("rho", "gradient"),
        ) as cuda:
            cuda.set_density(arrays["density"])
            with cuda.xc_task(points, ids, "PBE") as explicit:
                explicit_integrals, explicit_potential = explicit.xc(
                    weights, "PBE", reset=True, download=True
                )
                assert explicit.view.ao_ids

            with cuda.xc_task(points, None, "PBE") as identity:
                identity_integrals, identity_potential = identity.xc(
                    weights, "PBE", reset=True, download=True
                )
                assert not identity.view.ao_ids
                assert identity.view.nactive == basis.nao

            check(identity_integrals, explicit_integrals)
            check(identity_potential, explicit_potential)


@pytest.mark.parametrize("mask", range(1, 16))
@pytest.mark.parametrize("foreign_stream", [False, True])
@pytest.mark.parametrize("map_kind", ["identity", "subset", "empty"])
def test_resident_restricted_products_match_general_spin_path_bitwise(
    artifact: typing.Any, mask: int, foreign_stream: bool, map_kind: str
) -> None:
    """Copying one ordered spin panel must preserve every requested feature.

    Exercise producer ordering and owned-density lifetime, followed by a UKS
    replacement and a host replacement on the same plan. No source-pointer or
    numerical-equality heuristic is allowed to retain restricted provenance.
    """
    import cupy as cp

    meta, arrays = load_fixture("water")
    ingredients = tuple(
        name
        for bit, name in enumerate(("rho", "gradient", "sigma", "tau"))
        if mask & (1 << bit)
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        CudaGrid(
            basis,
            artifact,
            order=1,
            tile_points=7,
            active_ao_capacity=basis.nao,
            ingredients=ingredients,
        ) as cuda,
    ):
        ids = {
            "identity": None,
            "subset": np.arange(0, basis.nao, 2, dtype=np.uintp),
            "empty": np.empty(0, dtype=np.uintp),
        }[map_kind]
        total = np.ascontiguousarray(arrays["density"].sum(axis=0))
        # Host uploads symmetrize accepted near-symmetric fixture inputs;
        # resident uploads require that canonicalization at their producer.
        total = np.ascontiguousarray(0.5 * (total + total.T))
        restricted = np.stack((0.5 * total, 0.5 * total))
        cuda.set_density(restricted)
        expected = [
            cuda.evaluate(arrays["points"][:count], ao_ids=ids, download_jets=True)
            for count in (0, 1, 7)
        ]
        with cuda._borrow_current_task() as task:
            owner_stream = cp.cuda.ExternalStream(task.view.stream)
        producer = cp.cuda.Stream(non_blocking=True) if foreign_stream else owner_stream
        with producer:
            device_total = cp.asarray(total)
        cuda.set_density_device(
            device_id=cp.cuda.runtime.getDevice(),
            alpha=device_total.data.ptr,
            beta=None,
            matrix_elements=total.size,
            spins=1,
            source_stream=producer.ptr,
        )
        with producer:
            device_total.fill(123.0)
        for count, reference in zip((0, 1, 7), expected, strict=True):
            actual = cuda.evaluate(
                arrays["points"][:count], ao_ids=ids, download_jets=True
            )
            for name in reference:
                np.testing.assert_array_equal(
                    actual[name].view(np.uint64), reference[name].view(np.uint64)
                )
        unrestricted = restricted.copy()
        unrestricted[0] *= 0.7
        unrestricted[1] *= 1.3
        with producer:
            device_spins = cp.asarray(unrestricted)
        cuda.set_density_device(
            device_id=cp.cuda.runtime.getDevice(),
            alpha=device_spins[0].data.ptr,
            beta=device_spins[1].data.ptr,
            matrix_elements=total.size,
            spins=2,
            source_stream=producer.ptr,
        )
        actual = cuda.evaluate(arrays["points"][:7], ao_ids=ids)
        cuda.set_density(unrestricted)
        reference = cuda.evaluate(arrays["points"][:7], ao_ids=ids)
        for name in reference:
            np.testing.assert_array_equal(
                actual[name].view(np.uint64), reference[name].view(np.uint64)
            )


@pytest.mark.parametrize("mask", range(1, 16))
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_cooperative_features_match_independent_density_contractions(
    artifact: typing.Any, mask: int, representation: str
) -> None:
    """Cover warp AO tails, scalar fallback, vacuum and independent spin sums."""
    from generativeqc import Primitive, Shell
    from generativeqc_compiler.dft import density_features

    atoms = [("H", (2.0 * atom - 3.0, 0.0, 0.0)) for atom in range(4)]
    shells = tuple(
        Shell(atom, angular, (Primitive(0.7, 1.0), Primitive(1.3, -0.1)))
        for atom in range(4)
        for angular in range(4)
    )
    ingredients = tuple(
        name
        for bit, name in enumerate(("rho", "gradient", "sigma", "tau"))
        if mask & (1 << bit)
    )
    with (
        NativeAO(atoms, basis=shells, representation=representation) as basis,
        CudaGrid(
            basis,
            artifact,
            order=1,
            tile_points=17,
            active_ao_capacity=basis.nao,
            ingredients=ingredients,
        ) as cuda,
    ):
        generator = np.random.default_rng(24913)
        points = generator.normal(size=(17, 3))
        matrices = generator.normal(size=(2, basis.nao, basis.nao)) * 0.03
        density = matrices + matrices.swapaxes(1, 2)
        all_jets = basis.evaluate(points, order=1)
        for empty_spins in (0, 1, 2):
            current = density.copy()
            if empty_spins:
                current[2 - empty_spins :] = 0.0
            cuda.set_density(current)
            for ids in (
                np.arange(basis.nao, dtype=np.uintp),
                np.arange(33, dtype=np.uintp),
                np.arange(0, basis.nao, 2, dtype=np.uintp),
                np.arange(31, dtype=np.uintp),
                np.empty(0, dtype=np.uintp),
            ):
                local_density = current[:, ids[:, None], ids[None, :]]
                for count in (0, 1, 17):
                    expected = density_features(
                        all_jets[:, :count, ids],
                        local_density,
                        ingredients=ingredients,
                    )
                    actual = cuda.evaluate(points[:count], ao_ids=ids)
                    for name in expected:
                        if count:
                            check(actual[name], expected[name])
                        else:
                            np.testing.assert_array_equal(actual[name], expected[name])


def test_orders_zero_to_three_and_budget_rejection(artifact: typing.Any) -> None:
    meta, arrays = load_fixture("f_spherical")
    with NativeAO(**basis_arguments(meta)) as basis:
        with pytest.raises(ValueError, match="budget"):
            CudaGrid(basis, artifact, budget_bytes=1)
        for order, jets in enumerate((1, 4, 10, 20)):
            with CudaGrid(basis, artifact, order=order, tile_points=13) as cuda:
                actual = cuda.evaluate(
                    arrays["points"][-13:], features=False, download_jets=True
                )
                check(actual["ao_jets"], arrays["ao_jets"][:jets, -13:])
                with pytest.raises(ValueError, match="tile shape"):
                    cuda.evaluate(arrays["points"])


def test_prepared_reuse_changed_geometry_and_ragged_failures(
    artifact: typing.Any,
) -> None:
    items, densities = [], []
    for name in ("h2", "water", "f_spherical"):
        meta, arrays = load_fixture(name)
        items.append(
            {
                **basis_arguments(meta),
                "spec": GridSpec(3, 3, 6),
                "tile_points": 13,
                "order": 3,
                "artifact": artifact,
                "backend": "cuda",
            }
        )
        densities.append(arrays["density"])
    with PreparedGridBatch(items) as gpu:
        first = gpu.execute(densities)
        again = gpu.execute(densities)
        for index, item in enumerate(items):
            with PreparedGrid(**{**item, "backend": "cpu"}) as cpu:
                reference = cpu.integrate(densities[index])
                for key in ("electrons", "integrated_tau"):
                    check(first[index]["result"][key], reference[key])
                    check(again[index]["result"][key], reference[key])
        assert [r["point_begin"] for r in first] == [0, 108, 270]
        failure = gpu.execute(
            [densities[0], np.full_like(densities[1], np.nan), densities[2]]
        )
        assert [r["status"] for r in failure] == ["pass", "fail", "pass"]
        invalid_type = gpu.execute([densities[0], {"invalid": 1}, densities[2]])
        assert [r["status"] for r in invalid_type] == ["pass", "fail", "pass"]
    with PreparedGrid(**items[0]) as plan:
        first = plan.integrate(densities[0])
        xyz = np.array(items[0]["atoms"][1][1]) + [0.07, -0.02, 0.03]
        coordinates = [items[0]["atoms"][0][1], xyz]
        plan.reconfigure(coordinates=coordinates)
        changed = plan.integrate(densities[0])
        assert changed["identity"] != first["identity"]
        with PreparedGrid(
            **{**items[0], "atoms": [(1, r) for r in coordinates], "backend": "cpu"}
        ) as cpu:
            expected = cpu.integrate(densities[0])
            check(changed["electrons"], expected["electrons"])
        with pytest.raises(ValueError):
            plan.reconfigure(spec=replace(plan.grid.spec, radial_points=0))
        check(plan.integrate(densities[0])["electrons"], changed["electrons"])


def test_shared_posthf_runtime_after_cache_extraction() -> None:
    """Exercise the existing cuBLAS MO consumer after extracting shared caching."""
    from tools.generativeqc_posthf.conventions import MOBlock
    from tools.generativeqc_posthf.cuda import compile_cuda as compile_posthf
    from tools.generativeqc_posthf.fixtures import fixture_snapshot, source_arguments
    from tools.generativeqc_posthf.fixtures import load_fixture as load_posthf
    from tools.generativeqc_posthf.providers import ConventionalProvider
    from tools.generativeqc_posthf.sources import NativeSource

    artifact = compile_posthf(
        CudaCompilerAdapter(
            Path(
                os.environ.get(
                    "GENERATIVEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc"
                )
            ),
            cuda_target_info(os.environ.get("GENERATIVEQC_GRID_CUDA_ARCH", "sm_120")),
        ),
        Path("/tmp/dft160-posthf-cache"),
    )
    meta, arrays = load_posthf("h2")
    reference = fixture_snapshot(meta, arrays)
    with (
        NativeSource(**source_arguments(meta)) as source,
        ConventionalProvider(
            reference, source, backend="cuda", cuda_artifact=artifact
        ) as provider,
    ):
        result = provider.get(MOBlock.from_spaces(reference, "ovov"))
        check(result.to_host(), arrays["conventional_mo"][0:1, 1:2, 0:1, 1:2])


def test_gpu_iterator_density_isolation_and_failed_update(
    artifact: typing.Any,
) -> None:
    meta, data = load_fixture("h2")
    options = {
        **basis_arguments(meta),
        "spec": GridSpec(3, 3, 6),
        "tile_points": 7,
        "backend": "cuda",
        "artifact": artifact,
    }
    with PreparedGrid(**options) as plan:
        first = plan.iter_features(data["density"])
        original = next(first)
        second = plan.iter_features(data["density"] * 2)
        doubled = next(second)
        check(doubled.features["rho"], 2 * original.features["rho"])
        with pytest.raises(RuntimeError, match="stale"):
            next(first)
        identity = plan.identity
        with pytest.raises(ValueError, match="budget"):
            plan.reconfigure(budget_bytes=plan.plan.peak_bytes)
        assert plan.identity == identity
        assert np.isfinite(next(second).features["rho"]).all()
