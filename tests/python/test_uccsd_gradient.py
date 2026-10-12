"""Complete native UCCSD CPU/CUDA forces and transactional derivative gates."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_uccsd_physical_response import _independent_fd, _independent_gradient
from test_uccsd_solver import _snapshot

from tools.generativeqc_cc.solver import SolverOptions
from tools.generativeqc_cc.uccsd_gradient import (
    PreparedUCCSDGradient,
    UCCSDGradientOptions,
)
from tools.generativeqc_cc.uccsd_lambda import PreparedUCCSDLambda, UCCSDLambdaOptions
from tools.generativeqc_cc.uccsd_physical_response import (
    PreparedUCCSDPhysicalResponse,
    UCCSDPhysicalResponseOptions,
)
from tools.generativeqc_cc.uccsd_solver import PreparedUCCSD
from tools.generativeqc_posthf.export import export_uhf
from tools.generativeqc_posthf.providers import ConventionalProvider
from tools.generativeqc_posthf.sources import NativeSource

_H2 = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.4))]
_OH = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.83))]
_NH2 = [("N", (0.0, 0.0, 0.0)), ("H", (0.0, -1.51, 1.13)), ("H", (0.0, 1.51, 1.13))]


def _native_oracle(source: NativeSource, *, broken: bool = False) -> Any:
    """Share only normalized shell primitives; PySCF evaluates everything else."""
    pyscf = pytest.importorskip("pyscf")
    cc = pytest.importorskip("pyscf.cc")
    labels = [
        f"{pyscf.data.elements.ELEMENTS[atom.atomic_number]}{index}"
        for index, atom in enumerate(source.atoms)
    ]
    basis = {label: [] for label in labels}
    for shell in source.shells:
        basis[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *(
                    (primitive.exponent, primitive.coefficient)
                    for primitive in shell.primitives
                ),
            ]
        )
    molecule = pyscf.gto.M(
        atom=[
            (label, atom.position)
            for label, atom in zip(labels, source.atoms, strict=True)
        ],
        basis=basis,
        unit="Bohr",
        spin=source.multiplicity - 1,
        cart=source.representation == "cartesian",
        verbose=0,
    )
    restricted = source.multiplicity == 1 and not broken
    reference = pyscf.scf.RHF(molecule) if restricted else pyscf.scf.UHF(molecule)
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-13, 1e-10, 300
    if broken:
        density = np.zeros((2, source.nbf, source.nbf))
        density[0, 0, 0] = density[1, 1, 1] = 1.0
        reference.kernel(dm0=density)
    else:
        reference.kernel()
    assert reference.converged
    if restricted:
        reference = reference.to_uhf()
    if broken:
        assert np.linalg.norm(reference.make_rdm1()[0] - reference.make_rdm1()[1]) > 0.1
    oracle = cc.UCCSD(reference)
    oracle.conv_tol, oracle.conv_tol_normt, oracle.max_cycle = 1e-13, 1e-11, 400
    oracle.kernel()
    assert oracle.converged
    oracle.solve_lambda()
    assert oracle.converged_lambda
    return oracle


@pytest.mark.parametrize(
    "atoms,spin,broken,representation,native_reference",
    [
        (_H2, 0, False, "cartesian", True),
        (_OH, 1, False, "cartesian", True),
        (_NH2, 1, False, "spherical", False),
        (
            [("H", (0.0, 0.0, 2.45665 * index)) for index in range(4)],
            0,
            True,
            "cartesian",
            False,
        ),
    ],
)
def test_complete_cpu_native_gradient_matches_independent_oracle(
    atoms: list, spin: int, broken: bool, representation: str, native_reference: bool
) -> None:
    with NativeSource(
        atoms, multiplicity=spin + 1, representation=representation
    ) as source:
        oracle = _native_oracle(source, broken=broken)
        if native_reference:
            snapshot, _ = export_uhf(source, max_iterations=300, tolerance=1e-12)
        else:
            snapshot = replace(
                _snapshot(oracle),
                geometry_hash=source.geometry_hash,
                basis_hash=source.basis_hash,
                representation=source.representation,
            )
        with (
            ConventionalProvider(snapshot, source) as provider,
            PreparedUCCSD(
                snapshot, provider, options=SolverOptions(max_iterations=400)
            ) as parent,
        ):
            state = parent.solve()
            assert state.converged
            with (
                PreparedUCCSDLambda(parent, state) as lambdas,
                PreparedUCCSDPhysicalResponse(
                    lambdas, lambdas.solve(), provider
                ) as physical,
                PreparedUCCSDGradient(physical) as owner,
            ):
                result = owner.solve()
                np.testing.assert_allclose(
                    result.total_energy, oracle.e_tot, atol=3e-9, rtol=0
                )
                np.testing.assert_allclose(
                    result.gradient, _independent_gradient(oracle), atol=2e-6, rtol=0
                )
                np.testing.assert_allclose(
                    result.gradient[0, 2], _independent_fd(oracle), atol=2e-7, rtol=0
                )
                np.testing.assert_allclose(
                    result.gradient.sum(axis=0), 0, atol=2e-9, rtol=0
                )
                assert result.provenance["eri_policy"] == "packaged-cpu"
                assert (
                    result.provenance["one_electron_policy"]
                    == "explicit-native-compatibility"
                )
                assert result.provenance["shell_quartets"] == len(source.shells) ** 4
                assert (
                    result.provenance["executed_generated_scalar_work"]
                    <= owner.planned_work
                )
                for value in (result.gradient, result.forces):
                    with pytest.raises(ValueError):
                        value.setflags(write=True)


@pytest.fixture
def native_physical() -> Any:
    with NativeSource(_H2) as source:
        snapshot, _ = export_uhf(source, tolerance=1e-12)
        with (
            ConventionalProvider(snapshot, source) as provider,
            PreparedUCCSD(snapshot, provider) as parent,
            PreparedUCCSDLambda(parent, parent.solve()) as lambdas,
            PreparedUCCSDPhysicalResponse(
                lambdas, lambdas.solve(), provider
            ) as physical,
        ):
            yield physical


@pytest.mark.parametrize(
    "options",
    [
        UCCSDGradientOptions(max_bytes=1),
        UCCSDGradientOptions(max_total_work=1),
        UCCSDGradientOptions(max_shell_quartets=1),
        UCCSDGradientOptions(max_primitive_terms=1),
    ],
)
def test_gradient_admission_precedes_any_response_or_derivative(
    native_physical: Any, options: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("rejected derivative admission performed scientific work")

    monkeypatch.setattr(native_physical, "solve", forbidden)
    monkeypatch.setattr(
        native_physical.source, "weighted_eri_shell_gradient_cpu", forbidden
    )
    with pytest.raises(MemoryError):
        PreparedUCCSDGradient(native_physical, options=options)


def test_late_derivative_failure_and_borrowed_lifetime_never_publish(
    native_physical: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    with PreparedUCCSDGradient(native_physical) as owner:
        original = owner.source.weighted_eri_shell_gradient_cpu
        calls = 0

        def late_failure(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == len(owner.source.shells) ** 4:
                raise RuntimeError("injected last-quartet failure")
            return original(*args, **kwargs)

        with monkeypatch.context() as context:
            context.setattr(
                owner.source, "weighted_eri_shell_gradient_cpu", late_failure
            )
            with pytest.raises(RuntimeError, match="last-quartet"):
                owner.solve()
        with monkeypatch.context() as context:
            context.setattr(
                owner.source,
                "weighted_eri_shell_gradient_cpu",
                lambda *args: (np.full((4, 3), np.nan), 0),
            )
            with pytest.raises(ValueError, match="slot gradients"):
                owner.solve()
        native_physical.close()
        with pytest.raises(RuntimeError, match="closed"):
            owner.solve()
    with pytest.raises(RuntimeError, match="closed"):
        owner.solve()


def test_native_complete_gradient_matches_reconverged_molecular_finite_difference(
    native_physical: Any,
) -> None:
    with PreparedUCCSDGradient(native_physical) as owner:
        gradient = owner.solve().gradient
    from tools.generativeqc_cc.complete_gradient import complete_gradient_validation

    restricted = complete_gradient_validation(native_physical.source)
    np.testing.assert_allclose(gradient, restricted.gradient, atol=2e-8, rtol=0)
    energies = []
    for sign in (-1, 1):
        moved = [
            (
                symbol,
                tuple(
                    np.asarray(position)
                    + (np.array([0, 0, sign * 1e-4]) if index == 1 else 0)
                ),
            )
            for index, (symbol, position) in enumerate(_H2)
        ]
        with NativeSource(moved) as source:
            snapshot, _ = export_uhf(source, tolerance=1e-12)
            with (
                ConventionalProvider(snapshot, source) as provider,
                PreparedUCCSD(snapshot, provider) as parent,
            ):
                state = parent.solve()
                assert state.converged
                energies.append(snapshot.reference_energy + state.correlation_energy)
    np.testing.assert_allclose(
        gradient[1, 2], (energies[1] - energies[0]) / 2e-4, atol=2e-7, rtol=0
    )


def test_private_cpu_derivative_failures_leave_outputs_unchanged(
    native_physical: Any,
) -> None:
    """Exercise the real C ABI, including transactional primitive diagnostics."""
    import ctypes as ct

    from tools.generativeqc_posthf.sources import pointer

    source = native_physical.source
    gradient = np.full((4, 3), 73.0)
    weights = np.ones((1, 1, 1, 1))
    evaluations = ct.c_uint64(79)
    indices = (ct.c_size_t * 4)(len(source.shells), 0, 0, 0)
    with pytest.raises(RuntimeError, match="out of range"):
        source._call(
            "generativeqc_posthf_weighted_eri_shell_gradient_cpu_v1",
            source._handle,
            indices,
            pointer(weights),
            weights.size,
            pointer(gradient),
            gradient.size,
            ct.byref(evaluations),
        )
    np.testing.assert_array_equal(gradient, 73.0)
    assert evaluations.value == 79
    one = np.full((len(source.atoms), 3), 83.0)
    invalid = np.full((source.nbf, source.nbf), np.nan)
    with pytest.raises(RuntimeError, match="nonfinite"):
        source._call(
            "generativeqc_posthf_one_electron_gradient_cpu_v1",
            source._handle,
            pointer(invalid),
            pointer(invalid),
            invalid.size,
            pointer(one),
            one.size,
        )
    np.testing.assert_array_equal(one, 83.0)


@pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_TENSOR_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
@pytest.mark.parametrize(
    "atoms,spin,broken",
    [
        (_NH2, 1, False),
        ([("H", (0.0, 0.0, 2.45665 * index)) for index in range(4)], 0, True),
    ],
)
def test_complete_cuda_gradient_matches_independent_oracle(
    atoms: list, spin: int, broken: bool, tmp_path: Path
) -> None:
    from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from generativeqc_compiler.common.cuda_target import cuda_target_info

    from tools.generativeqc_cc.native_tensor_cuda import CudaCCTensorExecutor
    from tools.generativeqc_posthf.cuda import compile_cuda
    from tools.generativeqc_response.spin_cuda import CudaSpinJKBackend

    assert os.environ.get("SLURM_JOB_ID"), (
        "real CUDA gates require finite Slurm allocation"
    )
    cache = Path(
        os.environ.get("GENERATIVEQC_UCCSD_PHYSICAL_TEST_CACHE", str(tmp_path))
    )
    compiler = CudaCompilerAdapter(
        Path(os.environ["GENERATIVEQC_NVCC"]),
        cuda_target_info(os.environ.get("GENERATIVEQC_TENSOR_ARCH", "sm_120")),
    )
    artifact = compile_cuda(compiler, cache / "transform")
    executor = CudaCCTensorExecutor(
        max_bytes=1 << 30, compiler=compiler, cache=cache / "tensor"
    )
    with NativeSource(atoms, multiplicity=spin + 1) as source:
        oracle = _native_oracle(source, broken=broken)
        snapshot = replace(
            _snapshot(oracle),
            geometry_hash=source.geometry_hash,
            basis_hash=source.basis_hash,
        )
        with (
            CudaSpinJKBackend(source) as orbital_backend,
            ConventionalProvider(
                snapshot,
                source,
                backend="cuda",
                cuda_artifact=artifact,
                budget_bytes=512 << 20,
                cache_policy="lru",
            ) as provider,
            PreparedUCCSD(
                snapshot,
                provider,
                options=SolverOptions(max_iterations=400),
                executor=executor,
            ) as parent,
        ):
            state = parent.solve()
            assert state.converged
            with (
                PreparedUCCSDLambda(
                    parent, state, options=UCCSDLambdaOptions(max_bytes=2 << 30)
                ) as lambdas,
                PreparedUCCSDPhysicalResponse(
                    lambdas,
                    lambdas.solve(),
                    provider,
                    response_backend=orbital_backend,
                    options=UCCSDPhysicalResponseOptions(max_bytes=2 << 30),
                ) as physical,
                PreparedUCCSDGradient(
                    physical, options=UCCSDGradientOptions(max_bytes=3 << 30)
                ) as owner,
            ):
                result = owner.solve()
                np.testing.assert_allclose(
                    result.total_energy, oracle.e_tot, atol=3e-9, rtol=0
                )
                np.testing.assert_allclose(
                    result.gradient,
                    _independent_gradient(oracle),
                    atol=2e-6,
                    rtol=0,
                )
                assert result.provenance["one_electron_policy"] == "generated-cuda"
                assert (
                    result.provenance["physical_response"]["orbital_backend"] == "cuda"
                )
                assert (
                    result.provenance["physical_response"]["tensor_backend"]
                    == "cuda-fp64-ordinary-stream"
                )
