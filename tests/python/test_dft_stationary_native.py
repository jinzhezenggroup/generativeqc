"""The real #162 producer, source binding and lifetime gates (Slurm on CUDA)."""

import os
import typing
from dataclasses import replace

import numpy as np
import pytest
from generativeqc import Calculator, GridSpec, KsOptions
from generativeqc._dft_gradient import (
    StableGridMotion,
    StationaryDerivativeContract,
    StationaryKsState,
    bind_generated_xc_geometry,
    scf_regularization_identity,
    xc_regularization_identity,
)
from generativeqc_compiler.dft import ExplicitGrid, NativeAO
from generativeqc_compiler.xc import functional

from tools.generativeqc_validation.dft_gradient import h2_overlap

ATOMS = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)


@pytest.mark.parametrize(
    "method,device",
    [
        ("lda-rks", "cpu"),
        ("pbe-rks", "cpu"),
        ("lda-uks", "cpu"),
        ("pbe-uks", "cpu"),
        *(
            pytest.param(
                method,
                "cuda",
                marks=pytest.mark.skipif(
                    os.environ.get("GENERATIVEQC_DFT_CUDA_TEST") != "1",
                    reason="Slurm CUDA gate",
                ),
            )
            for method in ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks")
        ),
    ],
)
def test_native_snapshot_rejects_relabeling_and_replay(
    method: typing.Any, device: typing.Any
) -> None:
    unrestricted = method.endswith("uks")
    charge, multiplicity = (-1, 2) if unrestricted else (0, 1)
    calculator = Calculator(
        method=method,
        device=device,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [ATOMS], charges=[charge], multiplicities=[multiplicity]
    ) as batch:
        with NativeAO(ATOMS, charge=charge, multiplicity=multiplicity) as basis:
            with pytest.raises(RuntimeError, match="invalid argument"):
                StationaryKsState.from_native(batch, basis)
            batch.execute(strict=True)
            from generativeqc._snapshot_grid_cache import SnapshotGridCache

            # Exercise the public-force cache on CPU too, without requiring a
            # GPU to test that content reuse never grants a stale native lease.
            batch._snapshot_grid_cache = SnapshotGridCache()
            state = StationaryKsState.from_native(batch, basis)
            assert not state._source.grid_cache_work["exact_grid_reused"]
            assert state.identity.spin == (
                "polarized" if unrestricted else "unpolarized"
            )
            assert state.identity.ingredients == (
                ("rho",) if method.startswith("lda") else ("rho", "sigma")
            )
            assert state._source.backend == device
            assert state._source.metadata[0] == (2 if device == "cpu" else 3)
            if device == "cpu":
                assert state._source.metadata[12] == 2**64 - 1
            assert state._source.grid_spec == GRID
            assert np.all(state._source.atomic_weights > 0)
            contract = StationaryDerivativeContract(state.identity)
            assert contract.validate(state) is state
            with pytest.raises(AttributeError, match="provenance is immutable"):
                state._source.metadata = tuple([1] * 16)
            np.testing.assert_allclose(state.overlap, h2_overlap(basis), atol=2e-14)
            np.testing.assert_allclose(
                np.trace(state.density @ state.overlap, axis1=1, axis2=2),
                [2, 1] if unrestricted else [2],
                atol=1e-9,
            )

            # A self-consistent change of AO metric must still fail the native
            # content proof. Algebra alone accepts this congruence transform.
            transformed = replace(
                state,
                overlap=2 * state.overlap,
                density=state.density / 2,
                weighted_density=state.weighted_density / 2,
                coefficients=state.coefficients / np.sqrt(2),
                fock=2 * state.fock,
            )
            assert contract._validate_arrays(transformed) is transformed
            with pytest.raises(ValueError, match="snapshot content"):
                contract.validate(transformed)

            spec = functional(
                "PBE" if method.startswith("pbe") else "LDA_XC_PW",
                spin="polarized" if unrestricted else "unpolarized",
            )
            generated = bind_generated_xc_geometry(
                contract, state, spec, basis, state.grid
            )
            assert generated.regularization_identity == scf_regularization_identity()
            shift = np.array([0.13, -0.07, 0.05])
            translation = StableGridMotion(
                topology_identity=state.identity.topology_identity,
                centers=np.broadcast_to(shift, generated.partials.centers.shape),
                points=np.broadcast_to(shift, generated.partials.points.shape),
                weights=np.zeros_like(generated.partials.weights),
            )
            assert generated.directional(translation).total == pytest.approx(
                0.0, abs=2e-10
            )

            # Relabeling the exact native state as the interior diagnostic
            # domain is still forbidden even though #163-A now has a real bridge.
            relabeled = replace(
                state,
                identity=replace(
                    state.identity,
                    regularization_identity=xc_regularization_identity(spec),
                ),
            )
            with pytest.raises(ValueError, match="native stationary state identity"):
                StationaryDerivativeContract(relabeled.identity).validate(relabeled)

            # Same AO count and geometry, different exponents: labels and sizes
            # cannot substitute for the exact normalized source basis.
            shells = list(basis.shells)
            shell = shells[0]
            primitive = replace(
                shell.primitives[0], exponent=1.1 * shell.primitives[0].exponent
            )
            shells[0] = replace(shell, primitives=(primitive, *shell.primitives[1:]))
            with NativeAO(
                ATOMS, basis=shells, charge=charge, multiplicity=multiplicity
            ) as wrong:
                with pytest.raises(ValueError, match="basis/overlap source"):
                    StationaryKsState.from_native(batch, wrong)
                forged = replace(
                    state,
                    identity=replace(state.identity, basis_identity=wrong.identity),
                )
                with pytest.raises(
                    ValueError, match="native stationary state identity"
                ):
                    StationaryDerivativeContract(forged.identity).validate(forged)

            wrong_grid = ExplicitGrid(
                state.grid.points,
                state.grid.weights * 1.01,
                state.grid.owners,
                {"test": "wrong weights"},
            )
            with pytest.raises(ValueError, match="grid source"):
                StationaryKsState.from_native(batch, basis, wrong_grid)

            with calculator.prepare_batch(
                [ATOMS], charges=[charge], multiplicities=[multiplicity]
            ) as other_batch:
                other_batch.execute(strict=True)
                other = StationaryKsState.from_native(other_batch, basis)
                with pytest.raises(
                    ValueError, match="native stationary state identity"
                ):
                    contract.validate(replace(state, _source=other._source))

            batch.execute(strict=True)
            with pytest.raises(ValueError, match="stale"):
                contract.validate(state)
            current = StationaryKsState.from_native(batch, basis)
            assert current.grid is state.grid
            assert current._source.grid_cache_work["exact_grid_reused"]
            assert current.identity.solve_epoch > state.identity.solve_epoch
            stale_handle = state._source._handle
            assert type(stale_handle) is int  # No mutable ctypes .value alias.
            with pytest.raises(AttributeError, match="provenance is immutable"):
                state._source._handle = current._source._handle
            with pytest.raises(AttributeError, match="provenance is immutable"):
                del state._source._handle
            assert state._source._handle == stale_handle
            with pytest.raises(ValueError, match="stale"):
                contract.validate(state)
            assert (
                StationaryDerivativeContract(current.identity).validate(current)
                is current
            )

            # A rejected geometry update revokes the next proof too, before a
            # new successful solve. This calls the actual native invalidation.
            result = batch.execute(coordinates=[[0.0]], strict=False)
            assert not result.items[0].succeeded
            with pytest.raises(ValueError, match="stale"):
                StationaryDerivativeContract(current.identity).validate(current)

            batch.execute(strict=True)
            final = StationaryKsState.from_native(batch, basis)
        # Source AO lifetime does not own the native snapshot; batch lifetime does.
        assert StationaryDerivativeContract(final.identity).validate(final) is final
        final._source.close()
        assert final._source._handle == 0
        final._source.close()  # Explicit lease closure is idempotent.
        with pytest.raises(ValueError, match="stale"):
            StationaryDerivativeContract(final.identity).validate(final)
    with pytest.raises(RuntimeError, match="closed"):
        StationaryDerivativeContract(final.identity).validate(final)
    assert batch._snapshot_grid_cache is None


def test_snapshot_cache_replaces_moved_geometry_without_reusing_the_lease() -> None:
    """A cached exact grid is not permission to relabel a changed native model."""
    from generativeqc._snapshot_grid_cache import SnapshotGridCache

    calculator = Calculator(method="pbe-rks", ks_options=KsOptions(grid=GRID))
    moved = [("H", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))]
    with calculator.prepare_batch([ATOMS]) as batch:
        batch._snapshot_grid_cache = SnapshotGridCache()
        batch.execute(strict=True)
        with NativeAO(ATOMS) as basis:
            original = StationaryKsState.from_native(batch, basis)
        batch.execute([[position for _, position in moved]], strict=True)
        with NativeAO(moved) as basis:
            current = StationaryKsState.from_native(batch, basis)
            assert current.grid is not original.grid
            assert not current._source.grid_cache_work["exact_grid_reused"]
            assert batch._snapshot_grid_cache.grid is current.grid
            with pytest.raises(ValueError, match="grid source"):
                StationaryKsState.from_native(batch, basis, original.grid)
            with pytest.raises(ValueError, match="stale"):
                StationaryDerivativeContract(original.identity).validate(original)
            batch._snapshot_grid_cache.max_bytes = 0
            batch._snapshot_grid_cache.clear()
            uncached = StationaryKsState.from_native(batch, basis)
            assert uncached.grid.identity == current.grid.identity
            assert uncached._source.grid_cache_work["retained_bytes"] == 0
            assert uncached.grid is not current.grid
