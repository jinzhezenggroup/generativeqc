"""Exercise the native provider proof without pretending to execute CUDA forces."""

import ctypes as ct
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
from generativeqc import _native
from generativeqc._ks_snapshot import NativeKsSnapshot


def _snapshot(
    coulomb: int = 1,
    exchange: int = 2**32 - 1,
    threshold: float = 1e-10,
    *,
    fitted: bool = True,
) -> tuple[typing.Any, list[str]]:
    checks: list[str] = []

    def current() -> None:
        checks.append("current")

    def binding(
        batch: object, handle: object, j: typing.Any, k: typing.Any, metric: typing.Any
    ) -> int:
        checks.append("native")
        ct.cast(j, ct.POINTER(ct.c_uint32))[0] = coulomb
        ct.cast(k, ct.POINTER(ct.c_uint32))[0] = exchange
        ct.cast(metric, ct.POINTER(ct.c_double))[0] = threshold
        return 0

    snapshot = SimpleNamespace(
        check_current=current,
        _library=SimpleNamespace(generativeqc_ks_snapshot_fock_provider_v1=binding),
        _batch=SimpleNamespace(
            _batch=None,
            _context=None,
            _calculator=SimpleNamespace(
                _density_fitting_mode=1 if fitted else _native.DENSITY_FITTING_NONE
            ),
        ),
        _handle=None,
        coefficients=(1.0, 1.0, 0.0),
    )
    return snapshot, checks


@pytest.mark.parametrize("coulomb", (0, 1, 2))
@pytest.mark.parametrize("exchange", (0, 1, 2, 2**32 - 1))
def test_provider_proof_uses_native_approximation(coulomb: int, exchange: int) -> None:
    snapshot, checks = _snapshot(coulomb, exchange)
    names = {0: "exact", 1: "density-fitted", 2: "seminumerical-cosx"}
    assert NativeKsSnapshot.fock_provider_proof(snapshot) == (
        names[coulomb],
        names.get(exchange),
        1e-10,
    )
    assert checks == ["current", "native", "current"]


@pytest.mark.parametrize("approximation", (0, 1, 2**32 - 1, 2, 99))
def test_range_provider_proof_uses_checked_native_approximation(
    approximation: int,
) -> None:
    snapshot, checks = _snapshot()

    def binding(batch: object, handle: object, result: typing.Any) -> int:
        checks.append("native")
        ct.cast(result, ct.POINTER(ct.c_uint32))[0] = approximation
        return 0

    snapshot._library.generativeqc_ks_snapshot_range_provider_v1 = binding
    if approximation in (0, 1, 2**32 - 1):
        assert NativeKsSnapshot.range_density_fitted.fget(snapshot) == (
            approximation == 1
        )
    else:
        with pytest.raises(ValueError, match="unknown native LR approximation"):
            NativeKsSnapshot.range_density_fitted.fget(snapshot)
    assert checks[0] == checks[-1] == "current"
    assert "native" in checks


@pytest.mark.parametrize("stale_before_read", (False, True))
def test_range_provider_proof_rejects_token_changes(stale_before_read: bool) -> None:
    snapshot, checks = _snapshot()
    native_read = False

    def current() -> None:
        if stale_before_read or native_read:
            raise RuntimeError("stale token")

    def binding(batch: object, handle: object, result: typing.Any) -> int:
        nonlocal native_read
        native_read = True
        checks.append("native")
        ct.cast(result, ct.POINTER(ct.c_uint32))[0] = 1
        return 0

    snapshot.check_current = current
    snapshot._library.generativeqc_ks_snapshot_range_provider_v1 = binding
    with pytest.raises(RuntimeError, match="stale token"):
        NativeKsSnapshot.range_density_fitted.fget(snapshot)
    assert native_read == (not stale_before_read)


def test_missing_range_provider_proof_never_assumes_direct() -> None:
    snapshot, checks = _snapshot()
    with pytest.raises(NotImplementedError, match="LR approximation proof"):
        NativeKsSnapshot.range_density_fitted.fget(snapshot)
    assert "current" in checks


@pytest.mark.parametrize("field", ("coulomb", "exchange"))
def test_provider_proof_rejects_unknown_approximation(field: str) -> None:
    snapshot, _ = _snapshot(**{field: 99})
    with pytest.raises(ValueError, match="unknown"):
        NativeKsSnapshot.fock_provider_proof(snapshot)


@pytest.mark.parametrize("threshold", (-1.0, float("nan"), float("inf")))
def test_provider_proof_rejects_invalid_metric(threshold: float) -> None:
    snapshot, _ = _snapshot(threshold=threshold)
    with pytest.raises(ValueError, match="metric threshold"):
        NativeKsSnapshot.fock_provider_proof(snapshot)


@pytest.mark.parametrize("failure_check", (1, 2))
def test_provider_proof_checks_token_before_and_after_native_read(
    failure_check: int,
) -> None:
    snapshot, checks = _snapshot()
    count = 0

    def current() -> None:
        nonlocal count
        count += 1
        if count == failure_check:
            raise RuntimeError("stale token")

    snapshot.check_current = current
    with pytest.raises(RuntimeError, match="stale token"):
        NativeKsSnapshot.fock_provider_proof(snapshot)
    assert ("native" in checks) == (failure_check == 2)


def test_fitted_integral_reserve_checks_the_live_snapshot() -> None:
    from generativeqc_compiler.method.stationary_resources import (
        stationary_fitted_integral_reserve,
    )

    checks = []
    snapshot = SimpleNamespace(
        density_fitted=True, check_current=lambda: checks.append("current")
    )
    assert NativeKsSnapshot.stationary_integral_device_reserve(
        snapshot, atoms=96, aos=768, primitives=704
    ) == stationary_fitted_integral_reserve(atoms=96, aos=768, primitives=704)
    assert checks == ["current"]


def test_direct_snapshot_cannot_offer_a_fitted_integral_reserve() -> None:
    snapshot = SimpleNamespace(density_fitted=False, check_current=lambda: None)
    with pytest.raises(ValueError, match="fitted snapshot"):
        NativeKsSnapshot.stationary_integral_device_reserve(
            snapshot, atoms=3, aos=24, primitives=22
        )


def test_missing_native_proof_never_guesses_a_fitted_provider() -> None:
    snapshot, _ = _snapshot()
    snapshot._library = SimpleNamespace()
    with pytest.raises(NotImplementedError, match="provider provenance"):
        NativeKsSnapshot.fock_provider_proof(snapshot)


@pytest.mark.parametrize("exchange", (0.0, 0.25))
def test_exact_only_legacy_proof_preserves_exchange_presence(exchange: float) -> None:
    snapshot, _ = _snapshot(fitted=False)
    snapshot._library = SimpleNamespace()
    snapshot.coefficients = (1.0, 1.0, exchange)
    assert NativeKsSnapshot.fock_provider_proof(snapshot) == (
        "exact",
        "exact" if exchange else None,
        0.0,
    )


def _assert_decode_admission(
    decode: typing.Callable,
    *,
    mismatches: tuple[str, ...] = (
        "owner",
        "functional",
        "coefficients",
        "spin",
        "basis",
        "grid",
        "requested_fitting",
        "native_fitting",
        "exchange",
    ),
) -> None:
    """Exercise the real decoder with detached native-export records."""
    from test_libxc_snapshot_decode_identity import _decode_fixture

    options, source, basis, grid = _decode_fixture(
        "pbe", "unpolarized", selection="default"
    )
    state = decode(source, basis, grid)
    assert "identity" in state, "decoder bypassed state publication"
    assert state["identity"].functional_identity == options.functional.identity
    assert state["identity"].grid_identity == grid.identity
    assert state["_source"] is source

    class StaleSnapshot(RuntimeError):
        pass

    for mismatch in mismatches:
        _, source, basis, grid = _decode_fixture(
            "pbe", "unpolarized", selection="default"
        )
        if mismatch == "owner":

            def reject_stale() -> None:
                raise StaleSnapshot("revoked native snapshot")

            source.check_current = reject_stale
        elif mismatch == "functional":
            metadata = list(source.metadata)
            metadata[6] += 1
            source.metadata = tuple(metadata)
        elif mismatch == "coefficients":
            source.values[-3] = 0.5
        elif mismatch == "spin":
            polarized, _, _, _ = _decode_fixture(
                "pbe", "polarized", selection="default"
            )
            source._batch._calculator.ks_options = polarized
        elif mismatch == "basis":
            object.__setattr__(basis, "packed", 2 * basis.packed)
        elif mismatch == "grid":
            grid = type(grid)(grid.points, 2 * grid.weights, grid.owners, {})
        elif mismatch == "requested_fitting":
            source._density_fitting_requested = True
        elif mismatch == "native_fitting":
            source.fock_provider_proof = lambda: ("density-fitted", None, 1e-10)
        else:
            source.fock_provider_proof = lambda: ("exact", "exact", 0.0)
        try:
            decode(source, basis, grid)
        except (ValueError, StaleSnapshot):
            pass
        else:
            raise AssertionError(f"decoder admitted mismatched {mismatch}")


@pytest.mark.parametrize(
    "mismatch",
    (
        "owner",
        "functional",
        "coefficients",
        "spin",
        "basis",
        "grid",
        "requested_fitting",
        "native_fitting",
        "exchange",
    ),
)
def test_snapshot_decode_rejects_mismatched_native_sources(mismatch: str) -> None:
    """Behavioral admission survives source refactors without a body fingerprint."""
    _assert_decode_admission(NativeKsSnapshot.decode, mismatches=(mismatch,))


@pytest.mark.parametrize(
    "coulomb,exchange", [(0, None), (1, None), (0, "density-fitted")]
)
def test_fitted_fallback_policy_uses_native_proof(
    coulomb: int, exchange: str | None
) -> None:
    proof = ("density-fitted" if coulomb == 1 else "exact", exchange, 1e-10)
    snapshot = SimpleNamespace(fock_provider_proof=lambda: proof)
    assert NativeKsSnapshot.density_fitted.fget(snapshot) == (
        "density-fitted" in proof[:2]
    )


def test_snapshot_api_compiles_with_direct_capability_dependencies(
    tmp_path: Path, native_cxx: typing.Any
) -> None:
    """Compile the real CPU translation unit, without transitive-header stubs."""
    from tools.generate_libxc_semilocal_cpu_registry import emit_header

    root = Path(__file__).resolve().parents[2]
    # Use the production generator for this build-owned declaration dependency;
    # keep the probe independent of an existing CMake build or handwritten stubs.
    registry = tmp_path / "libxc_semilocal_cpu/generated_libxc_semilocal_registry.hpp"
    registry.parent.mkdir()
    registry.write_text(emit_header(), encoding="utf-8")
    native_cxx.compile_object(
        root / "src/api/c_api_ks_snapshot.cpp",
        tmp_path / "ks_snapshot.o",
        args=(
            "-std=c++20",
            "-DGENERATIVEQC_HAS_CUDA=0",
            "-DGENERATIVEQC_HAS_OPENBLAS=0",
            "-DGENERATIVEQC_CUDA_PROVIDER_CUMETAL=0",
            f"-I{root / 'include'}",
            f"-I{root / 'src'}",
            f"-I{root / 'src/xtb/native'}",
            f"-I{root / 'src/xtb/native/src'}",
            f"-I{tmp_path}",
        ),
    )
