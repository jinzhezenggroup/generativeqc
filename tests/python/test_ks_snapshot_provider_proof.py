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


def test_capacity_audit_rejects_bypassed_native_provider_proof(tmp_path: Path) -> None:
    from tools.dft_mp_v1 import qualify_capacity

    relative = "python/generativeqc/_ks_snapshot.py"
    source = (Path(__file__).resolve().parents[2] / relative).read_text()
    assert source.count("self.fock_provider_proof()") == 2
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_text(
        source.replace("self.fock_provider_proof()", '("exact", None, 0.0)')
    )
    with pytest.raises(RuntimeError, match="snapshot functional contract changed"):
        qualify_capacity._snapshot_functional_contract(tmp_path)


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
