"""Exercise real cold/prepared construction before topology or device work."""

from pathlib import Path

import pytest
from _stationary_artifact_test_support import stationary_route
from generativeqc import _stationary_cuda as runtime


@pytest.mark.parametrize("entrypoint", ("cold", "prepared"))
@pytest.mark.parametrize("profile_device", (False, True))
@pytest.mark.parametrize("integral_derivatives", (False, True))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_constructor_receives_requested_profile_and_admitted_derivatives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entrypoint: str,
    profile_device: bool,
    integral_derivatives: bool,
    spin: str,
) -> None:
    class ConstructionReached(Exception):
        pass

    probe = stationary_route(
        monkeypatch,
        tmp_path,
        entrypoint,
        native_integrals=not integral_derivatives,
        spin=spin,
    )
    artifact = object()
    monkeypatch.setattr(runtime, "compile_stationary_cuda", lambda *a, **k: artifact)

    def construct(
        basis: object,
        selected_artifact: object,
        compiler: object,
        device: int,
        points: int,
        records: int,
        budget: int,
        *,
        profile_device: bool = False,
        integral_derivatives: bool = True,
        spin_blocks: int = 1,
        **kwargs: object,
    ) -> None:
        assert selected_artifact is artifact
        captured.update(
            profile=profile_device, derivatives=integral_derivatives, spins=spin_blocks
        )
        raise ConstructionReached

    captured = {}
    monkeypatch.setattr(runtime, "_CudaSources", construct)
    with pytest.raises(ConstructionReached):
        probe.run(aot=False, profile_device=profile_device)
    assert captured == {
        "profile": profile_device,
        "derivatives": integral_derivatives,
        "spins": 2 if spin == "polarized" else 1,
    }
    # Source-construction failure must release the borrowed grid owner.
    assert probe.grid_closed
