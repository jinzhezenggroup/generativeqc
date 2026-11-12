"""Host-only contracts for physical ECP adapters, not numerical qualification."""

import typing
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_ecp_halogen_dft as halogen
import test_ecp_stuttgart_dft as stuttgart


@pytest.mark.parametrize(
    ("family", "symbol"),
    [(halogen, "Br"), (halogen, "I"), (stuttgart, "Na"), (stuttgart, "K")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_physical_force_gate_dispatch(
    family: typing.Any,
    symbol: str,
    device: str,
    representation: str,
    method: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run the actual family and public gate entrypoints up to native execution."""
    serialized = []

    @dataclass(frozen=True)
    class Basis:
        elements: tuple[SimpleNamespace, ...]
        representation: str = "spherical"

        def write(self, path: Path) -> None:
            serialized.append((path, self))

    basis = Basis((SimpleNamespace(shells=(SimpleNamespace(angular_momentum=1),)),))
    spin = int(method.endswith("uks"))
    mol = SimpleNamespace(
        spin=0,
        charge=0,
        cart=False,
        nao_nr=lambda: 5,
        build=lambda *args: None,
    )
    measurements = {}

    def physical_fixture(selected_symbol: str, *, spin: int = 0) -> typing.Any:
        assert selected_symbol == symbol
        assert spin == (0 if family is halogen else int(method.endswith("uks")))
        mol.spin = mol.charge = spin
        return [(symbol, (0, 0, 0)), ("H", (0, 0, 3))], basis, mol

    class EndpointReached(Exception):
        pass

    def calculator(record: typing.Any, selected_method: str) -> typing.NoReturn:
        assert selected_method == method
        expected = replace(basis, representation=representation)
        if representation == "spherical":
            assert record == tmp_path / "spherical-ecp.json"
            assert serialized == [(record, expected)]
        else:
            assert record == expected
            assert not serialized
        assert mol.cart == (representation == "cartesian")
        assert mol.spin == spin
        assert mol.charge == (-spin if family is halogen else spin)
        raise EndpointReached

    fixture_name = "heavy_fixture" if family is halogen else "stuttgart_fixture"
    monkeypatch.setattr(family, fixture_name, physical_fixture)
    gates = family.cpu_gates if device == "cpu" else family.cuda_gates
    monkeypatch.setattr(gates, "calculator", calculator)
    entrypoint = (
        family.test_halogen_complete_force_analytic_and_fd
        if family is halogen
        else family.test_stuttgart_complete_force_analytic_and_fd
    )
    with pytest.raises(EndpointReached):
        entrypoint(
            device,
            symbol,
            representation,
            method,
            monkeypatch,
            measurements.__setitem__,
            tmp_path,
        )
    assert measurements["parameter_sha256"] == family.PARAMETER_SHA256[symbol]
    assert measurements["effective_charges"] == (
        [halogen.VALENCE_CHARGES[symbol], 1] if family is halogen else [1, 1]
    )
    assert measurements["nao"] == 5
    if family is halogen:
        assert measurements["qualification_charge"] == -spin
        assert measurements["qualification_multiplicity"] == spin + 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
@pytest.mark.parametrize("return_mean_field", [False, True])
def test_halogen_reference_preserves_return_mode_and_independent_guess(
    device: str,
    method: str,
    return_mean_field: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gates = halogen.cpu_gates if device == "cpu" else halogen.cuda_gates
    molecule, state, solver = object(), object(), object()
    result = (-1.0, object(), solver) if return_mean_field else (-1.0, object())
    measurements = {}

    def reference(
        mol: typing.Any,
        stationary: typing.Any,
        selected_method: str,
        *,
        initial_guess: str = "1e",
        maximum_cycles: int = 150,
        return_mean_field: bool = False,
    ) -> typing.Any:
        assert mol is molecule and stationary is state and selected_method == method
        assert return_mean_field == (len(result) == 3)
        assert initial_guess == ("atom" if method.endswith("uks") else "1e")
        assert maximum_cycles == (600 if method.endswith("uks") else 150)
        return result

    monkeypatch.setattr(gates, "reference", reference)
    selected = halogen.select_fixture(
        "Br", device, monkeypatch, measurements.__setitem__
    )
    assert (
        selected.reference(molecule, state, method, return_mean_field=return_mean_field)
        is result
    )
    if method.endswith("uks"):
        assert measurements["independent_initial_guess"] == "atom"
        assert measurements["independent_maximum_cycles"] == 600
    else:
        assert not measurements


@pytest.mark.parametrize(("family", "symbol"), [(halogen, "Br"), (stuttgart, "Na")])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "option", ["d_shell", "f_shell", "f_on_h", "compact_d_oracle", "compact_f_oracle"]
)
def test_physical_adapter_rejects_synthetic_fixture_variants(
    family: typing.Any,
    symbol: str,
    device: str,
    option: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_fixture(*args: typing.Any, **kwargs: typing.Any) -> typing.NoReturn:
        pytest.fail("unsupported variants must not select a physical s/p fixture")

    fixture_name = "heavy_fixture" if family is halogen else "stuttgart_fixture"
    monkeypatch.setattr(family, fixture_name, unexpected_fixture)
    gates = family.select_fixture(symbol, device, monkeypatch, lambda *args: None)
    with pytest.raises(AssertionError):
        gates.fixture(**{option: True})
