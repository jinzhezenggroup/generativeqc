"""Exercise snapshot decoding with typed methods and a host-only wire fixture."""

from types import SimpleNamespace

import numpy as np
import pytest
from generativeqc._dft_gradient import scf_regularization_identity
from generativeqc._ks_snapshot import NativeKsSnapshot
from generativeqc.ks import (
    KsOptions,
    _native_semilocal_code,
    native_dft_carrier_for_ir,
    resolve_ks_method,
    resolve_ks_options,
    scf_domain_for_method,
)
from generativeqc_compiler.common.provenance import canonical_hash
from generativeqc_compiler.dft.ao import NativeAO
from generativeqc_compiler.dft.grid import ExplicitGrid, GridSpec
from generativeqc_compiler.method import resolve_bulk_ks
from generativeqc_compiler.xc._generated_native_semilocal import SCF_DOMAIN_BY_VERSION
from generativeqc_compiler.xc.libxc_work import LIBXC_WORK_DOMAIN_VERSION


def _decode_fixture(name: str, spin: str, *, selection: str = "automatic") -> tuple:
    if selection == "automatic":
        method_ir = resolve_bulk_ks(name, spin=spin, backend="cpu").method
        carrier = native_dft_carrier_for_ir(method_ir)
    else:
        carrier = f"{name}-{'uks' if spin == 'polarized' else 'rks'}"
        method_ir, _ = resolve_ks_method(carrier)
    spec = GridSpec(radial_points=12, angular_polar=6, angular_azimuth=12)
    options = resolve_ks_options(
        carrier,
        KsOptions(composition=None if selection == "default" else method_ir, grid=spec),
    )
    domain_version = (
        LIBXC_WORK_DOMAIN_VERSION
        if selection == "automatic"
        else next(
            version
            for version, domain in SCF_DOMAIN_BY_VERSION.items()
            if domain == options.scf_domain
        )
    )
    spins = 2 if spin == "polarized" else 1
    occupations = np.array([1.0, 0.0] if spins == 2 else [2.0])
    charge = 2 - occupations.sum()

    # Construct only the detached AO records consumed by decode. No native
    # handle, SCF, AO evaluation, or claimed molecular qualification is involved.
    basis = NativeAO.__new__(NativeAO)
    basis.atoms = (SimpleNamespace(atomic_number=2, position=(0.0, 0.0, 0.0)),)
    basis.shells = (SimpleNamespace(angular_momentum=0, atom_index=0),)
    basis.nao = basis.natom = 1
    basis.packed = np.array([1.0])
    basis.charge = charge
    basis.multiplicity = spins
    basis.representation = "cartesian"
    basis.identity = "host-wire-fixture-basis"
    grid = ExplicitGrid([[0.25, 0.0, 0.0]], [1.0], (0,), {"test": "wire"})
    radii = np.zeros(119)
    for z, radius in spec.element_radii:
        radii[z] = radius
    values = np.concatenate(
        [
            [0.0, charge],
            [2.0, 0.0, 0.0, 0.0],
            occupations,  # density
            -np.ones(spins),  # Fock
            np.ones(spins),  # orbital coefficients
            -np.ones(spins),  # orbital energies
            occupations,
            -occupations,  # energy-weighted density
            [1.0],  # overlap
            basis.packed,
            grid.points.ravel(),
            grid.weights,
            grid.owners,
            [
                spec.version,
                spec.radial_points,
                spec.angular_polar,
                spec.angular_azimuth,
                spec.partition_iterations,
                spec.coincident_tolerance,
            ],
            radii,
            [1.0],  # atomic weight
            options.coefficients,
        ]
    )
    metadata = (
        6,
        1,
        spins,
        1,
        len(basis.packed),
        len(grid.points),
        _native_semilocal_code(method_ir),
        domain_version,
        1,
        1,
        1,
        1,
        2**64 - 1,
        0,
        spins,
        len(values),
    )
    source = SimpleNamespace(
        metadata=metadata,
        values=values,
        backend="cpu",
        _library=SimpleNamespace(),
        _batch=SimpleNamespace(
            _calculator=SimpleNamespace(ks_options=options, _method_name=carrier),
            _snapshot_grid_cache=None,
        ),
        _density_fitting_requested=False,
        check_current=lambda: None,
        fock_provider_proof=lambda: ("exact", None, 0.0),
    )
    return options, source, basis, grid


@pytest.mark.parametrize("name", ("LDA_C_VWN_4", "GGA_C_APBE", "MGGA_X_LTA"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_automatic_snapshot_decode_uses_semantic_domain(name: str, spin: str) -> None:
    options, source, basis, grid = _decode_fixture(name, spin)
    state = NativeKsSnapshot.decode(source, basis, grid)
    identity = state["identity"]
    assert source.method_ir.identity == options.method_ir.identity
    assert source.functional_code == _native_semilocal_code(options.method_ir)
    assert identity.functional_identity == options.functional.identity
    assert identity.spin == spin
    expected = canonical_hash({"scf_domain": options.scf_domain})
    carrier_domain = scf_domain_for_method(source._batch._calculator._method_name)
    assert options.scf_domain != carrier_domain
    assert identity.regularization_identity == expected


@pytest.mark.parametrize("name", ("lda", "pbe", "r2scan"))
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
@pytest.mark.parametrize("selection", ("default", "explicit"))
def test_curated_snapshot_decode_retains_regularization_identity(
    name: str, spin: str, selection: str
) -> None:
    options, source, basis, grid = _decode_fixture(name, spin, selection=selection)
    state = NativeKsSnapshot.decode(source, basis, grid)
    expected = canonical_hash({"scf_domain": options.scf_domain})
    assert expected == scf_regularization_identity(
        source._batch._calculator._method_name
    )
    assert state["identity"].regularization_identity == expected
    assert source.method_ir.identity == options.method_ir.identity
    assert source.functional_code == _native_semilocal_code(options.method_ir)


@pytest.mark.parametrize("mismatch", ("functional", "coefficients", "spin", "grid"))
def test_automatic_snapshot_decode_rejects_source_mismatch(mismatch: str) -> None:
    _, source, basis, grid = _decode_fixture("GGA_C_APBE", "unpolarized")
    if mismatch == "functional":
        fields = list(source.metadata)
        fields[6] += 1
        source.metadata = tuple(fields)
    elif mismatch == "coefficients":
        source.values[-3] = 0.5
    elif mismatch == "spin":
        polarized, _, _, _ = _decode_fixture("GGA_C_APBE", "polarized")
        source._batch._calculator.ks_options = polarized
    else:
        grid = ExplicitGrid(
            grid.points, 2 * grid.weights, grid.owners, {"test": "wrong"}
        )
    error = "grid source mismatch" if mismatch == "grid" else "composition mismatch"
    with pytest.raises(ValueError, match=error):
        NativeKsSnapshot.decode(source, basis, grid)


@pytest.mark.parametrize(
    "native_proof,requested_fitting,error",
    (
        (("density-fitted", None, 1e-10), False, "density fitting"),
        (("exact", None, 0.0), True, "density fitting"),
        (("exact", "exact", 0.0), False, "exchange presence"),
    ),
)
def test_snapshot_decode_rejects_forged_native_provider_provenance(
    native_proof: tuple[str, str | None, float],
    requested_fitting: bool,
    error: str,
) -> None:
    """Real decode must reject an inconsistent J/K provider before publication.

    This host-only execution oracle complements the separately retained
    qualification-owner hash without inspecting the decoder's source spelling.
    """
    _, source, basis, grid = _decode_fixture(
        "pbe", "unpolarized", selection="default"
    )
    source.fock_provider_proof = lambda: native_proof
    source._density_fitting_requested = requested_fitting
    with pytest.raises(ValueError, match=error):
        NativeKsSnapshot.decode(source, basis, grid)
