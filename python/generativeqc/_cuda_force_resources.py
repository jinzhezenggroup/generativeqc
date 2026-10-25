"""Bounded ordinary CUDA force policy shared by execution and inventories."""

CUDA_FORCE_DEVICE_CAP = 512 << 20
CUDA_FORCE_HOST_CAP = 256 << 20
DF_FORCE_DEVICE_CAP = 1 << 30
DF_FORCE_HOST_CAP = 512 << 20
DF_FORCE_MIN_ATOMS = 48
DIRECT_FORCE_MIN_ATOMS = 96


def ordinary_cuda_force_limits(
    atoms: int, *, density_fitted: bool, hamiltonian: str
) -> tuple[int, int]:
    """Return device/host caps before choosing a tile or allocating scratch.

    Large all-electron owners reserve enough for a larger AO tile and the
    phased Becke cache together. Direct's crossover is later: small Direct
    owners already admit useful tiles under the original limits. The Direct
    resource inventory uses this same policy for its retained force allowance.
    Explicit diagnostic caps remain independent; ECP keeps its existing limits.
    """
    minimum_atoms = DF_FORCE_MIN_ATOMS if density_fitted else DIRECT_FORCE_MIN_ATOMS
    if hamiltonian == "all-electron" and atoms >= minimum_atoms:
        return DF_FORCE_DEVICE_CAP, DF_FORCE_HOST_CAP
    return CUDA_FORCE_DEVICE_CAP, CUDA_FORCE_HOST_CAP
