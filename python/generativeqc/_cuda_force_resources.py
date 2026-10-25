"""Bounded ordinary CUDA force policy shared by execution and inventories."""

CUDA_FORCE_DEVICE_CAP = 512 << 20
CUDA_FORCE_HOST_CAP = 256 << 20
DF_FORCE_DEVICE_CAP = 1 << 30
DF_FORCE_HOST_CAP = 512 << 20
DF_FORCE_MIN_ATOMS = 48


def ordinary_cuda_force_limits(
    atoms: int, *, density_fitted: bool, hamiltonian: str
) -> tuple[int, int]:
    """Return device/host caps before choosing a tile or allocating scratch.

    Large all-electron DF owners reserve enough for a larger AO tile and the
    phased Becke cache together. Explicit diagnostic caps remain independent.
    Direct and ECP owners keep their existing inventory-backed reservations.
    """
    if density_fitted and hamiltonian == "all-electron" and atoms >= DF_FORCE_MIN_ATOMS:
        return DF_FORCE_DEVICE_CAP, DF_FORCE_HOST_CAP
    return CUDA_FORCE_DEVICE_CAP, CUDA_FORCE_HOST_CAP
