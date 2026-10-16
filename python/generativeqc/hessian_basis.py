"""Fixed normalized Cartesian/public AO maps for stationary nuclear derivatives."""

from __future__ import annotations

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.dft import NativeAO


class HessianBasis:
    """Borrow public AOs and own an optional Cartesian derivative workspace.

    Nuclear motion does not differentiate the angular representation map. The
    map is read from the native normalized AO expansions, rather than inferred
    by fitting grid values or reconstructing a separate spherical convention.
    """

    def __init__(self, basis: NativeAO) -> None:
        self.public = basis
        self.cartesian = basis
        self.transform = None
        if basis.representation == "cartesian":
            return
        if basis.representation != "real_spherical":
            raise ValueError("unknown Hessian AO representation")
        self.cartesian = NativeAO(
            basis.atoms,
            basis=basis.shells,
            charge=basis.charge,
            multiplicity=basis.multiplicity,
        )
        try:
            transform = np.zeros((self.cartesian.nao, basis.nao))
            public_records = basis.packed[
                3 * basis.natom + 2 * basis.nprimitive :
            ].reshape(-1, 16)
            cart_records = self.cartesian.packed[
                3 * basis.natom + 2 * self.cartesian.nprimitive :
            ].reshape(-1, 16)
            cart_lookup = {
                (int(record[1]), tuple(map(int, record[4:7]))): (index, record[7])
                for index, record in enumerate(cart_records)
            }
            for column, record in enumerate(public_records):
                for term in range(int(record[3])):
                    offset = 4 + 4 * term
                    key = (int(record[1]), tuple(map(int, record[offset : offset + 3])))
                    row, scale = cart_lookup[key]
                    transform[row, column] += record[offset + 3] / scale
            self.transform = immutable(transform)
        except Exception:
            self.close()
            raise

    def density(self, value: np.ndarray) -> np.ndarray:
        """Pull public AO cotangents back without changing their spin axis."""
        if self.transform is None:
            return value
        return self.transform @ value @ self.transform.T

    def matrix(self, value: np.ndarray) -> np.ndarray:
        """Project Cartesian operators into the original variational space."""
        if self.transform is None:
            return value
        return self.transform.T @ value @ self.transform

    def close(self) -> None:
        """Release the owned workspace, never the caller's public AO owner."""
        if self.cartesian is not self.public:
            self.cartesian.close()
