"""Build-only constants for the incumbent residual-ECP two-grid rule.

Integral quadrature stays in the integral compiler; the method layer consumes
only fixed-measure AO bilinears and does not import integral policy.
"""

from __future__ import annotations

import numpy as np

from generativeqc_compiler.common.array_graph import evaluate_array_graph

from .ecp_grid import harmonic_roots, radial_map_roots
from .ecp_policy import (
    COARSE_POLAR_POINTS,
    COARSE_RADIAL_POINTS,
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)


def ecp_fixed_quadratures() -> tuple[dict[str, np.ndarray], ...]:
    """Freeze the existing graph-owned radial map/harmonics during the build.

    NumPy supplies fixed Legendre constants only during source generation.
    Runtime geometry, AO evaluation and electronic contractions stay on CUDA.
    """
    result = []
    for radial, polar in (
        (COARSE_RADIAL_POINTS, COARSE_POLAR_POINTS),
        (REFINED_RADIAL_POINTS, REFINED_POLAR_POINTS),
    ):
        nodes, weights = np.polynomial.legendre.leggauss(radial)
        graph, roots = radial_map_roots()
        radius, measure = evaluate_array_graph(
            graph, roots, {"z": nodes, "weight": weights}
        )
        nodes, weights = np.polynomial.legendre.leggauss(polar)
        height, azimuth = np.broadcast_arrays(
            nodes[:, None], np.arange(2 * polar)[None, :] * np.pi / polar
        )
        sphere = np.stack(
            (
                np.sqrt(1 - height**2) * np.cos(azimuth),
                np.sqrt(1 - height**2) * np.sin(azimuth),
                height,
            ),
            axis=-1,
        ).reshape(-1, 3)
        angular = np.broadcast_to(weights[:, None] * np.pi / polar, height.shape)
        graph, roots = harmonic_roots()
        harmonics = np.array(
            [
                np.broadcast_to(value, (len(sphere),))
                for value in evaluate_array_graph(
                    graph, roots, dict(zip("xyz", sphere.T, strict=True))
                )
            ]
        )
        result.append(
            {
                "radial": np.stack((radius, measure), axis=1),
                "sphere": sphere,
                "angular": angular.ravel(),
                "projector": harmonics * angular.ravel(),
            }
        )
    return tuple(result)
