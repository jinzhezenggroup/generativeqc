"""Array execution of the shared nuclear Coulomb program and atom incidence."""

from generativeqc_compiler.common.array_execution import arrays as np
from generativeqc_compiler.common.array_graph import evaluate_array_graph
from generativeqc_compiler.method.nuclear_repulsion import nuclear_pair_program


def nuclear_repulsion_hvp(
    centers: np.ndarray, charges: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Apply generated pair Hessians with equal-and-opposite physical scatter."""
    first, second = np.triu_indices(len(centers), 1)
    result = np.zeros_like(centers)
    if not len(first):
        return result
    delta = centers[first] - centers[second]
    moving = direction[first] - direction[second]
    distances = np.linalg.norm(delta, axis=1)
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0):
        raise ValueError("nuclear Hessian requires distinct finite centers")
    graph, roots = nuclear_pair_program()
    variables = {
        "charge_product": charges[first] * charges[second],
        **{f"r{axis}": delta[:, axis] for axis in range(3)},
        **{f"dr{axis}": moving[:, axis] for axis in range(3)},
    }
    values = np.asarray(evaluate_array_graph(graph, roots[4:], variables)).T
    np.add.at(result, first, values)
    np.add.at(result, second, -values)
    return result
