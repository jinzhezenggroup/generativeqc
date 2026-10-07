"""Gaussian occupied-count bindings for the shared weighted-Gram lowerer."""

from .optimize import prepare_for_backend
from .scf import density_program, weighted_density_program
from .weighted_gram import WeightedGram, recognize_weighted_gram, template_hash
from .weighted_gram_emit import emit_occupied


def _region(weighted: bool) -> WeightedGram:
    builder = weighted_density_program if weighted else density_program
    output = "weighted_density" if weighted else "density"
    # occupied[] selects a virtual weight prefix of the complete n-by-n
    # eigenframe; it does not shorten the physical orbital/state stride.
    return recognize_weighted_gram(
        prepare_for_backend(builder(1, 3, spin_count=2, orbital_count=3), "cuda"),
        output,
    )


def density_template_hash() -> str:
    return template_hash(_region(False).adapter.program)


def weighted_density_template_hash() -> str:
    return template_hash(_region(True).adapter.program)


def emit_density_cuda() -> str:
    return emit_occupied(_region(False), _region(True), backend="cuda")
