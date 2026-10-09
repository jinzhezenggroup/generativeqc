"""Adopt the issue-specific OH spatial-equivalence validation policy.

The hash-bound certificate/capture tools remain unchanged so retained evidence
keeps its original provenance. This layer evaluates current policy from the
actual tensors, never from a saved diagnostic's claimed status.
"""

from __future__ import annotations

from typing import Any

from tools.oh_uhf_symmetry import certify_oh

POLICY_ID = "oh-sto3g-axial-spatial-equivalence-v1"


def evaluate_oh_density(**certificate_inputs: Any) -> dict:
    """Evaluate the frozen six-AO OH domain without requiring a canonical gauge.

    Inputs are the exact keyword arguments of :func:`certify_oh`; callers must
    bind independent Hamiltonian tensors and primary densities to the bundled
    primitives, spin populations and admitted original/moved geometry using
    the capture runner's provenance checks. Unsupported inputs raise rather
    than falling back to an unconstrained orbital alignment.

    ``equivalence_decision`` accepts only a recomputed full spatial certificate.
    The independent raw result retains its absolute 1e-7 gate, including FAIL.
    This is a density-comparison decision, not an energy/force, convergence,
    stability, performance or general endpoint acceptance decision. Production
    densities and historical receipts are never modified.
    """
    certificate = certify_oh(**certificate_inputs)
    return {
        "policy_id": POLICY_ID,
        "canonical_determinant_required": False,
        "equivalence_decision": (
            "ACCEPTED_EQUIVALENT"
            if certificate["symmetry"]["status"] == "PASS"
            else "REJECTED"
        ),
        "raw": certificate["raw"],
        "symmetry": certificate["symmetry"],
        "production_density_modified": certificate["production_density_modified"],
    }
