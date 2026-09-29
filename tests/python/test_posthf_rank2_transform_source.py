"""Structural guard for #1576 rank-2 AO/MO transforms."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _function_body(relative: str, signature: str) -> str:
    text = (ROOT / relative).read_text()
    start = text.index(signature)
    opening = text.index("{", start)
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[opening : index + 1]
    raise AssertionError(f"unbalanced function body for {relative}:{signature}")


def test_production_rank2_transforms_use_shared_cubic_primitive():
    cases = [
        (
            "src/posthf/mp2_derivative_common.cpp",
            "std::vector<double> pullback_matrix(",
            ("for (std::size_t u =", "for (std::size_t v =", "for (std::size_t p =", "for (std::size_t q ="),
        ),
        (
            "src/posthf/mp2_force.cpp",
            "std::vector<double> hcore_mo(",
            ("for (std::size_t mu =", "for (std::size_t nu =", "result[p * n + q] +="),
        ),
        (
            "src/cc/rccsdt_force.cpp",
            "RawHamiltonian raw_hamiltonian(",
            ("for (std::size_t mu =", "for (std::size_t nu =", "out.h[p * n + q] +="),
        ),
    ]
    for relative, signature, forbidden in cases:
        body = _function_body(relative, signature)
        assert "tensor::cpu_congruence(" in body
        for token in forbidden:
            assert token not in body, f"{relative} reintroduced quartic rank-2 transform: {token}"
