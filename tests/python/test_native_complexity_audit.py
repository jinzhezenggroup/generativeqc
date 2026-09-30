"""Regression coverage for the native high-order source audit."""

from pathlib import Path

from tools.audit_native_complexity import (
    DEFAULT_EXCLUDES,
    audit_text,
    audit_tree,
)

ROOT = Path(__file__).resolve().parents[2]


def test_audit_detects_quartic_rank2_matrix_chain() -> None:
    source = r"""
void transform(std::size_t n, const double* c, const double* a, double* out) {
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu)
          out[p * n + q] += c[mu * n + p] * a[mu * n + nu] * c[nu * n + q];
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "matrix-chain-candidate"
    assert findings[0].lhs == "out[p * n + q]"


def test_audit_does_not_call_true_four_index_source_reducible() -> None:
    source = r"""
void contract(std::size_t n, const double* eri, const double* d, double* fock) {
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s) {
          const auto quartet = ((p * n + q) * n + r) * n + s;
          fock[p * n + q] += eri[quartet] * d[r * n + s];
        }
}
"""
    findings = audit_text(source, path="synthetic.cpp")
    assert len(findings) == 1
    assert findings[0].classification == "high-order-loop"


def test_production_has_no_avoidable_rank2_quartic_scalar_reduction() -> None:
    scanned, findings = audit_tree(ROOT / "src", excludes=DEFAULT_EXCLUDES)
    assert scanned > 0
    candidates = [
        finding for finding in findings if finding.classification == "matrix-chain-candidate"
    ]
    assert not candidates, (
        "production source reintroduced avoidable high-order rank-2 contractions: "
        + ", ".join(f"{item.path}:{item.line}" for item in candidates)
    )
